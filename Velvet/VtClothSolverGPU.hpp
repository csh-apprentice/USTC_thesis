#pragma once

#include <iostream>


#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "helper_cuda.h"
#include "Mesh.hpp"
#include "VtClothSolverGPU.cuh"
#include "VtBuffer.hpp"
#include "SpatialHashGPU.hpp"
#include "MouseGrabber.hpp"

#include <fstream>
#include <direct.h>
#include <filesystem>

#include <numeric>




using namespace std;

namespace Velvet
{
	class VtClothSolverGPU : public Component
	{
	public:

		void Start() override
		{
			Global::simParams.numParticles = 0;
			m_colliders = Global::game->FindComponents<Collider>();
			m_mouseGrabber.Initialize(&positions, &velocities, &invMasses);
			//ShowDebugGUI();
		}

		void Update() override
		{
			m_mouseGrabber.HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			m_mouseGrabber.UpdateGrappedVertex();
			UpdateColliders(m_colliders);

			Timer::StartTimer("GPU_TIME");
			Simulate();
			//Simulate_Gauss();
			Timer::EndTimer("GPU_TIME");
		}

		void OnDestroy() override
		{
			positions.destroy();
			normals.destroy();
		}

		void Simulate()
		{
			//cout << "the size of the cluster is " << cluster_size << endl;
			Timer::StartTimerGPU("Solver_Total");
			//==========================
			// Prepare
			//==========================
			float frameTime = Timer::fixedDeltaTime();
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;
			float w_k1 = 1.0f;  // Jacobi simulation parameter
			float w_k2 = 1.0f;  // Jacobi simulation parameter

			//==========================
			//validation
			//==========================
			//float *residual_avg =new float;
			//*residual_avg = 0;

			//==========================
			// Launch kernel
			//==========================
			SetSimulationParams(&Global::simParams);

			// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
			// This can produce unstable behavior, such as vertex flashing between two sides.
			// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
			CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);
			//std::cout << "the size of the lambda is " << Lambdas.size();
			//std::cout << "lambda 100 is" << Lambdas[100];
			
			LambdaInit(Lambdas, (uint)stretchLengths.size());
			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{
				
				PredictPositions(current,predicted, velocities, positions, substepTime);
				
				
				if (Global::simParams.enableSelfCollision)
				{
					if (substep % Global::simParams.interleavedHash == 0)
					{
						m_spatialHash->Hash(predicted);
					}
					CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
				}
				CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

				//LambdaInit(Lambdas, (uint)stretchLengths.size());

				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					
					//SolveStretch(predicted, deltas, deltaCounts, Lambdas, stretchIndices, stretchLengths, invMasses, substepTime, (uint)stretchLengths.size());
					SolveStretch(predicted, deltas, deltaCounts, Lambdas,deltaLambdas,deltaLambdasCounts,stretchIndices, stretchLengths, invMasses, substepTime,(uint)stretchLengths.size());
					SolveAttachment(predicted, deltas, deltaCounts, invMasses,
						attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, (uint)attachParticleIDs.size());
					//SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
					Copy(last, current);
					Copy(current, predicted);

					ApplyDeltas(predicted, deltas, deltaCounts);
					
					
					if (iteration < 1)
					{
						w_k1 = 1.0;
						w_k2 = 2 / (2 - (Global::simParams.under_relax_coeff * Global::simParams.under_relax_coeff));
					}
					else if (iteration == 1)
					{
						w_k1 = 2 / (2 - (Global::simParams.under_relax_coeff * Global::simParams.under_relax_coeff));
						w_k2 = 4 / (4 - (Global::simParams.under_relax_coeff * Global::simParams.under_relax_coeff) * w_k1);
					}
						
					else
					{
						w_k1 = 4 / (4 - (Global::simParams.under_relax_coeff * Global::simParams.under_relax_coeff) * w_k1);
						w_k2= 4 / (4 - (Global::simParams.under_relax_coeff * Global::simParams.under_relax_coeff) * w_k1);
					}

					if (iteration > 0)
						Chebyshev_PBD(predicted, current, last, w_k1, w_k2);
					
					
					
					
					
					


					//ApplyLambdaDeltas(Lambdas, deltaLambdas, deltaLambdasCounts, (uint)stretchLengths.size());
				//std:cout << Lambdas << std::endl;
					
				}
				
				Finalize(velocities, positions, predicted, substepTime);
				
			}

			ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));
			Residual_Avg(residual_list, positions, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
			//Constraint_Avg(constraint_list, Lambdas, positions, stretchIndices, stretchLengths, invMasses, substepTime, (uint)stretchLengths.size());

			//==========================
			// Sync
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();

			positions.sync();
			normals.sync();
			float *residual_data = residual_list.data();
			resiual_strain.push_back(accumulate(residual_data, residual_data + residual_list.size(), 0.0));
			//float* constraint_data = constraint_list.data();
			//constraint_strain.push_back(accumulate(constraint_data, constraint_data + constraint_list.size(), 0.0));

			
			//residual starin visualization
			/*
			if (resiual_strain.size() == 600)
			{
				//string filename = "./residual_strain/Cheby_horizontal_hang_zerosdotsix.txt";
				
				string filename = "./residual_strain/XPBD_horizontal_hang.txt";
				std::filesystem::path p(filename);
				//p.parent_path will return the folder path of the file
				std::filesystem::create_directories(p.parent_path());
				ofstream f(filename); 
				for (int j = 0; j < resiual_strain.size(); ++j) {
					f<< resiual_strain[j] << " ";
				}
				fmt::print("(Info(VtClothSolverGPU.hpp)): Residual error in {} steps is stored in {} \n", resiual_strain.size(),filename);
				
			}
			*/
			
			/*
			if (constraint_strain.size() == 600)
			{
				string filename = "./constraint_strain/XPBD_horizontal_hang_4.txt";

				//string filename = "./constraint_strain/PBD_horizontal_hang.txt";
				std::filesystem::path p(filename);
				//p.parent_path will return the folder path of the file
				std::filesystem::create_directories(p.parent_path());
				ofstream f(filename);
				for (int j = 0; j < constraint_strain.size(); ++j) {
					f << constraint_strain[j] << " ";
				}
				fmt::print("(Info(VtClothSolverGPU.hpp)): Residual error in {} steps is stored in {} \n", constraint_strain.size(), filename);

			}
			*/
		}

		void Simulate_Gauss()
		{
			//cout << "the size of the cluster is " << cluster_size << endl;
			//cout << "the size of the stretch is " << stretchLengths.size() << endl;
			Timer::StartTimerGPU("Solver_Total");
			//==========================
			// Prepare
			//==========================
			float frameTime = Timer::fixedDeltaTime();
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;

			//==========================
			// Launch kernel
			//==========================
			SetSimulationParams(&Global::simParams);

			// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
			// This can produce unstable behavior, such as vertex flashing between two sides.
			// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
			CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);
			//std::cout << "the size of the lambda is " << Lambdas.size();
			//std::cout << "lambda 100 is" << Lambdas[100];


			for (int substep = 0; substep < Global::simParams.numSubsteps; substep++)
			{

				PredictPositions(current, predicted, velocities, positions, substepTime);


				if (Global::simParams.enableSelfCollision)
				{
					if (substep % Global::simParams.interleavedHash == 0)
					{
						m_spatialHash->Hash(predicted);
					}
					CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
				}
				CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

				LambdaInit(Lambdas, (uint)stretchLengths.size());

				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{
					int cluster_offset = 0;
				    //SolveStretch(predicted, deltas, deltaCounts, Lambdas, deltaLambdas, deltaLambdasCounts, stretchIndices, stretchLengths, invMasses, substepTime, cluster[0]+cluster[1]+cluster[2]+cluster[3]);
					//SolveStretch(predicted, deltas, deltaCounts, Lambdas, deltaLambdas, deltaLambdasCounts, stretchIndices, stretchLengths, invMasses, substepTime, cluster_size);
					for (int cluster_index = 0; cluster_index < cluster_size; cluster_index++)  //修改cluster对应线程
					{
						
						SolveStretch_Gauss(cluster_offset, predicted,  Lambdas, stretchIndices, stretchLengths, invMasses, substepTime, cluster[cluster_index]);
						//SolveStretch_Gauss(cluster_offset, predicted, deltas, Lambdas, stretchIndices, stretchLengths, invMasses, substepTime, (uint)stretchLengths.size(), cluster_size);
						cluster_offset += cluster[cluster_index];
						//cudaDeviceSynchronize();
					}
					//cout << "the size of the cluster is " << cluster_offset << endl;
					cluster_offset = 0;
					for (int cluster_index = 0; cluster_index < attachSlotPositions.size(); cluster_index++)  //修改cluster对应线程
					{
						SolveAttachment_Gauss(cluster_offset, predicted, invMasses, attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, predicted.size());
						cluster_offset += predicted.size();
						
						//SolveStretch_Gauss(cluster_offset, predicted, deltas, Lambdas, stretchIndices, stretchLengths, invMasses, substepTime, cluster[cluster_index], cluster_size);
						
					}
					//
					
					//ApplyDeltas(predicted, deltas, deltaCounts);
					//ApplyLambdaDeltas(Lambdas, deltaLambdas, deltaLambdasCounts, (uint)stretchLengths.size());

				}

				Finalize(velocities, positions, predicted, substepTime);
			}

			ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));
			//Residual_Avg(residual_list, positions, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
			Constraint_Avg(constraint_list, Lambdas, positions, stretchIndices, stretchLengths, invMasses, substepTime, (uint)stretchLengths.size());

			//==========================
			// Sync
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();

			positions.sync();
			normals.sync();
			
			//float* residual_data = residual_list.data();
			//resiual_strain.push_back(accumulate(residual_data, residual_data + residual_list.size(), 0.0));
			float* constraint_data = constraint_list.data();
			constraint_strain.push_back(accumulate(constraint_data, constraint_data + constraint_list.size(), 0.0));
			if(constraint_strain.size()==1)
				fmt::print("(Info(VtClothSolverGPU.hpp)): The number of constraints is {} \n", constraint_list.size());


			//string name = name;
			/*
			//validation od residual strain
			if (resiual_strain.size() == 600)
			{
				//string filename = "./residual_strain/Cheby_horizontal_hang_zerosdotsix.txt";

				string filename = "./residual_strain/Gauss_horizontal_hang.txt";
				std::filesystem::path p(filename);
				//p.parent_path will return the folder path of the file
				std::filesystem::create_directories(p.parent_path());
				ofstream f(filename);
				for (int j = 0; j < resiual_strain.size(); ++j) {
					f << resiual_strain[j] << " ";
				}
				fmt::print("(Info(VtClothSolverGPU.hpp)): Residual error in {} steps is stored in {} \n", resiual_strain.size(), filename);

			}
			*/
			/*
			if (constraint_strain.size() == 600)
			{
				string filename = "./constraint_strain/Gauss_horizontal_hang.txt";

				//string filename = "./constraint_strain/PBD_horizontal_hang.txt";
				std::filesystem::path p(filename);
				//p.parent_path will return the folder path of the file
				std::filesystem::create_directories(p.parent_path());
				ofstream f(filename);
				for (int j = 0; j < constraint_strain.size(); ++j) {
					f << constraint_strain[j] << " ";
				}
				fmt::print("(Info(VtClothSolverGPU.hpp)): Residual error in {} steps is stored in {} \n", constraint_strain.size(), filename);

			}
			*/
		}
	public:

		int AddCloth(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			Timer::StartTimer("INIT_SOLVER_GPU");

			int prevNumParticles = Global::simParams.numParticles;
			int newParticles = (int)mesh->vertices().size();

			// Set global parameters
			Global::simParams.numParticles += newParticles;
			Global::simParams.particleDiameter = particleDiameter;
			Global::simParams.deltaTime = Timer::fixedDeltaTime();
			Global::simParams.maxSpeed = 2 * particleDiameter / Timer::fixedDeltaTime() * Global::simParams.numSubsteps;

			// Allocate managed buffers
			positions.registerNewBuffer(mesh->verticesVBO());
			normals.registerNewBuffer(mesh->normalsVBO());

			for (int i = 0; i < mesh->indices().size(); i++)
			{
				indices.push_back(mesh->indices()[i] + prevNumParticles);
			}

			velocities.push_back(newParticles, Global::simParams.init_v);


			positions[Global::simParams.offset_index] += Global::simParams.offset;
			//for (int index = 0; index < Global::simParams.offset_index.size(); index++)
				//positions[Global::simParams.offset_index[index]] += Global::simParams.offset[index];

			predicted.push_back(newParticles, glm::vec3(0));
			current.push_back(newParticles, glm::vec3(0));
			last.push_back(newParticles, glm::vec3(0));
			
			deltas.push_back(newParticles, glm::vec3(0));
			deltaCounts.push_back(newParticles, 0);
			deltaLambdasCounts.push_back(newParticles, 0);
			invMasses.push_back(newParticles, 1.0f);



			// Initialize buffer datas
			InitializePositions(positions, prevNumParticles, newParticles, modelMatrix);
			cudaDeviceSynchronize();
			positions.sync();

			// Initialize member variables
			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, Global::simParams.numParticles);
			m_spatialHash->SetInitialPositions(positions);

			double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;
			fmt::print("Info(ClothSolverGPU): AddCloth done. Took time {:.2f} ms\n", time);
			fmt::print("Info(ClothSolverGPU): Use recommond max vel = {}\n", Global::simParams.maxSpeed);

			return prevNumParticles;
		}

		void Applycluster(shared_ptr<Mesh> mesh)
		{
			//std::cout << "in the apply_cluster function, mesh has " << mesh->get_cluster().size() << " clusters " << endl;
			cluster = mesh->get_cluster();
			cluster_size = cluster.size();
		
		}

		void AddLambdas()
		{
			int numConstraints = (uint)stretchLengths.size();
			Lambdas.push_back(numConstraints, 0.0f);
			deltaLambdas.push_back(numConstraints, 0.0f);
		}

		void AddStretch(int idx1, int idx2, float distance)
		{
			stretchIndices.push_back(idx1);
			stretchIndices.push_back(idx2);
			stretchLengths.push_back(distance);
			residual_list.push_back(0.0);
			constraint_list.push_back(0.0);
		}

		void AddAttachSlot(glm::vec3 attachSlotPos)
		{
			attachSlotPositions.push_back(attachSlotPos);
		}

		void AddAttach(int particleIndex, int slotIndex, float distance)
		{
			if (distance == 0) invMasses[particleIndex] = 0;
			attachParticleIDs.push_back(particleIndex);
			attachSlotIDs.push_back(slotIndex);
			attachDistances.push_back(distance);
		}

		void AddBend(uint idx1, uint idx2, uint idx3, uint idx4, float angle)
		{
			bendIndices.push_back(idx1);
			bendIndices.push_back(idx2);
			bendIndices.push_back(idx3);
			bendIndices.push_back(idx4);
			bendAngles.push_back(angle);
		}

		void UpdateColliders(vector<Collider*>& colliders)
		{
			sdfColliders.resize(colliders.size());

			for (int i = 0; i < colliders.size(); i++)
			{
				const Collider* c = colliders[i];
				if (!c->enabled) continue;
				SDFCollider sc;
				sc.type = c->type;
				sc.position = c->actor->transform->position;
				sc.scale = c->actor->transform->scale;
				sc.curTransform = c->curTransform;
				sc.invCurTransform = glm::inverse(c->curTransform);
				sc.lastTransform = c->lastTransform;
				sc.deltaTime = Timer::fixedDeltaTime();
				sdfColliders[i] = sc;
			}
		}

		

	public: // Sim buffers

		VtMergedBuffer<glm::vec3> positions;
		VtMergedBuffer<glm::vec3> normals;
		VtBuffer<uint> indices;

		VtBuffer<glm::vec3> velocities;
		VtBuffer<glm::vec3> predicted;
		VtBuffer<glm::vec3> current;
		VtBuffer<glm::vec3> last;
		VtBuffer<glm::vec3> deltas;
		VtBuffer<int> deltaCounts;
		VtBuffer<float> invMasses;

		//for XPBD
		VtBuffer<float> Lambdas;
		VtBuffer<float> deltaLambdas;
		VtBuffer<int> deltaLambdasCounts;

		VtBuffer<int> stretchIndices;
		VtBuffer<float> stretchLengths;
		VtBuffer<uint> bendIndices;
		VtBuffer<float> bendAngles;

		vector<int> cluster;
		int cluster_size;

       // for testing
		vector<float> resiual_strain;
		VtBuffer<float> residual_list;
		vector<float> constraint_strain;
		VtBuffer<float> constraint_list;



		// Attach attachParticleIndices[i] with attachSlotIndices[i] w
		// where their expected distance is attachDistances[i]
		VtBuffer<int> attachParticleIDs;
		VtBuffer<int> attachSlotIDs;
		VtBuffer<float> attachDistances;
		VtBuffer<glm::vec3> attachSlotPositions;

		VtBuffer<SDFCollider> sdfColliders;

	private:

		shared_ptr<SpatialHashGPU> m_spatialHash;
		vector<Collider*> m_colliders;
		MouseGrabber m_mouseGrabber;

		void ShowDebugGUI()
		{
			GUI::RegisterDebug([this]() {
				{
					static int particleIndex1 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID1", &particleIndex1, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					ImGui::Text(fmt::format("Position: {}", predicted[particleIndex1]).c_str());
					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex1]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex1]);
					ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());
					auto norm = normals[particleIndex1];
					ImGui::Text(fmt::format("Normal: [{:.3f},{:.3f},{:.3f}]", norm.x, norm.y, norm.z).c_str());

					static int neighborRange1 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange1", &neighborRange1, 0, 63);
					ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange1 + particleIndex1 * Global::simParams.maxNumNeighbors]).c_str());
					ImGui::Indent(-10);
				}

				{
					static int particleIndex2 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID2", &particleIndex2, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					ImGui::Text(fmt::format("Position: {}", predicted[particleIndex2]).c_str());
					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex2]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex2]);
					ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());

					static int neighborRange2 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange2", &neighborRange2, 0, 63);
					ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange2 + particleIndex2 * Global::simParams.maxNumNeighbors]).c_str());
					ImGui::Indent(-10);
				}
				static int cellID = 0;
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "CellID", &cellID, 0, (int)m_spatialHash->cellStart.size() - 1);
				int start = m_spatialHash->cellStart[cellID];
				int end = m_spatialHash->cellEnd[cellID];
				ImGui::Indent(10);
				ImGui::Text(fmt::format("CellStart.HashID: {}", start).c_str());
				ImGui::Text(fmt::format("CellEnd.HashID: {}", end).c_str());

				if (start != 0xffffffff && end > start)
				{
					static int particleHash = 0;
					particleHash = clamp(particleHash, start, end - 1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "HashID", &particleHash, start, end - 1);
					ImGui::Text(fmt::format("ParticleHash: {}", m_spatialHash->particleHash[particleHash]).c_str());
					ImGui::Text(fmt::format("ParticleIndex: {}", m_spatialHash->particleIndex[particleHash]).c_str());
				}
				});
		}

	};
}
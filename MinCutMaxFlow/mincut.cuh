// Author: Giovanni Rasera
// Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
// Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
// Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
// Help From: https://arxiv.org/pdf/2404.00270
// Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
// Help From: https://www.adrian-haarbach.de/idp-graph-algorithms/implementation/maxflow-push-relabel/index_en.html
// Help From: https://www.geeksforgeeks.org/push-relabel-algorithm-set-2-implementation/
#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <limits>
#include <limits.h>
#include <iomanip>

// CPU data
// TODO: use CSR rappresentation
typedef std::vector<std::vector<int>> Graph;
typedef std::vector<std::vector<int>> ResidualFlow;
typedef std::vector<int> ExcessFlow;
typedef std::vector<int> Height;
typedef int Excess_total;

// GPU data
typedef int* GPUGraph;
typedef int* GPUResidualFlow;
typedef int* GPUExcessFlow;
typedef int* GPUHeight;

namespace parallel {
    namespace GoldbergTarjan{
        __global__ void push(){

        }

        __global__ void relable(){

        }

        // Initialize the flow
        void preflow(){
            
        }

        /*
        Given: 
            G : G(V,E) the directed graph
            Gf : Gf(V, Ef) the residual graph
            cf : cf(v,u) residual flow on (u, v) 
            e : e(v) the access flow of vertex v
            h : the height of vertex v
            excessTotal : Excess_total the sum of excess flow
        Outputs:
            e(t) : the maximum flow value
            TODO: i need the vertex cut or the edge cut 
            and i can use it to vertex cut the graph
        */ 
        void minCutMaxFlow(Graph &G, Graph &Gf, ExcessFlow &e, Height &h, int source, int to){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
            int N = G.size(); 

            // Initialize
            Excess_total excessTotal = 0;

            // Step 0: Preflow
            ResidualFlow cf; cf.assign(N, std::vector<int>(N, 0));
            preflow();            

            int *dev_Gf, *dev_e, *dev_h;

            // // static memory allocation
            cudaMalloc((void**)&dev_Gf, N * N * sizeof(int));
            cudaMemcpy(dev_Gf, host_Gf, N * N * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_e, N*sizeof(int));
            cudaMalloc((void**)&dev_h, N*sizeof(int));
            cudaMemcpy(dev_e, host_e, N*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_h, host_h, N*sizeof(int), cudaMemcpyHostToDevice);

            // start while
            while(true){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = G.size(); // = |V|

                while(cicle > 0){
		            push<<<1, N>>>();	
                    cudaDeviceSynchronize();
                    
                    cudaMemcpy(host_Gf, dev_Gf, N * N * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(host_e, dev_e, N*sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(host_h, dev_h, N*sizeof(int), cudaMemcpyDeviceToHost);

                    std::cin.ignore(); 
                    cicle--;
                }
            }
        }
    };
};

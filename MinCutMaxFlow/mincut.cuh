// Author: Giovanni Rasera
// Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
// Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
// Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
// Help From: https://arxiv.org/pdf/2404.00270
// Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
// Help From: https://www.adrian-haarbach.de/idp-graph-algorithms/implementation/maxflow-push-relabel/index_en.html
#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <iomanip>

// CPU data
// TODO: use CSR rappresentation
typedef std::vector<std::vector<int>> Graph;
typedef std::vector<std::vector<int>> ResidualFlow;
typedef std::vector<int> ExcessFlow;
typedef std::vector<int> Height;
typedef int Excess_total;

// GPU data
typedef int** GPUGraph;
typedef int* GPUExcessFlow;
typedef int* GPUHeight;

namespace parallel {
    /*
    From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
    Definitions: page 11
        - each node x:
            - has access flow u(x) and height(x)
            - outgoing edges to neighbors(x, *) with capacity c(x, *)
        - node x is active: if u(x) > 0 and h(x) < HEIGHT_MAX
        - active node x
            - can push to neighbor y: if (x, y) > 0, h(y) = h(x) -1
            - is relabled: if for all c(x, *) > 0, h(*) >= h(x)
    */
    namespace GoldbergTarjan{
        __global__ void push(GPUGraph G, GPUGraph Gf, int V, GPUExcessFlow e, GPUHeight height, int HEIGHT_MAX){
            // calcualte x with thread id instead of passing int
            int x = threadIdx.x;
            printf("called push: %d, e[]:%d, height[]:%d\n", x, e[x], height[x]);

            if(e[x] > 0 && height[x] < HEIGHT_MAX){
                for(int y = 0; y < V; y++){
                    if(height[y] == height[x]+1){
                        int flow = min(Gf[x][y], e[x]);
                        e[x] -= flow; e[y] += flow; // atomic ?
                        Gf[x][y] -= flow; // atomic ? 
                        Gf[y][x] += flow; // atomic ?
                    }
                }
            }
        }

        __global__ void relable(GPUGraph G, GPUGraph Gf, int V, int x, GPUExcessFlow e, GPUHeight height, int HEIGHT_MAX){
            if(e[x] > 0 && height[x] < HEIGHT_MAX){
                int my_height = HEIGHT_MAX;
                for(int y = 0; y < V; y++){
                    if(G[x][y] > 0){
                        my_height = min(my_height, height[y]+1);
                    }
                }
                height[x] = my_height;
            }
        }       

        void preflow(const Graph &G, Graph &Gf, ExcessFlow &e, Excess_total &excessTotal){
            std::cout << "called Preflow" << std::endl;
            // maybe i can parallelize this
            for(int s = 0; s < G.size(); s++){
                for(int v = 0; v < G.size(); v++){
                    if(G[s][v] > 0){
                        Gf[s][v] = 0;
                        Gf[v][s] = G[s][v];
                        e[v] = G[s][v];
                        excessTotal += G[s][v];
                    }
                }
            }
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
            preflow(G, Gf, e, excessTotal);            

            std::cout << "ExcessFlow e: ";
            for(int i = 0; i < N; i++){
                std::cout << e[i] << ", ";
            }
            std::cout << "\n";
            std::cout << "ExcessTotal: " << excessTotal << std::endl;

            // prepare GPU data
            int host_Gf[N][N], host_e[N], host_h[N];
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    host_Gf[i][j] = Gf[i][j];
                }
            }
            for(int j = 0; j < N; j++){
                host_e[j] = e[j];
                host_h[j] = height[j];
            }

            int **dev_Gf, *dev_e, *dev_h;

            // static memory allocation
            cudaMalloc((void**)&dev_Gf, N * sizeof(int*));
            for(int i=0; i<N; i++){
                cudaMalloc((void**)&(host_Gf[i]), N*sizeof(int));
            }
            cudaMemcpy(dev_Gf, host_Gf, N*sizeof(int *), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_e, N*sizeof(int));
            cudaMalloc((void**)&dev_h, N*sizeof(int));
            cudaMemcpy(dev_e, host_e, N*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_h, host_h, N*sizeof(int), cudaMemcpyHostToDevice);

            // start while
            while(e[source] + e[to] < excessTotal){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = G.size(); // = |V|
                while(cicle > 0){
                    // TODO: implement this page 5 of 2404.00270v1.pdf
		            push<<<1, N>>>(dev_Gf, dev_Gf, N, dev_e, dev_h, N);	
		            relable<<<1, N>>>(dev_Gf, dev_Gf, N, 0, dev_e, dev_h, N);	

                    //cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost); 

                    cudaDeviceSynchronize();

                    std::cout << "ExcessFlow e: ";
                    for(int i = 0; i < N; i++){
                        std::cout << e[i] << ", ";
                    }
                    std::cout << "\n";
                    std::cout << "ExcessTotal: " << excessTotal << std::endl;
                    std::cout << ">>>";std::cin.ignore();

                    cicle--;
                }
            }
        }
    };
};

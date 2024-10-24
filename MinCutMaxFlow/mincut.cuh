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
        // If the vertex is active, the thread
        // u will find its neighbor vertex v whose height is minimum among other neighbor 
        // vertices (line:10 - 13). The thread then push flow from u to v (line:15 - 19)
        // when h(u) > h(v); otherwise, the thread will relabel the active vertex u.
        __global__ void push(GPUGraph G, GPUGraph Gf, int V, GPUExcessFlow e, GPUHeight h, int HEIGHT_MAX){
            // calcualte x with thread id instead of passing int
            int x = threadIdx.x;
            int u = x;
            //printf("called push: %d, e[]:%d, height[]:%d, H_MAX:%d\n", x, e[x], height[x], HEIGHT_MAX);

            if(false && x == 0){
                for(int i = 0; i < V; i ++){
                    for(int j = 0; j < V; j++){
                        printf("%d ", Gf[i*V+j]);
                    }
                    printf("\n");
                }
            }

            if(e[u] > 0 && h[x] < HEIGHT_MAX){
                // line 10 from 2404.00270v1.pdf
                int hprime = INT_MAX;
                for(int v = 0; v < V; v++){
                    if(Gf[u*V+v] > 0){ // is (u,v) £ Ef ?
                        printf("pushing: (%d, %d) £ Ef\n", u, v);
                        // find min
                        hprime = min(hprime, h[v]);
                    }
                }
                printf("hprime: %d\n", hprime);
                // line 14 from 2404.00270v1.pdf
                if(h[u] > hprime){
                    for(int vprime = 0; vprime < V; vprime++){
                        printf("u, v': (%d, %d) \n", u, vprime);
                        if(Gf[u*V+vprime] > 0 && h[u] > h[vprime]){ 
                            int d = min(e[u], Gf[u*V+vprime]);
                            // atomic operations 
                        }
                    }
                }

            }
        }

        __global__ void relable(GPUGraph G, GPUGraph Gf, int V, int x_unused, GPUExcessFlow e, GPUHeight height, int HEIGHT_MAX){
            int x = threadIdx.x;
            //printf("called relable: %d, e[]:%d, height[]:%d, H_MAX:%d\n", x, e[x], height[x], HEIGHT_MAX);
            if(false && x == 0){
                for(int i = 0; i < V; i ++){
                    for(int j = 0; j < V; j++){
                        printf("%d ", Gf[i*V+j]);
                    }
                    printf("\n");
                }
            }
            // if(e[x] > 0 && height[x] < HEIGHT_MAX){
            //     int my_height = HEIGHT_MAX;
            //     for(int y = 0; y < V; y++){
            //         if(G[x][y] > 0){
            //             my_height = min(my_height, height[y]+1);
            //         }
            //     }
            //     height[x] = my_height;
            // }
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
            int host_Gf[N*N], host_e[N], host_h[N];
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    host_Gf[i*N+j] = Gf[i][j];
                }
            }
            
            for(int j = 0; j < N; j++){
                host_e[j] = e[j];
                host_h[j] = h[j];
            }

            int *dev_Gf, *dev_e, *dev_h;

            // static memory allocation
            cudaMalloc((void**)&dev_Gf, N * N * sizeof(int));
            cudaMemcpy(dev_Gf, host_Gf, N * N * sizeof(int), cudaMemcpyHostToDevice);

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
                    cudaDeviceSynchronize();

                    relable<<<1, N>>>(dev_Gf, dev_Gf, N, 0, dev_e, dev_h, N);	
                    cudaDeviceSynchronize();

                    cudaMemcpy(dev_Gf, host_Gf, N * N * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(dev_e, host_e, N*sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(dev_h, host_h, N*sizeof(int), cudaMemcpyDeviceToHost);


                    // print
                    std::cout << "ExcessFlow e: ";
                    for(int i = 0; i < N; i++){
                        std::cout << host_e[i] << ", ";
                    }

                    std::cout << "\n";

                    std::cout << "height h: ";
                    for(int i = 0; i < N; i++){
                        std::cout << host_h[i] << ", ";
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

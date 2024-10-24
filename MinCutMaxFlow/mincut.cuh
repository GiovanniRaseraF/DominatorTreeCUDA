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

namespace sequential{
        void pushrelable(GPUGraph G, GPUGraph Gf, GPUResidualFlow cf, int x, int V, GPUExcessFlow e, GPUHeight h, int HEIGHT_MAX){
            // calcualte x with thread id instead of passing int

            std::cout << "\n\nInside pushrelabel\ne: ";
            for(int j = 0; j < V; j++){
                std::cout << e[j] << " ";
            }
            std::cout << "\n";

            std::cout << "h: ";
            for(int j = 0; j < V; j++){
                std::cout << h[j] << " ";
            }
            std::cout << "\n";

            std::cout << "graph:\n";
            for(int i = 0; i < V; i ++){
                for(int j = 0; j < V; j++){
                    printf("%d/%d  ", Gf[i*V+j], cf[i*V+j]);
                }
                printf("\n");
            }

            std::cin.ignore();

            int u = x;

            if(e[u] > 0 && h[u] < V){
                // line 10 from 2404.00270v1.pdf
                int hprime = INT_MAX;
                int vprime = INT_MAX;
                for(int v = 0; v < V; v++){
                    if(Gf[u*V+v] > 0){ // is (u,v) £ Ef ?
                        std::cout << "v: " << v << " " << "hprime: " << hprime << std::endl;
                        if(h[v] < hprime){
                            hprime = h[v];
                            vprime = v;
                        }
                    }
                }
                std::cout << u << " " << "hprime: " << hprime << std::endl;
                if(h[u] > hprime){
                    int d = std::min(e[u], cf[u*V+vprime]);
                    cf[u*V+vprime]-=d;
                    e[u]-=d;
                    cf[vprime*V+u]+=d;
                    e[vprime]-=d;
                }else{
                    h[u] = hprime + 1;
                }

            }
        }
    
};

namespace parallel {
    namespace GoldbergTarjan{
        __global__ void pushrelable(GPUGraph G, GPUGraph Gf, int V, GPUExcessFlow e, GPUHeight h, int HEIGHT_MAX){
            // calcualte x with thread id instead of passing int
            int u = threadIdx.x;;

            if(e[u] > 0 && h[u] < HEIGHT_MAX){
                // line 10 from 2404.00270v1.pdf
                int hprime = INT_MAX;
                int vprime = INT_MAX;
                for(int v = 0; v < V; v++){
                    if(Gf[u*V+v] > 0){ // is (u,v) £ Ef ?
                        if(h[v] < hprime){
                            hprime = h[v];
                            vprime = v;
                        }
                    }
                }

                if(h[u] > hprime){
                    int d = min(e[u], Gf[u*V+vprime]);
                    Gf[u*V+vprime]-=d;
                    e[u]-=d;
                    Gf[vprime*V+u]+=d;
                    e[vprime]-=d;
                }else{
                    h[u] = hprime + 1;
                }

            }
        }

        // Initialize the flow
        void preflow(const Graph &G, Graph &Gf, ResidualFlow &cf, ExcessFlow &e, Excess_total &excessTotal){
            std::cout << "called Preflow" << std::endl;
            // maybe i can parallelize this
            for(int s = 0; s < G.size(); s++){
                for(int v = 0; v < G.size(); v++){
                    if(G[s][v] > 0){
                        cf[s][v] = 0;
                        Gf[s][v] = 0;
                        cf[v][s] = G[s][v];
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
            ResidualFlow cf(N); 
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    cf[i].push_back(0);
                }
            }
            preflow(G, Gf, cf, e, excessTotal);            

            // std::cout << "ExcessFlow e: ";
            // for(int i = 0; i < N; i++){
            //     std::cout << e[i] << ", ";
            // }
            // std::cout << "\n";
            // std::cout << "ExcessTotal: " << excessTotal << std::endl;

            // prepare GPU data
            int host_Gf[N*N], host_e[N], host_h[N], host_cf[N*N];
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    host_Gf[i*N+j] = Gf[i][j];
                    host_cf[i*N+j] = cf[i][j];
                }
            }
            
            for(int j = 0; j < N; j++){
                host_e[j] = e[j];
                host_h[j] = h[j];
            }

            std::cout << "\n\n\ne: ";
            for(int j = 0; j < N; j++){
                std::cout << host_e[j] << " ";
            }
            std::cout << "\n";

            std::cout << "h: ";
            for(int j = 0; j < N; j++){
                std::cout << host_h[j] << " ";
            }
            std::cout << "\n";

            std::cout << "graph:\n";
            for(int i = 0; i < N; i ++){
                for(int j = 0; j < N; j++){
                    printf("%d/%d  ", host_Gf[i*N+j], host_cf[i*N+j]);
                }
                printf("\n");
            }

            std::cin.ignore();


            // int *dev_Gf, *dev_e, *dev_h;

            // // static memory allocation
            // cudaMalloc((void**)&dev_Gf, N * N * sizeof(int));
            // cudaMemcpy(dev_Gf, host_Gf, N * N * sizeof(int), cudaMemcpyHostToDevice);

            // cudaMalloc((void**)&dev_e, N*sizeof(int));
            // cudaMalloc((void**)&dev_h, N*sizeof(int));
            // cudaMemcpy(dev_e, host_e, N*sizeof(int), cudaMemcpyHostToDevice);
            // cudaMemcpy(dev_h, host_h, N*sizeof(int), cudaMemcpyHostToDevice);

            // start while
            //while((host_e[source] + host_e[to]) < excessTotal){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = G.size(); // = |V|
                while(cicle > 0){
                    for(int u = 0; u < N; u++){
		                sequential::pushrelable(host_Gf, host_Gf, host_cf, u, N, host_e, host_h, N);	
                    }

		            //pushrelable<<<1, N>>>(dev_Gf, dev_Gf, N, dev_e, dev_h, N);	
                    //cudaDeviceSynchronize();
                    
                    // cudaMemcpy(host_Gf, dev_Gf, N * N * sizeof(int), cudaMemcpyDeviceToHost);
                    // cudaMemcpy(host_e, dev_e, N*sizeof(int), cudaMemcpyDeviceToHost);
                    // cudaMemcpy(host_h, dev_h, N*sizeof(int), cudaMemcpyDeviceToHost);

                    std::cout << "\n\n\ne: ";
                    for(int j = 0; j < N; j++){
                        std::cout << host_e[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "h: ";
                    for(int j = 0; j < N; j++){
                        std::cout << host_h[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "graph:\n";
                    for(int i = 0; i < N; i ++){
                        for(int j = 0; j < N; j++){
                            printf("%d/%d  ", host_Gf[i*N+j], host_cf[i*N+j]);
                        }
                        printf("\n");
                    }

                    // std::cout << "\n";
                    // std::cout << "ExcessTotal: " << excessTotal << std::endl;
                    std::cout << ">>>" << "\ncicle: " << cicle << "\ne(0): " << host_e[source] << "\ne[to]: " << host_e[to] << "\nexcessTotal: " << excessTotal << "\n"; 
                    std::cin.ignore();

                    cicle--;
                }
            //}
        }
    };
};

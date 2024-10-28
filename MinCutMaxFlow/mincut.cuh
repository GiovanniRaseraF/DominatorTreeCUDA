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

constexpr int num_nodes = 7;
std::vector<int> heights{num_nodes};
int ExcessTotal = 0;
std::vector<int> offsets{num_nodes * num_nodes};
std::vector<int> destinations{num_nodes * num_nodes};
std::vector<int> capacities{num_nodes * num_nodes};
std::vector<int> excesses{num_nodes};
std::vector<int> forward_flows{num_nodes*num_nodes};
std::vector<int> backward_flows{num_nodes*num_nodes};
int source = 0;
constexpr int sink = num_nodes-1;

namespace sequential {
    void preflow(int source){
        heights[source] = num_nodes; 
        ExcessTotal = 0;

        // Initialize preflow
        for (int i = offsets[source]; i < offsets[source + 1]; ++i) {
            int dest = destinations[i];
            int cap = capacities[i];

            excesses[dest] = cap;
            forward_flows[i] = 0; // residualFlow[(source, dest)] = 0
            backward_flows[i] = cap; // residualFlow[(dest, source)] = cap
            ExcessTotal += cap;
            printf("Source: %d's neighbor: %d\n", source, dest);
        }
    }

    bool push(int v){
    // Find the outgoing edge (v, w) in foward edge with h(v) = h(w) + 1
        for (int i = offsets[v]; i < offsets[v + 1]; ++i) {
            int w = destinations[i];
            if (heights[v] == heights[w] + 1) {
            // Push flow
                int flow = std::min(excesses[v], forward_flows[i]);
                if (flow == 0) continue;
                forward_flows[i] -= flow;
                backward_flows[i] += flow;
                excesses[v] -= flow;
                excesses[w] += flow;
                printf("Pushing flow %d from %d(%d) to %d(%d)\n", flow, v, excesses[v], w, excesses[w]);
                return true;
            }
        }
    }


    void relabel(int u){
        heights[u]+=1;
    }

    int findActiveNode(void){
        int max_height = num_nodes;
        int return_node = -1;
        for (int i = 0; i < num_nodes; ++i) {
            if (excesses[i] > 0 && i != source && i != sink) {
                if (heights[i] < max_height) {
                    max_height = heights[i];
                    return_node = i;
                }
            }
        }
        return return_node;
    }

    int countActiveNodes(void){
        int count = 0;
        for (int i = 0; i < num_nodes; ++i) {
            if (excesses[i] > 0 && i != source && i != sink) {
            count++;
            }
        }
        return count;
    }

    void maxflow() {
        preflow(source);

        printf("Preflow done\n");
        printf("Excess total: %d\n", ExcessTotal);

        int active_node = findActiveNode();

        while(active_node != -1) {
            /* If there is an outgoing edge (v, w) of v in Gf with h(v) = h(w) + 1 */
            //printf("#active nodes: %d\n", countActiveNodes());
            if (!push(active_node)) {
            printf("Relabeling %d\n", active_node);
            relabel(active_node);
            }
            active_node = findActiveNode();
        }


        /* Calculate Max flow */
        /* Sum all all rflow(u, sink)*/
        printf("Max flow: %d\n", excesses[sink]);

    }
};


namespace parallel {


    namespace GoldbergTarjan{
        __global__ void pushrelable(GPUGraph G, GPUGraph Gf, int V, GPUExcessFlow e, GPUHeight h, int HEIGHT_MAX){
            // calcualte x with thread id instead of passing int
            int u = threadIdx.x;
            printf("threadnidx: %d\n", u);
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
                    atomicSub(&Gf[u*V+vprime], d);
                    atomicSub(&e[u], d);
                    atomicAdd(&Gf[vprime*V+u], d);
                    atomicAdd(&e[vprime], d);
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
            ResidualFlow cf; cf.assign(N, std::vector<int>(N, 0)); 
            preflow(G, Gf, cf, e, excessTotal);            

            h[source] = N;
            e[source] = INT_MAX;

            // prepare GPU data
            int host_G[N*N], host_Gf[N*N], host_e[N], host_h[N], host_cf[N*N];
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    host_G[i*N+j] = G[i][j];
                    host_Gf[i*N+j] = Gf[i][j];
                }
            }

            for(int j = 0; j < N; j++){
                host_e[j] = e[j];
                host_h[j] = h[j];
            }


            int *dev_Gf, *dev_e, *dev_h, *dev_G;

            // static memory allocation
            cudaMalloc((void**)&dev_G, N * N * sizeof(int));
            cudaMemcpy(dev_G, host_G, N * N * sizeof(int), cudaMemcpyHostToDevice);
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
		            pushrelable<<<1, N>>>(dev_G, dev_Gf, N, dev_e, dev_h, N);	
                    cudaDeviceSynchronize();
                    
                    cudaMemcpy(host_Gf, dev_Gf, N * N * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(host_e, dev_e, N*sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(host_h, dev_h, N*sizeof(int), cudaMemcpyDeviceToHost);

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
                            printf("%d/%d  ", G[i][j], host_Gf[i*N+j]);
                        }
                        printf("\n");
                    }

                    // std::cout << "\n";
                    // std::cout << "ExcessTotal: " << excessTotal << std::endl;
                    //std::cout << ">>>" << "\ncicle: " << cicle << "\ne(0): " << host_e[source] << "\ne[to]: " << host_e[to] << "\nexcessTotal: " << excessTotal << "\n"; 
                    std::cin.ignore();

                    cicle--;
                }
            }
        }
    };
};

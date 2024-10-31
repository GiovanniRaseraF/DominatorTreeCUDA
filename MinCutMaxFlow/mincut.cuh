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
std::vector<int> heights(num_nodes, 0);
int ExcessTotal = 0;
std::vector<int> G(num_nodes * num_nodes, 0);
std::vector<int> Gr(num_nodes * num_nodes, 0);
std::vector<int> flow_index(num_nodes * num_nodes, 0);
std::vector<int> destinations(num_nodes * num_nodes, 0);
std::vector<int> capacities(num_nodes * num_nodes, 0);
std::vector<int> excesses(num_nodes, 0);
std::vector<int> forward_flows(num_nodes*num_nodes, 0);
std::vector<int> backward_flows(num_nodes*num_nodes, 0);
int source = 0;
constexpr int sink = 6;
constexpr int to = num_nodes-1;

namespace sequential {
    void preflow(int source){
        heights[source] = num_nodes; 
        ExcessTotal = 0;

        // Initialize preflow
        for (int i = (source*num_nodes); i < (source*num_nodes)+num_nodes; ++i) {
            if(G[i] > 0){
                int dest = i - (source*num_nodes);
                int cap = G[i];

                excesses[dest] = cap;
                forward_flows[i] = 0; // residualFlow[(source, dest)] = 0
                backward_flows[i] = cap; // residualFlow[(dest, source)] = cap
                ExcessTotal += cap;
                printf("Source: %d's neighbor: %d\n", source, dest);
            }
        }

        // Initialize flow index
        for (int u = 0; u < num_nodes; u++) {
            for (int i = (u*num_nodes); i < (u + 1)*num_nodes; i++) {
                if(Gr[i] > 0){
                    int v = i - (u*num_nodes); 
                    // Find the forward edge index
                    for (int j = (v*num_nodes); j < (v + 1)*num_nodes; j++) {
                        if(G[j] > 0){
                            if (G[j] == u) {
                                flow_index[i] = j;
                                break;
                            }
                        }
                        
                    }
                }
            }
        }
    }

    bool push(int v){
        // Find the outgoing edge (v, w) in foward edge with h(v) = h(w) + 1
        for (int i = (v*num_nodes); i < (v*num_nodes)+num_nodes; ++i) {
            if(G[i] > 0){
                //std::cout << "node before push" << std::endl;
                int w = i - (v*num_nodes);
                if (heights[v] == heights[w] + 1) {
                    //std::cout << "node after push" << std::endl;
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

        // Find the outgoing edge (v, w) in backward edge with h(v) = h(w) + 1
        for (int i = (v*num_nodes); i < (v*num_nodes)+num_nodes; ++i) {
            if(Gr[i] > 0){
                int w = i - (v*num_nodes);
                if (heights[v] == heights[w] + 1) {
                    // Push flow
                    int push_index = flow_index[i];
                    int flow = std::min(excesses[v], backward_flows[push_index]);
                    if (flow == 0) continue;
                    backward_flows[push_index] -= flow;
                    forward_flows[push_index] += flow;
                    excesses[v] -= flow;
                    excesses[w] += flow;
                    printf("Pushing flow %d from %d(%d) to %d(%d)\n", flow, v, excesses[v], w, excesses[w]);
                    return true;
                }
            }
        }

        return false;
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

    void buildGr(){
        for (int u = 0; u < num_nodes; u++){
            for (int v = 0; v < num_nodes; v++){
                Gr[u*num_nodes+v] = G[v*num_nodes+u];
            }
        }
    }

    void maxflow() {
        G[source*num_nodes + 1] = 3;
        G[source*num_nodes + 2] = 9;
        G[source*num_nodes + 3] = 5;
        G[source*num_nodes + 4] = 6;
        G[source*num_nodes + 5] = 2;

        G[1*num_nodes+2] = 3;
        G[2*num_nodes+3] = 3;
        G[2*num_nodes+1] = 3;
        G[3*num_nodes+2] = 3;
        G[4*num_nodes+3] = 4;
        G[5*num_nodes+4] = 1;


        G[1*num_nodes+to] = 10;
        G[2*num_nodes+to] = 2;
        G[3*num_nodes+to] = 1;
        G[4*num_nodes+to] = 8;
        G[5*num_nodes+to] = 9;

        buildGr();

        std::cout << "graph:\n";
        for(int i = 0; i < num_nodes; i ++){
            for(int j = 0; j < num_nodes; j++){
                printf("%d ", G[i*num_nodes+j]);
            }
            printf("\n");
        }
        std::cout << "before preflow heights: ";
        for(int i = 0; i < num_nodes; i ++){
            printf("%d ", heights[i]);
        }
        std::cout << "\n";
        std::cout << "before preflow excesses: ";
        for(int i = 0; i < num_nodes; i ++){
            printf("%d ", excesses[i]);
        }
        std::cout << "\n";

        preflow(source);

        std::cout << "after preflow heights: ";
        for(int i = 0; i < num_nodes; i ++){
            printf("%d ", heights[i]);
        }
        std::cout << "\n";
        std::cout << "after preflow excesses: ";
        for(int i = 0; i < num_nodes; i ++){
            printf("%d ", excesses[i]);
        }
        std::cout << "\n";

        printf("Preflow done\n");
        printf("Excess total: %d\n", ExcessTotal);

        int active_node = findActiveNode();
        std::cin.ignore();

        while(active_node != -1) {
            /* If there is an outgoing edge (v, w) of v in Gf with h(v) = h(w) + 1 */
            //printf("#active nodes: %d\n", countActiveNodes());
            if (!push(active_node)) {
                printf("Relabeling %d\n", active_node);
                relabel(active_node);
            }
            std::cout << "g/f/b:\n";
            for(int i = 0; i < num_nodes; i ++){
                for(int j = 0; j < num_nodes; j++){
                    printf("%d/%d/%d  ", G[i*num_nodes+j], 
                    forward_flows[i*num_nodes+j], 
                    backward_flows[i*num_nodes+j]);
                }
                printf("\n");
            }

            std::cout << "heights: ";
            for(int i = 0; i < num_nodes; i ++){
                printf("%d ", heights[i]);
            }
            std::cout << "\n";

            std::cout << "excesses: ";
            for(int i = 0; i < num_nodes; i ++){
                printf("%d ", excesses[i]);
            }
            std::cout << "\n";

            active_node = findActiveNode();
            std::cout << "active_node" << active_node << std::endl;
        }


        /* Calculate Max flow */
        /* Sum all all rflow(u, sink)*/
        printf("Max flow: %d\n", excesses[sink]);
        std::cout << "graph:\n";
        for(int i = 0; i < num_nodes; i ++){
            for(int j = 0; j < num_nodes; j++){
                printf("%d/%d  ", G[i*num_nodes+j], 
                //forward_flows[i*num_nodes+j], 
                backward_flows[i*num_nodes+j]);
            }
            printf("\n");
        }

        std::cout << "heights: ";
        for(int i = 0; i < num_nodes; i ++){
            printf("%d ", heights[i]);
        }
        std::cout << "\n";

        for(int i = 0; i < num_nodes; i ++){
            for(int j = 0; j < num_nodes; j++){
                //printf("%d-%d/%d  ", G[i*num_nodes+j], forward_flows[i*num_nodes+j], backward_flows[i*num_nodes+j]);
                if(G[i*num_nodes+j] > 0 && (G[i*num_nodes+j] == backward_flows[i*num_nodes+j])){
                    printf("Delete edge: %d -> %d\n", i, j);
                }
            }
            //printf("\n");
        }
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

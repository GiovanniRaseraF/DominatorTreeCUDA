// Author: Giovanni Rasera
// Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
// Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
// Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
// Help From: https://arxiv.org/pdf/2404.00270
// Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <iomanip>

typedef std::vector<std::vector<int>> Graph;
typedef std::vector<std::vector<int>> ResidualFlow;
typedef std::vector<int> ExcessFlow;
typedef std::vector<int> Height;
typedef int Excess_total;

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
        __global__ void push(){
            printf("TODO: GPU push");
        }

        __global__ void relable(){
            printf("TODO: GPU relable");
        }       

        void preflow(const Graph &G, Graph &Gf, ExcessFlow &e, Excess_total &excessTotal){
            // maybe i can parallelize this
            for(int s = 0; s < G.size(); s++){
                for(int v = 0; v < G.size(); v++){
                    Gf[s][v] = 0;
                    Gf[v][s] = G[s][v];
                    e[v] = G[s][v];
                    excessTotal += G[s][v];
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
        void MinCutMaxFlow(Graph &G, Graph &Gf, ExcessFlow &e, Height &h, int source, int to){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
            
            // Initialize
            Excess_total excessTotal = 0;

            // Step 0: Preflow
            preflow(G, Gf, e, excessTotal);            

            while(e(source) + e(to) < excessTotal){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = graph.size(); // = |V|
                while(cicle > 0){
                    // TODO: implement this page 5 of 2404.00270v1.pdf
                    // push<<<1, 1>>>();
                    // relable<<<1, 1>>>();
                    // cudaDeviceSynchronize();

                    cicle--;
                }
            }
        }
    };
};
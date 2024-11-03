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

typedef std::vector<std::vector<int>> Graph;

// GPU data
typedef int* GPUOffsets;
typedef int* GPUrOffsets;
typedef int* GPUDestinations;
typedef int* GPUrDestinations;
typedef int* GPUCapacities;
typedef int* GPUrCapacities;

typedef int* GPUFlowIndex;

typedef int* GPUHeights;

typedef int* GPUForwardFlow;
typedef int* GPUBackwardFlow;

typedef int* GPUExcesses;

typedef int* GPUExcessTotal;

// implementation
namespace parallel {
    namespace GoldbergTarjan{
        __global__ void push(
            GPUOffsets offsets,
            GPUrOffsets Roffsets,
            GPUDestinations destinations,
            GPUrDestinations Rdestinations,
            GPUCapacities capacities,
            GPUrCapacities Rcapacities,
            GPUFlowIndex flowIndex,
            GPUHeights heights,
            GPUForwardFlow forwardFlows,
            GPUBackwardFlow backwardFlows,
            GPUExcesses excesses,
            GPUExcessTotal excessTotal,
            int numNodes,
            int numEdges,
            int source,
            int to
        ){

        }

        __global__ void relable(
            GPUOffsets offsets,
            GPUrOffsets Roffsets,
            GPUDestinations destinations,
            GPUrDestinations Rdestinations,
            GPUCapacities capacities,
            GPUrCapacities Rcapacities,
            GPUFlowIndex flowIndex,
            GPUHeights heights,
            GPUForwardFlow forwardFlows,
            GPUBackwardFlow backwardFlows,
            GPUExcesses excesses,
            GPUExcessTotal excessTotal,
            int numNodes,
            int numEdges,
            int source,
            int to
        ){

        }

        // Initialize the flow
        void preflow(
            GPUOffsets offsets,
            GPUrOffsets Roffsets,
            GPUDestinations destinations,
            GPUrDestinations Rdestinations,
            GPUCapacities capacities,
            GPUrCapacities Rcapacities,
            GPUFlowIndex flowIndex,
            GPUHeights heights,
            GPUForwardFlow forwardFlows,
            GPUBackwardFlow backwardFlows,
            GPUExcesses excesses,
            GPUExcessTotal excessTotal,
            int numNodes,
            int numEdges,
            int source,
            int to
        ){
            heights[source] = numNodes; 
            *excessTotal = 0;

            // Initialize preflow
            for (int i = offsets[source]; i < offsets[source + 1]; ++i) {
                int dest = destinations[i];
                int cap = capacities[i];

                excesses[dest] = cap;
                forwardFlows[i] = 0; 
                backwardFlows[i] = cap;
                *excessTotal = *excessTotal + cap;
            } 
        }

        void test(int *o, int l){
            for(int i = 0; i < l; i++){
                std::cout << o[i] << std::endl;
            }
        }

        void minCutMaxFlow(Graph &G, int source, int to){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
            constexpr int numNodes = 7;
            constexpr int numEdges = 15;
            int offsets[numNodes+1]{0, 6, 8, 11, 13, 15, 18, 18, };
            int destinations[numEdges]{1, 2, 3, 5, 5, 5, 2, 6, 1, 3, 6, 2, 6, 3, 6, };
            int capacities[numEdges]{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, };
            int forwardFlow[numEdges]{1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, };
            int backwardFlows[numEdges]{0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, };
            int rOffsets[numNodes+1]{0, 0, 2, 5, 8, 10, 13, 18, };
            int rDestinations[numEdges]{0, 2, 0, 1, 3, 0, 2, 4, 5, 5, 0, 0, 0, 1, 2, };
            int flowIndex[numEdges]{0, 8, 1, 6, 11, 2, 9, 13, 0, 0, 3, 3, 3, 7, 10, };
            int heights[numNodes]{7, 6, 6, 0, 0, 2, 7, };
            int excesses[numNodes]{0, 0, 0, 2, 0, 0, 1, };
        }
    };
};

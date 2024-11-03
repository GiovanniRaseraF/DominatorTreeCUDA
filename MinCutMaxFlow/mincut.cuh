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
            std::cout << "TODO: push" << std::endl;

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
            std::cout << "TODO: relable" << std::endl;

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

        void minCutMaxFlow(Graph &G, int source, int to){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
        }
    };
};

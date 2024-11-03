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

        void print(
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
            printf("int offsets[numNodes+1]{");
            for (int i=0; i < numNodes + 1; i++) {
                printf("%d, ", offsets[i]);
            }
            printf("};\n");

            printf("int rOffsets[numNodes+1]{");
            for (int i=0; i < numNodes + 1; i++) {
                printf("%d, ", Roffsets[i]);
            }
            printf("};\n");

            printf("int destinations[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", destinations[i]);
            }
            printf("};\n");

            printf("int rDestinations[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", Rdestinations[i]);
            }
            printf("};\n");

            printf("int capacities[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", capacities[i]);
            }
            printf("};\n");

            printf("int rCapacities[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", capacities[i]);
            }
            printf("};\n");

            printf("int flowIndex[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", flowIndex[i]);
            }
            printf("};\n");

            printf("int heights[numNodes]{");
            for (int i=0; i < numNodes; i++) {
                printf("%d, ", heights[i]);
            }
            printf("};\n");

            printf("int forwardFlow[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", forwardFlows[i]);
            }
            printf("};\n");

            printf("int backwardFlows[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", backwardFlows[i]);
            }
            printf("};\n");

            printf("int excesses[numNodes]{");
            for (int i=0; i < numNodes; i++) {
                printf("%d, ", excesses[i]);
            }
            printf("};\n");

            printf("int excessTotal[1]{%d};\n", excessTotal);
        }

        void minCutMaxFlow(Graph &G, int source, int to){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
            constexpr int numNodes = 7;
            constexpr int numEdges = 15;
            int offsets[numNodes+1]{0, 6, 8, 11, 13, 15, 18, 18, };
            int rOffsets[numNodes+1]{0, 0, 2, 5, 8, 10, 13, 18, };

            int destinations[numEdges]{1, 2, 3, 5, 5, 5, 2, 6, 1, 3, 6, 2, 6, 3, 6, };
            int rDestinations[numEdges]{0, 2, 0, 1, 3, 0, 2, 4, 5, 5, 0, 0, 0, 1, 2, };

            int capacities[numEdges]{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, };
            int rCapacities[numEdges]{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, };

            int flowIndex[numEdges]{0, 8, 1, 6, 11, 2, 9, 13, 0, 0, 3, 3, 3, 7, 10, };
            int heights[numNodes]{0, 0, 0, 0, 0, 0, 0, };

            int forwardFlow[numEdges]{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, };
            int backwardFlows[numEdges]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
            int excesses[numNodes]{0, 0, 0, 0, 0, 0, 0, };

            int excessTotal[1]{352493152};

            print(
                offsets,
                rOffsets,

                destinations,
                rDestinations,

                capacities,
                rCapacities,

                flowIndex,
                heights,

                forwardFlow,
                backwardFlows,
                excesses,

                excessTotal,
                numNodes,
                numEdges,
                source,
                to
            );
        }
    };
};

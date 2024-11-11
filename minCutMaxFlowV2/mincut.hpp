// Author: Giovanni Rasera
// Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
// Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
// Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
// Help From: https://arxiv.org/pdf/2404.00270
// Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
// Help From: https://www.adrian-haarbach.de/idp-graph-algorithms/implementation/maxflow-push-relabel/index_en.html
// Help From: https://www.geeksforgeeks.org/push-relabel-algorithm-set-2-implementation/
#pragma once

#include "commons.hpp"

// implementation
namespace parallel {
    namespace GoldbergTarjan{
        int findActiveNodeGPU(
           PARAMPASS 
        ){
            int max_height = numNodes;
            int return_node = -1;

            for (int i = 0; i < numNodes; ++i) {
                if (excesses[i] > 0 && i != source && i != to) {
                    if (heights[i] < max_height) {
                        max_height = heights[i];
                        return_node = i;
                    }
                }
            }

            return return_node;
        }

        int findActiveNode(
           PARAMPASS 
        ){
            int max_height = numNodes;
            int return_node = -1;

            for (int i = 0; i < numNodes; ++i) {
                if (excesses[i] > 0 && i != source && i != to) {
                    if (heights[i] < max_height) {
                        max_height = heights[i];
                        return_node = i;
                    }
                }
            }

            return return_node;
        }

        bool push(
            PARAMPASS,
            int v,
            bool *ret
        ){
            // Find the outgoing edge (v, w) in foward edge with h(v) = h(w) + 1
            for (int i = offsets[v]; i < offsets[v + 1]; ++i) {
                int w = destinations[i];
                if (heights[v] == heights[w] + 1) {
                    // Push flow
                    int flow = std::min(excesses[v], forwardFlows[i]);
                    if (flow == 0) continue;

                    forwardFlows[i] -= flow;
                    backwardFlows[i] += flow;
                    excesses[v] -= flow;
                    excesses[w] += flow;

                    printf("->Pushing flow %d from %d(%d) to %d(%d)\n", flow, v, excesses[v], w, excesses[w]);
                    //*ret = true;
                    return true;
                }
            }

            // Find the outgoing edge (v, w) in backward edge with h(v) = h(w) + 1
            for (int i = Roffsets[v]; i < Roffsets[v+1]; ++i) {
                int w = Rdestinations[i];
                if (heights[v] == heights[w] + 1) {
                    // Push flow
                    int push_index = flowIndex[i];
                    int flow = std::min(excesses[v], backwardFlows[push_index]);
                    if (flow == 0) continue;

                    backwardFlows[push_index] -= flow;
                    forwardFlows[push_index] += flow;
                    excesses[v] -= flow;
                    excesses[w] += flow;

                    printf("<-Pushing flow %d from %d(%d) to %d(%d)\n", flow, v, excesses[v], w, excesses[w]);
                    //*ret = true;
                    return true;
                }
            }

            //*ret = false;
            return false;
        }

        // Initialize the flow
        void preflow(
           PARAMPASS 
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

        void relabel(
            GPUHeights heights,
            int u
        ){
            heights[u]+=1;
        }

       

        void minCutMaxFlow(Graph &G, int source, int to,
            int *offsets,int *rOffsets,
            int *destinations,int *rDestinations,
            int *capacities,int *rCapacities,
            int *flowIndex,int *heights,
            int *forwardFlow,int *backwardFlows,int *excesses,int numNodes,int numEdges
        ){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;

            int excessTotal[1]{0};
            bool ret[1]{false};

            // print(
            //     offsets,rOffsets,
            //     destinations,rDestinations,
            //     capacities,rCapacities,
            //     flowIndex,heights,
            //     forwardFlow,backwardFlows,excesses,
            //     excessTotal,numNodes,numEdges,source,to
            // );

            //printf("Preflow: \n");
            preflow(
                offsets,rOffsets,
                destinations,rDestinations,
                capacities,rCapacities,
                flowIndex,heights,
                forwardFlow,backwardFlows,excesses,
                excessTotal,numNodes,numEdges,source,to
            );

            // gpu structure
            int * gpu_offsets;
            int * gpu_rOffsets;

            int * gpu_destinations;
            int * gpu_rDestinations;

            int * gpu_capacities;
            int * gpu_rCapacities;

            int * gpu_flowIndex;
            int * gpu_heights;

            int * gpu_forwardFlow;
            int * gpu_backwardFlows;
            int * gpu_excesses;

            int * gpu_excessTotal;
            int * gpu_numNodes;
            int * gpu_numEdges;
            int * gpu_source;
            int * gpu_to;
            int * gpu_active;
            int * gpu_re;

            // gpu malloc
            

            int active = findActiveNode(
                offsets,rOffsets,
                destinations,rDestinations,
                capacities,rCapacities,
                flowIndex,heights,
                forwardFlow,backwardFlows,excesses,
                excessTotal,numNodes,numEdges,source,to
            );

            while(active != -1){
                // for each node
                bool p = push(
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
                    to,
                    active,
                    ret
                );

                if(!p){
                    relabel(heights, active);
                }

                active = findActiveNode(
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
        printf("\n\n");
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

        std::cout << "\n\nMaxFlow: " << excesses[to] << std::endl;
        }
    };
};

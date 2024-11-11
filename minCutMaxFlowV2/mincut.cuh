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
        __global__ void pushGPU(
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
                    *ret = true;
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
                    *ret = true;
                }
            }

            *ret = false;
        }
    }
}
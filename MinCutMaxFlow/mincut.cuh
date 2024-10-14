// Author: Giovanni Rasera
// Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
// Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
// Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
// Help From: https://arxiv.org/pdf/2404.00270
// Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
#pragma once

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
        __global__ void push(){}
        __global__ void relable(){}       

        void MinCutFaxFlow(/* ?? */){

        }
    };
};
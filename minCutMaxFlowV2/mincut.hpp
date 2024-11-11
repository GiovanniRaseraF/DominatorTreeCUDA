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
#include "mincut.cuh"

// implementation
namespace parallel {
    namespace GoldbergTarjan{
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
            int *offsets,int *roffsets,
            int *destinations,int *rdestinations,
            int *capacities,int *rcapacities,
            int *flow_index,int *heights,
            int *fflow,int *bflow,int *excess_flow,int numNodes,int numEdges
        ){
            std::cout << "TODO: MinCutFaxFlow" << std::endl;
            int V = numNodes;
            int E = numEdges;
            int sink = to;
            int excessTotal[1]{0};
            bool ret[1]{false};

            // prefase
            // Configure the GPU
            int device = -1;
            cudaGetDevice(&device);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM);
            dim3 block_size(numThreadsPerBlock);
            size_t sharedMemSize = 3 * block_size.x * sizeof(int);

            

            preflow(
                offsets,        roffsets,
                destinations,   rdestinations,
                capacities,     rcapacities,
                flow_index,     heights,
                fflow,bflow,    excess_flow,
                excessTotal,
                numNodes,       numEdges,
                source,         to
            );

            // gpu structure
            int * gpu_offsets;
            int * gpu_roffsets;
            int * gpu_destinations;
            int * gpu_rdestinations;
            int * gpu_capacities;
            int * gpu_flow_index;
            int * gpu_height;
            int * gpu_fflows;
            int * gpu_bflows;
            int * gpu_excess_flow;

            void* original_kernel_args[] = {
                &V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, &gpu_bflows, 
                &gpu_roffsets, &gpu_rdestinations, &gpu_flow_index};

            // gpu malloc
            (cudaMalloc((void**)&gpu_offsets,       (V+1)*sizeof(int)));
            (cudaMalloc((void**)&gpu_roffsets,      (V+1)*sizeof(int)));
            (cudaMalloc((void**)&gpu_destinations,  E*sizeof(int)));
            (cudaMalloc((void**)&gpu_rdestinations, E*sizeof(int)));
            (cudaMalloc((void**)&gpu_capacities,    E*sizeof(int)));
            (cudaMalloc((void**)&gpu_flow_index,    E*sizeof(int)));
            (cudaMalloc((void**)&gpu_height,        V*sizeof(int)));
            (cudaMalloc((void**)&gpu_fflows,        E*sizeof(int)));
            (cudaMalloc((void**)&gpu_bflows,        E*sizeof(int)));
            (cudaMalloc((void**)&gpu_excess_flow,   V*sizeof(int)));

            // mem copy
            (cudaMemcpy(gpu_height,         heights,        V*sizeof(int),        cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_excess_flow,    excess_flow,    V*sizeof(int),        cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_offsets,        offsets,        (numNodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_destinations,   destinations,   numEdges*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_capacities,     capacities,     numEdges*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_fflows,         fflow,          numEdges*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_roffsets,       roffsets,       (numNodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_rdestinations,  rdestinations,  numEdges*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_bflows,         bflow,          numEdges*sizeof(int), cudaMemcpyHostToDevice));
            (cudaMemcpy(gpu_flow_index,     flow_index,     numEdges*sizeof(int), cudaMemcpyHostToDevice));

            //while((excess_flow[source] + excess_flow[sink]) < *excessTotal){
                (cudaMemcpy(gpu_height,        heights,         V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_excess_flow,   excess_flow,     V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_fflows,        fflow,          E*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_bflows,        bflow,          E*sizeof(int), cudaMemcpyHostToDevice));
                // (cudaMemset(gpu_cycle,         V,               sizeof(int))); // Reset the gpu_cycle to V

                // gpu call
                cudaLaunchCooperativeKernel((void*)push_relabel_kernel, num_blocks, block_size, original_kernel_args, sharedMemSize, 0);
                cudaDeviceSynchronize();

                (cudaMemcpy(heights,       gpu_height,         V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(excess_flow,   gpu_excess_flow,    V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(fflow,        gpu_fflows,         E*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(bflow,        gpu_bflows,         E*sizeof(int), cudaMemcpyDeviceToHost));
            //}
        }
    };
};


// while(active != -1){
//                 // for each node
//                 bool p = push(
//                     offsets,roffsets,
//                     destinations,rdestinations,
//                     capacities,rcapacities,
//                     flow_index,heights,
//                     fflow,bflow,excess_flow,

//                     excessTotal,
//                     numNodes,
//                     numEdges,
//                     source,
//                     to,
//                     active,
//                     ret
//                 );

//                 if(!p){
//                     relabel(heights, active);
//                 }

//                 active = findActiveNode(
//                     offsets,roffsets,
//                     destinations,rdestinations,
//                     capacities,rcapacities,
//                     flow_index,heights,
//                     fflow,bflow,excess_flow,

//                     excessTotal,
//                     numNodes,
//                     numEdges,
//                     source,
//                     to
//                 );

//             }
//         printf("\n\n");
//         print(
//             offsets,roffsets,
//             destinations,rdestinations,
//             capacities,rcapacities,
//             flow_index,heights,
//             fflow,bflow,excess_flow,

//             excessTotal,
//             numNodes,
//             numEdges,
//             source,
//             to
//         );
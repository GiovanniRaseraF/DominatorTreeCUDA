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
        // Initialize the flow
        void preflow(
            int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
            int *offsets, int *destinations, int* capacities, int* forward_flows, int* backward_flows,
            int *roffsets, int* rdestinations, int* flow_idx, int *Excess_total)
        {
            // initialising height values and excess flow, Excess_total values
            for(int i = 0; i < V; i++)
            {
                cpu_height[i] = 0; 
                cpu_excess_flow[i] = 0;
            }
    
            cpu_height[source] = V;
            *Excess_total = 0;

            // pushing flow in all edges going out from the source node
            for(int i = offsets[source];  i < offsets[source + 1]; i++) {
                int neighborID = destinations[i];
        
                if (capacities[i] > 0)  {
                    forward_flows[i] = 0;
                    backward_flows[i] = capacities[i];
                    cpu_excess_flow[neighborID] = capacities[i];
                    *Excess_total += cpu_excess_flow[neighborID];
                } else {
                    continue;
                }
            }
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
                V, source, sink, heights, excess_flow, 
                (offsets), (destinations), (capacities), (fflow), (bflow),
                (roffsets), (rdestinations), (flow_index), excessTotal);
    
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

            bool *mark,*scanned;
            mark =      (bool*)malloc(V*sizeof(bool));
            scanned =   (bool*)malloc(V*sizeof(bool));
            for(int i = 0; i < V; i++) mark[i] = false;

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

            // algo start
            while((excess_flow[source] + excess_flow[sink]) < *excessTotal){
                (cudaMemcpy(gpu_height,        heights,         V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_excess_flow,   excess_flow,     V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_fflows,        fflow,           E*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_bflows,        bflow,           E*sizeof(int), cudaMemcpyHostToDevice));
                // (cudaMemset(gpu_cycle,         V,               sizeof(int))); // Reset the gpu_cycle to V

                // gpu call
                cudaLaunchCooperativeKernel((void*)push_relabel_kernel, num_blocks, block_size, original_kernel_args, sharedMemSize, 0);
                cudaDeviceSynchronize();

                (cudaMemcpy(heights,        gpu_height,         V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(excess_flow,    gpu_excess_flow,    V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(fflow,          gpu_fflows,         E*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(bflow,          gpu_bflows,         E*sizeof(int), cudaMemcpyDeviceToHost));

                print(
                    V, E, source, sink, heights, excess_flow,
                    offsets, destinations, capacities, fflow, bflow,
                    roffsets, rdestinations, flow_index,
                    );

                global_relabel(
                    V, E, source, sink, heights, excess_flow,
                    offsets, destinations, capacities, fflow, bflow,
                    roffsets, rdestinations, flow_index,
                    excessTotal, 
                    mark, scanned);

                print(
                    V, E, source, sink, heights, excess_flow,
                    offsets, destinations, capacities, fflow, bflow,
                    roffsets, rdestinations, flow_index,
                    excessTotal);
            }
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
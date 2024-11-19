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
#include <thread>
#include <chrono>
#include <vector>
#include <tuple>

// implementation
namespace parallel {
    namespace GoldbergTarjan{
        // Used to find the minCut nodes to cut
        void dfs(
            bool *visited, 
            int V,
            int source,
            int *roffsets, int* rdestinations, int *bflow
        ){
            visited[source] = true;
            for(int i = roffsets[source]; i < roffsets[source+1]; ++i){
                int y = rdestinations[i];
                if(!visited[y] && bflow[y] > 0){
                    dfs(visited, V, y, roffsets, rdestinations, bflow);
                }
            }
        }

        // Initialize the flow
        void preflow(
            int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
            int *offsets, int *destinations, int* capacities, int* forward_flows, int* backward_flows,
            int *roffsets, int* rdestinations, int* flow_idx, int *Excess_total)
        {
            // initialising height values and excess flow, Excess_total values
            for(int i = 0; i < V; i++){
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
       
        int minCutMaxFlow(
            int source, int to,
            int *offsets,int *roffsets,
            int *destinations,int *rdestinations,
            int *capacities,int *rcapacities,
            int *flow_index,int *heights,
            int *fflow,int *bflow,int *excess_flow,
            int numNodes,int numEdges
        ){
            int V = numNodes;
            int E = numEdges;
            int sink = to;
            int excessTotal[1]{0};

            // Configure the GPU
            int device = -1;
            cudaGetDevice(&device);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            dim3 num_blocks(deviceProp.multiProcessorCount * numBlocksPerSM);
            dim3 block_size(numThreadsPerBlock);
            size_t sharedMemSize = 3 * block_size.x * sizeof(int);

            // Initilize the flow
            preflow(
                V, source, sink, heights, excess_flow, 
                (offsets), (destinations), (capacities), (fflow), (bflow),
                (roffsets), (rdestinations), (flow_index), excessTotal);

            // GPU structure
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

            // For parameters passing
            void* original_kernel_args[] = {
                &V, &source, &sink, &gpu_height, &gpu_excess_flow, 
                &gpu_offsets, &gpu_destinations, &gpu_capacities, &gpu_fflows, &gpu_bflows, 
                &gpu_roffsets, &gpu_rdestinations, &gpu_flow_index};

            bool *mark, *scanned, *visited;
            mark =      (bool*)malloc(V*sizeof(bool));
            scanned =   (bool*)malloc(V*sizeof(bool));
            visited =   (bool*)malloc(V*sizeof(bool));
            for(int i = 0; i < V; i++){ mark[i] = false; visited[i] = false;}

            // Allocate data for GPU
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

            // Copy to GPU
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

            using namespace std::chrono; 
            auto start = high_resolution_clock::now();
            // algo start
            while((excess_flow[source] + excess_flow[sink]) < *excessTotal){
                // Update GPU values
                (cudaMemcpy(gpu_height,        heights,         V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_excess_flow,   excess_flow,     V*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_fflows,        fflow,           E*sizeof(int), cudaMemcpyHostToDevice));
                (cudaMemcpy(gpu_bflows,        bflow,           E*sizeof(int), cudaMemcpyHostToDevice));

                // Gpu Call
                cudaLaunchCooperativeKernel((void*)push_relabel_kernel, num_blocks, block_size, original_kernel_args, sharedMemSize, 0);
                cudaDeviceSynchronize();
                
                // Get results
                (cudaMemcpy(heights,        gpu_height,         V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(excess_flow,    gpu_excess_flow,    V*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(fflow,          gpu_fflows,         E*sizeof(int), cudaMemcpyDeviceToHost));
                (cudaMemcpy(bflow,          gpu_bflows,         E*sizeof(int), cudaMemcpyDeviceToHost));
                
                // Relable the graph
                global_relabel(
                    V, E, 
                    source,     sink, 
                    heights,    excess_flow,
                    offsets,    destinations, capacities, 
                    fflow,      bflow,
                    roffsets,   rdestinations, flow_index,
                    excessTotal, 
                    mark,       scanned);
            }

            // time ends
            auto end = high_resolution_clock::now();

            // Info Print
            auto nanos      = duration_cast<nanoseconds>(end-start).count();
            auto micros     = duration_cast<microseconds>(end-start).count();
            auto millis     = duration_cast<milliseconds>(end-start).count();
            std::cout << "### " 
                << nanos    << ", " << micros   << ", " << millis   << ", " 
                << V << ", " << E << ", " << source << ", " << sink << ", " << excess_flow[sink] << std::endl;

            // Clear
            (cudaFree(gpu_height));
            (cudaFree(gpu_excess_flow));
            (cudaFree(gpu_offsets));
            (cudaFree(gpu_destinations));
            (cudaFree(gpu_capacities));
            (cudaFree(gpu_fflows));
            (cudaFree(gpu_roffsets));
            (cudaFree(gpu_rdestinations));
            (cudaFree(gpu_bflows));
            (cudaFree(gpu_flow_index));

            // Find node cuts
            std::vector<std::tuple<int, int>> ret;

            return excess_flow[sink];
        }
    };
};


            // printf("offsets: {");
            // for (int i=0; i < V; i++) {
            //     printf("%d, ", offsets[i]);
            // }
            // printf("};\n\n\n");

            // printf("dests: {");
            // for (int i=0; i < E; i++) {
            //     //if(fflow[i] == excess_flow[sink])
            //         printf("%d, ", destinations[i]);
            // }
            // printf("};\n\n\n");

            // printf("caps: {");
            // for (int i=0; i < E; i++) {
            //     //if(fflow[i] == excess_flow[sink])
            //         printf("%d, ", capacities[i]);
            // }
            // printf("};\n\n\n");

            // printf("h: {");
            // for (int i=0; i < V; i++) {
            //     printf("%d, ", heights[i]);
            // }
            // printf("};\n");

            // printf("e: {");
            // for (int i=0; i < V; i++) {
            //     printf("%d, ", excess_flow[i]);
            // }
            // printf("};\n");

            // printf("ff: {");
            // for (int i=0; i < E; i++) {
            //     std::cout << std::setw(5) << fflow[i];
            // }
            // printf("};\n");

            // printf("bf: {");
            // for (int i=0; i < E; i++) {
            //     std::cout << std::setw(5) << bflow[i];
            // }
            // printf("};\n");

            // printf("bf: {");
            // int count = 0;
            // for (int i=0; i < E; i++) {
            //     if(capacities[i] == 1 && bflow[i] == 1 && fflow[i] == 0 && i != source && i != sink){
            //         count++;
            //     }
            //     //printf("%d, ", bflow[i]);
            // }
            // printf("%d\n", count);
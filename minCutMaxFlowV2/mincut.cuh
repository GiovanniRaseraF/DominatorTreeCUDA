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
#include <cuda.h>
#include <cooperative_groups.h>
#include <bits/stdc++.h>
#include <vector>
#include <limits.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // put_time
#include <iostream>
#include <mutex>
#include <thread>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// macros declared
// #define numBlocksPerSM 1
// #define numThreadsPerBlock 1024
// #define numBlocksPerSM 2
// #define numThreadsPerBlock 512
#define numBlocksPerSM BLS
#define numThreadsPerBlock THS
#define INF INT_MAX
#define KERNEL_CYCLES V

// implementation
namespace parallel {
    namespace GoldbergTarjan {
        __global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow,
                                            int *gpu_offsets, int *gpu_destinations, int *gpu_capacities, int *gpu_fflows, int *gpu_bflows,
                                            int *gpu_roffsets, int *gpu_rdestinations, int *gpu_flow_idx){
            // u'th node is operated on by the u'th thread
            grid_group grid = this_grid();
            unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

            // cycle value is set to KERNEL_CYCLES as required
            int cycle = (KERNEL_CYCLES);

            while (cycle > 0){
                for (unsigned int u = idx; u < V; u += blockDim.x * gridDim.x){
                    int e_dash, h_dash, h_double_dash, v, v_dash, d;
                    int v_index = -1; // The index of the edge of u to v_dash
                    bool vinReverse = false;

                    //  Find the activate nodes
                    if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink){
                        e_dash = gpu_excess_flow[u];
                        h_dash = INF;
                        v_dash = -1; // Modify from NULL to -1

                        // For all (u, v) belonging to E_f (residual graph edgelist)
                        // Find (u, v) in both CSR format and revesred CSR format
                        for (int i = gpu_offsets[u]; i < gpu_offsets[u + 1]; i++){
                            v = gpu_destinations[i];
                            if (gpu_fflows[i] > 0){
                                h_double_dash = gpu_height[v];
                                if (h_double_dash < h_dash){
                                    v_dash = v;
                                    h_dash = h_double_dash;
                                    v_index = i;
                                    vinReverse = false;
                                }
                            }
                        }
                        // Find (u, v) in reversed CSR format
                        for (int i = gpu_roffsets[u]; i < gpu_roffsets[u + 1]; i++){
                            v = gpu_rdestinations[i];
                            int flow_idx = gpu_flow_idx[i];

                            if (gpu_bflows[flow_idx] > 0){
                                h_double_dash = gpu_height[v];
                                if (h_double_dash < h_dash){
                                    v_dash = v;
                                    h_dash = h_double_dash;
                                    v_index = flow_idx; // Find the bug here!!!
                                    vinReverse = true;
                                }
                            }
                        }

                        /* Push operation */
                        if (v_dash == -1){
                            gpu_height[u] = V;
                        }
                        else{
                            if (gpu_height[u] > h_dash){

                                if (!vinReverse){
                                    if (e_dash > gpu_fflows[v_index]){
                                        d = gpu_fflows[v_index];
                                    }else{
                                        d = e_dash;
                                    }

                                    /* Push flow to residual graph */
                                    atomicAdd(&gpu_bflows[v_index], d);
                                    atomicSub(&gpu_fflows[v_index], d);

                                    /* Update Excess Flow */
                                    atomicAdd(&gpu_excess_flow[v_dash], d);
                                    atomicSub(&gpu_excess_flow[u], d);
                                }else{
                                    if (e_dash > gpu_bflows[v_index]){
                                        d = gpu_bflows[v_index];
                                    }else{
                                        d = e_dash;
                                    }

                                    /* Push flow to residual graph */
                                    atomicAdd(&gpu_fflows[v_index], d);
                                    atomicSub(&gpu_bflows[v_index], d);

                                    /* Update Excess Flow */
                                    atomicAdd(&gpu_excess_flow[v_dash], d);
                                    atomicSub(&gpu_excess_flow[u], d);
                                }
                            }else{
                                gpu_height[u] = h_dash + 1;
                            }
                        }
                    }
                }

                // cycle value is decreased
                cycle = cycle - 1;
                grid.sync();
            }
        }

    }

    void global_relabel(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, 
                int *cpu_offsets, int *cpu_destinations, int* cpu_capacities, int* cpu_fflows, int* cpu_bflows, 
                int* cpu_roffsets, int* cpu_rdestinations, int* cpu_flow_idx,
                int *Excess_total, bool *mark, bool *scanned
        ){
        for (int u = 0; u < V; u++) {
            for (int i = cpu_offsets[u]; i < cpu_offsets[u + 1]; i++) {
                int v = cpu_destinations[i];
                if (cpu_height[u] > cpu_height[v] + 1) {
                    int flow;
                    if (cpu_excess_flow[u] < cpu_fflows[i]) {
                        flow = cpu_excess_flow[u];
                    } else {
                        flow = cpu_fflows[i];
                    }

                    cpu_excess_flow[u] -= flow;
                    cpu_excess_flow[v] += flow;
                    cpu_bflows[i] += flow;
                    cpu_fflows[i] -= flow;
                }
            }
        }

        std::list<int> Queue;
        int x,y,current;
    
        for(int i = 0; i < V; i++){
            scanned[i] = false;
        }

        // Enqueueing the sink and set scan(sink) to true 
        Queue.push_back(sink);
        scanned[sink] = true;
        cpu_height[sink] = 0;

        // bfs routine and assigning of height values with tree level values
        while(!Queue.empty()){
            // dequeue
            x = Queue.front();
            Queue.pop_front();

            // capture value of current level
            current = cpu_height[x];
        
            // increment current value
            current = current + 1;

            for(int i = cpu_roffsets[x]; i < cpu_roffsets[x + 1]; i++){
                y = cpu_rdestinations[i];
                int flow_index = cpu_flow_idx[i];
            
                if (cpu_fflows[flow_index] > 0) {
                    if(scanned[y] == false){
                        cpu_height[y] = current;
                        scanned[y] = true;
                        Queue.push_back(y);
                    }
                }

            }

            for (int i = cpu_offsets[x]; i < cpu_offsets[x + 1]; i++) {
                y = cpu_destinations[i];
                int flow_index = i;
            
                if (cpu_bflows[flow_index] > 0) {
                    if(scanned[y] == false){
                        cpu_height[y] = current;
                        scanned[y] = true;
                        Queue.push_back(y);
                    }
                }

            }
        }
        bool if_all_are_relabeled = true;

        for(int i = 0; i < V; i++){
            if(scanned[i] == false){
                if_all_are_relabeled = false;
                break;
            }
        }

        // if not all nodes are relabeled
        if(if_all_are_relabeled == false){
            for(int i = 0; i < V; i++){
                if( !( (scanned[i] == true) || (mark[i] == true) ) ){
                    mark[i] = true;
                    *Excess_total = *Excess_total - cpu_excess_flow[i];
                }
            }
        }
    }
}
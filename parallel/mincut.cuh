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
#include <deque>
#include <cuda_runtime.h>
#include <iomanip> // put_time
#include <iostream>
#include <mutex>
#include <thread>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// macros declared
#define numBlocksPerSM BLS
#define numThreadsPerBlock THS
#define INF V
#define KERNEL_CYCLES V

// implementation
namespace parallel {
    namespace GoldbergTarjan {
        __global__ void push_relabel_kernel(
            int V, 
            int source,             int sink, 
            int *gpu_height,        int *gpu_excess_flow,
            int *gpu_offsets,       int *gpu_destinations, 
            int *gpu_capacities,    
            int *gpu_fflows,        int *gpu_bflows,
            int *gpu_roffsets,      int *gpu_rdestinations, 
            int *gpu_flow_idx
        ){
            grid_group grid = this_grid();
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            int cycle = (KERNEL_CYCLES);

            while (cycle > 0){
                int countU = 0;
                for (int u = idx; u < V; u += blockDim.x * gridDim.x){
                    countU ++;
                    int e_dash, h_dash, h_double_dash, v, v_dash, d;
                    int v_index = -1; // The index of the edge of u to v_dash
                    bool vinReverse = false;

                    //  Find the activate nodes
                    if (gpu_excess_flow[u] > 0 && gpu_height[u] < V && u != source && u != sink){
                        e_dash = gpu_excess_flow[u];
                        h_dash = INF;
                        v_dash = -1; 

                        // Find (u, v) in both CSR format
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
                        // Find (u, v) in reversed CSR 
                        for (int i = gpu_roffsets[u]; i < gpu_roffsets[u + 1]; i++){
                            v = gpu_rdestinations[i];
                            int flow_idx = gpu_flow_idx[i];

                            if (gpu_bflows[flow_idx] > 0){
                                h_double_dash = gpu_height[v];
                                if (h_double_dash < h_dash){
                                    v_dash = v;
                                    h_dash = h_double_dash;
                                    v_index = flow_idx; 
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

                                    // Push flow to reverse graph
                                    atomicAdd(&gpu_bflows[v_index], d);
                                    atomicSub(&gpu_fflows[v_index], d);
                                    atomicAdd(&gpu_excess_flow[v_dash], d);
                                    atomicSub(&gpu_excess_flow[u], d);
                                }else{
                                    if (e_dash > gpu_bflows[v_index]){
                                        d = gpu_bflows[v_index];
                                    }else{
                                        d = e_dash;
                                    }

                                    // Push flow to graph
                                    atomicAdd(&gpu_fflows[v_index], d);
                                    atomicSub(&gpu_bflows[v_index], d);
                                    atomicAdd(&gpu_excess_flow[v_dash], d);
                                    atomicSub(&gpu_excess_flow[u], d);
                                }
                            }else{
                                gpu_height[u] = h_dash + 1;
                            }
                        }
                    }
                }

                if(countU > 0){
                    printf("countU: %d\n", countU);
                }
                
                cycle = cycle - 1;
                grid.sync();
            }
        }

    }

    // This function allows to decrease the Excess_total in
    // order to let the algoritm terminate
    void global_relabel(
        int V,          int E, 
        int source,     int sink, 
        int *height,    int *excess_flow, 
        int *offsets,   int *destinations, int* capacities, 
        int* fflows,    int* bflows, 
        int* roffsets,  int* rdestinations, 
        int* flow_idx,
        int *Excess_total, 
        bool *mark,     bool *scanned
    ){
        // for (int u = 0; u < V; u++) {
        //     for (int i = offsets[u]; i < offsets[u + 1]; i++) {
        //         int v = destinations[i];
        //         if (height[u] > height[v] + 1) {
        //             int flow;
        //             if (excess_flow[u] < fflows[i]) {
        //                 flow = excess_flow[u];
        //             } else {
        //                 flow = fflows[i];
        //             }

        //             excess_flow[u] -= flow;
        //             excess_flow[v] += flow;
        //             bflows[i] += flow;
        //             fflows[i] -= flow;
        //         }
        //     }
        // }

        std::deque<int> Queue;
        int x,y,current;
    
        for(int i = 0; i < V; i++){
            scanned[i] = false;
        }

        // Enqueueing the sink and set scan(sink) to true 
        Queue.push_back(sink);
        scanned[sink] = true;
        height[sink] = 0;

        // bfs routine and assigning of height values with tree level values
        while(!Queue.empty()){
            // dequeue
            x = Queue.front();
            Queue.pop_front();

            // capture value of current level
            current = height[x];
        
            // increment current value
            current = current + 1;

            for(int i = roffsets[x]; i < roffsets[x + 1]; i++){
                y = rdestinations[i];
                int flow_index = flow_idx[i];
            
                if (fflows[flow_index] > 0) {
                    if(scanned[y] == false){
                        height[y] = current;
                        scanned[y] = true;
                        Queue.push_back(y);
                    }
                }

            }

            for (int i = offsets[x]; i < offsets[x + 1]; i++) {
                y = destinations[i];
                int flow_index = i;
            
                if (bflows[flow_index] > 0) {
                    if(scanned[y] == false){
                        height[y] = current;
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
                if(scanned[i] != true && mark[i] != true){
                    mark[i] = true;
                    *Excess_total = *Excess_total - excess_flow[i];
                }
            }
        }
    }
}
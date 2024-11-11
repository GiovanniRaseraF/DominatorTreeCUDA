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
#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define numBlocksPerSM 1
#define numThreadsPerBlock 1024
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF INT_MAX
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V
#define TILE_SIZE 32

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
}
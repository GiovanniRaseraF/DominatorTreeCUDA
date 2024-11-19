// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.hpp"
#include "loader.hpp"

void run(std::string filename, int from, int to){
    int num_nodes{0};
    int num_edges{0};
    int num_edges_processed{0};
    int source_node{0};
    int sink_node{0};
    std::vector<int> graph_destinations{};
    std::vector<int> graph_offsets{};
    std::vector<int> graph_capacities{};

    // Load from specified file
    loader::buildFromTxtFile(
        filename, 
        num_nodes,      num_edges,
        num_edges_processed,
        source_node,    sink_node,
        graph_destinations, graph_offsets, graph_capacities
    );

    // Preparing all pointer for cpu graph
    int *offsets        = nullptr;
    int *roffsets       = nullptr;
    int *destinations   = nullptr;
    int *rdestinations  = nullptr;
    int *capacities     = nullptr;
    int *rcapacities    = nullptr;
    int *forward_flows  = nullptr;
    int *backward_flows = nullptr;
    int *flow_index     = nullptr;
    int *heights        = nullptr;
    int *excesses       = nullptr;

    // Loading the data from file to a CSR representation
    loader::buildFromCSRGraph(
        num_nodes,          num_edges,
        num_edges_processed,
        source_node,        sink_node,
        graph_destinations, graph_offsets,
        graph_capacities,   offsets,
        destinations,       capacities,
        rcapacities,        forward_flows,
        roffsets,           rdestinations,
        backward_flows,     flow_index,
        heights,            excesses
    );

    // Find MinCut
    auto micCutValue = parallel::GoldbergTarjan::minCutMaxFlow(
        from,               to,
        offsets,            roffsets,
        destinations,       rdestinations,
        capacities,         rcapacities,
        flow_index,         heights,
        forward_flows,      backward_flows,
        excesses,
        num_nodes,          num_edges
    );

    // Find what to cut
    std::vector<std::tuple<int, int>> ret;
    for(int i = 0; i < num_nodes; ++i){
      for(int j = offsets[i]; j < offsets[i+1]; j++){
        int dest = destinations[j];
        if( // Cut node condition
            excesses[i] == num_nodes*2-1 && 
            forward_flows[j] == 0 && 
            backward_flows[j] == 1
        ){
            ret.push_back({i, dest});
        }
      }
    }            

    // Cuts
    std::cout << "Nodes to remove in G are: " << std::endl;
    for(auto r : ret){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << "(" << from / 2 << ") : " << from << " -/-> " << to << std::endl; 
    }

    // Clear
    free(offsets);
    free(roffsets);
    free(destinations);
    free(rdestinations);
    free(capacities);
    free(rcapacities);
    free(forward_flows);
    free(backward_flows);
    free(flow_index);
    free(heights);
    free(excesses);
}
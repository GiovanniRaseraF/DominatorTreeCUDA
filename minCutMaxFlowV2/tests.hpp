// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.hpp"
#include "loader.hpp"

void test(){
    std::string filename;
    std::cout << "filename>>"; std::cin >> filename;

    int num_nodes{0};
    int num_edges{0};
    int num_edges_processed{0};
    int source_node{0};
    int sink_node{0};
    std::vector<int> graph_destinations{};
    std::vector<int> graph_offsets{};
    std::vector<int> graph_capacities{};

    loader::buildFromTxtFile(
        filename, 
        num_nodes,
        num_edges,
        num_edges_processed,
        source_node,
        sink_node,
        graph_destinations,
        graph_offsets,
        graph_capacities
    );

    std::cout << num_nodes << std::endl;
    std::cout << num_edges << std::endl;
    std::cout << num_edges_processed << std::endl;
    std::cout << source_node << std::endl;
    std::cout << sink_node << std::endl;

    int from = 0;
    int to = 0;
    std::cout << "source>>"; std::cin >> from;
    std::cout << "to>>"; std::cin >> to;

    int *offsets = nullptr;
    int *roffsets = nullptr;
    int *destinations = nullptr;
    int *rdestinations = nullptr;
    int *capacities = nullptr;
    int *rcapacities = nullptr;
    int *forward_flows = nullptr;
    int *backward_flows = nullptr;
    int *flow_index = nullptr;
    int *heights = nullptr;
    int *excesses = nullptr;

    loader::buildFromCSRGraph(
        num_nodes,num_edges,
        num_edges_processed,
        source_node,sink_node,
        graph_destinations,graph_offsets,
        graph_capacities,offsets,
        destinations,capacities,
        rcapacities,forward_flows,
        roffsets,rdestinations,
        backward_flows,flow_index,
        heights,excesses
    );

    parallel::GoldbergTarjan::minCutMaxFlow(
        from, to,
        offsets,roffsets,
        destinations,rdestinations,
        capacities,rcapacities,
        flow_index,heights,
        forward_flows,backward_flows,
        excesses,num_nodes,num_edges
    );
}
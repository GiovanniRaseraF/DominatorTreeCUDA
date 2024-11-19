// Author: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda/graph.cpp
// Modified by: GiovanniRaseraF

#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>

namespace loader{
    /*  This function loads a graph from a txt file
        formatted in this form:
            # Nodes: %num_nodes Edges: %num_edges
            u v c
        
        this meas:
            My graphs has num_nodes nodes and num_edges edges
            u -- c --> v
    */
    void buildFromTxtFile(
        const std::string &filename,
        int &num_nodes,
        int &num_edges,
        int &num_edges_processed,
        int &source_node,
        int &sink_node,
        std::vector<int> &destinations,
        std::vector<int> &offsets,
        std::vector<int> &capacities){
        std::ifstream file(filename);
        if (file.fail()){
            fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
            exit(1);
        }
        std::string line;
        std::unordered_map<int, std::vector<int>> adjacency_list;
        std::unordered_map<int, std::vector<int>> cap_list;

        int cnt = 0;
        while (std::getline(file, line)){
            std::stringstream ss(line);
            if (line.find("# Nodes:") != std::string::npos){
                sscanf(line.c_str(), "# Nodes: %d Edges: %d", &num_nodes, &num_edges);
                // num_edges += num_nodes;
                // num_nodes *= 2;
            }

            if (ss.str()[0] == '#')
                continue;
            int from, to, cap;
            ss >> from >> to >> cap;
            // basic one 
            // adjacency_list[from*2+1].push_back(to*2);
            // cap_list[from*2+1].push_back((num_edges+num_nodes)*2);
            adjacency_list[from].push_back(to);
            cap_list[from].push_back(cap);
            cnt++;
        }

        // for(int i = 0; i < num_nodes/2; i++){
        //     adjacency_list[i*2].push_back(i*2+1);
        //     cap_list[i*2].push_back(1);
        // }

        // num_nodes = adjacency_list.size();
        offsets.push_back(0);
        for (int i = 0; i < num_nodes; ++i){
            // some nodes have no out edges
            if (adjacency_list.count(i) == 0){
                offsets.push_back(offsets.back());
                continue;
            }
            sort(adjacency_list[i].begin(), adjacency_list[i].end());
            for (int neighbor : adjacency_list[i]){
                destinations.push_back(neighbor);
            }
            for (int cap : cap_list[i]){
                capacities.push_back(cap);
            }
            offsets.push_back(destinations.size());
        }

        num_edges_processed = cnt;
    }

    // Build the Graph for Node Cut
    void buildFromTxtFileForNodeCut(
        const std::string &filename,
        int &num_nodes,
        int &num_edges,
        int &num_edges_processed,
        int &source_node,
        int &sink_node,
        std::vector<int> &destinations,
        std::vector<int> &offsets,
        std::vector<int> &capacities){
        std::ifstream file(filename);
        if (file.fail()){
            fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
            exit(1);
        }
        std::string line;
        std::unordered_map<int, std::vector<int>> adjacency_list;
        std::unordered_map<int, std::vector<int>> cap_list;

        int cnt = 0;
        while (std::getline(file, line)){
            std::stringstream ss(line);
            if (line.find("# Nodes:") != std::string::npos){
                sscanf(line.c_str(), "# Nodes: %d Edges: %d", &num_nodes, &num_edges);
                num_edges += num_nodes;
                num_nodes *= 2;
            }

            if (ss.str()[0] == '#')
                continue;
            int from, to, cap;
            ss >> from >> to >> cap;
            // basic one 
            adjacency_list[from*2+1].push_back(to*2);
            cap_list[from*2+1].push_back(num_nodes*2);
            cnt++;
        }

        for(int i = 0; i < num_nodes/2; i++){
            adjacency_list[i*2].push_back(i*2+1);
            cap_list[i*2].push_back(1);
        }

        // num_nodes = adjacency_list.size();
        offsets.push_back(0);
        for (int i = 0; i < num_nodes; ++i){
            // some nodes have no out edges
            if (adjacency_list.count(i) == 0){
                offsets.push_back(offsets.back());
                continue;
            }
            sort(adjacency_list[i].begin(), adjacency_list[i].end());
            for (int neighbor : adjacency_list[i]){
                destinations.push_back(neighbor);
            }
            for (int cap : cap_list[i]){
                capacities.push_back(cap);
            }
            offsets.push_back(destinations.size());
        }

        num_edges_processed = cnt;
    }

    /*  This function create the CSR representation of the graph
        this is useful when using the graph in the GPU
    */
    void buildFromCSRGraph(
        int num_nodes,
        int num_edges,
        int &num_edges_processed,
        int &source_node,
        int &sink_node,
        std::vector<int> &graph_destinations,
        std::vector<int> &graph_offsets,
        std::vector<int> &graph_capacities,
        // to load
        int *&offsets,
        int *&destinations,
        int *&capacities,
        int *&rcapacities,
        int *&forward_flows,
        int *&roffsets,
        int *&rdestinations,
        int *&backward_flows,
        int *&flow_index,
        int *&heights,
        int *&excesses
    ){

        /* Allocate offsets, destinations, capacities, flows, roffsets, rdestinations, rflows */
        offsets =           (int *)malloc(sizeof(int) * (num_nodes + 1));
        destinations =      (int *)malloc(sizeof(int) * num_edges);
        capacities =        (int *)malloc(sizeof(int) * num_edges);
        rcapacities =       (int *)malloc(sizeof(int) * num_edges);
        forward_flows =     (int *)malloc(sizeof(int) * num_edges);
        roffsets =          (int *)malloc(sizeof(int) * (num_nodes + 1));
        rdestinations =     (int *)malloc(sizeof(int) * num_edges);
        backward_flows =    (int *)malloc(sizeof(int) * num_edges);
        flow_index =        (int *)malloc(sizeof(int) * num_edges);
        heights =           (int *)malloc(sizeof(int) * num_nodes);
        excesses =          (int *)malloc(sizeof(int) * num_nodes);

        for (int i = 0; i < num_nodes; i++){
            heights[i] = 0;
            excesses[i] = 0;
        }

        // Initialize offset vectors
        for (int i = 0; i < num_nodes + 1; ++i){
            offsets[i] = graph_offsets[i];
            roffsets[i] = 0;
        }
        for (int i = 0; i < num_edges; ++i){
            destinations[i] = graph_destinations[i];
            capacities[i] = graph_capacities[i];
            forward_flows[i] = graph_capacities[i]; // The initial residual flow is the same as capacity, not 0
            backward_flows[i] = 0;
        }

        std::vector<int> backward_counts(num_nodes, 0);

        // Count the number of edges for each node to prepare the offset vectors
        for (int i = 0; i < num_nodes; ++i){
            for (int j = graph_offsets[i]; j < graph_offsets[i + 1]; ++j){
                backward_counts[graph_destinations[j]]++;
            }
        }

        // Convert counts to actual offsets
        for (int i = 1; i <= num_nodes; ++i){
            roffsets[i] = roffsets[i - 1] + backward_counts[i - 1];
        }

        // Initialize backward count vector
        backward_counts.clear();
        for (int i = 0; i < num_nodes; ++i){
            backward_counts[i] = 0;
        }

        // Fill forward and backward edges
        for (int i = 0; i < num_nodes; ++i){
            for (int j = graph_offsets[i]; j < graph_offsets[i + 1]; ++j){
                int dest = graph_destinations[j];

                // Corresponding backward edge
                int backward_index = roffsets[dest] + backward_counts[dest];
                rdestinations[backward_index] = i;
                backward_counts[dest]++;
            }
        }

        // Initialize flow index
        for (int u = 0; u < num_nodes; u++){
            for (int i = roffsets[u]; i < roffsets[u + 1]; i++){
                int v = rdestinations[i];

                // Find the forward edge index
                for (int j = offsets[v]; j < offsets[v + 1]; j++){
                    if (destinations[j] == u){
                        flow_index[i] = j;
                        break;
                    }
                }
            }
        }
    }
};
#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace loader {
    void buildFromTxtFile(
        const std::string &filename, 
        int &num_nodes,
        int &num_edges,
        int &num_edges_processed,
        int &source_node,
        int &sink_node,
        std::vector<int> &destinations,
        std::vector<int> &offsets,
        std::vector<int> &capacities) {
        std::ifstream file(filename);
    if (file.fail()) {
        fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
        exit(1);
    }
    std::string line;
    std::unordered_map<int, std::vector<int>> adjacency_list;
    std::unordered_map<int, std::vector<int>> cap_list;
    int cnt = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        if (line.find("# Nodes:") != std::string::npos)
        sscanf(line.c_str(), "# Nodes: %d Edges: %d", &num_nodes, &num_edges);

        if (ss.str()[0] == '#')
        continue;
        int from, to, cap;
        ss >> from >> to >> cap;
        adjacency_list[from].push_back(to);
        cap_list[from].push_back(cap);
        cnt++;
    }

    // num_nodes = adjacency_list.size();
    offsets.push_back(0);
    for (int i = 0; i < num_nodes; ++i) {
        // some nodes have no out edges
        if (adjacency_list.count(i)==0) {
        offsets.push_back(offsets.back());
        continue;
        }
        sort(adjacency_list[i].begin(), adjacency_list[i].end());
        for (int neighbor : adjacency_list[i]) {
        destinations.push_back(neighbor);
        }
        for (int cap: cap_list[i]) {
        capacities.push_back(cap);
        }
        offsets.push_back(destinations.size());
    }

    num_edges_processed = cnt;
    }   
};
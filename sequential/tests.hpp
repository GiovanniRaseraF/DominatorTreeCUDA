// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include "mincut.hpp"

void justInitGraph(Graph &graph, Graph &rGraph) {
    std::cout << "Init G and rG with V: " << graph.size() << std::endl;
    for(int i = 0; i < graph.size(); i ++){
        for(int j = 0; j < graph.size(); j++){
            graph[i].push_back(0);
            rGraph[i].push_back(0);
        }
    }
}

// artifichal
void generateFromStartToFinish(Graph &graph){
    for(int i = 0; i < graph.size()-1; i++){
        graph[i][i+1] = 1;
    }
}



void testFile(std::string filename, int source, int to){
    std::cout << "\nTest from file" << std::endl;
    int VNodes = 0;
    int Eedges = 0;

    std::ifstream file(filename);
    if (file.fail()){
        fprintf(stderr, "\"%s\" does not exist!\n", filename.c_str());
        exit(1);
    }

    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    if (line.find("# Nodes:") != std::string::npos)
        sscanf(line.c_str(), "# Nodes: %d Edges: %d", &VNodes, &Eedges);

    Graph rGraph(VNodes);
    Graph graph(VNodes);

    justInitGraph(graph, rGraph);

    while (std::getline(file, line)){
        std::stringstream ss(line);
        int from, to, cap;
        ss >> from >> to >> cap;
        graph[from][to] = cap;
    }

    Graph rGraphPrime(VNodes*2);
    Graph graphPrime(VNodes*2);   

    justInitGraph(graphPrime, rGraphPrime);

    int sourcePrime = source; 
    int toPrime = to*2;

    // build G'
    sequential::FordFulkerson::buildGPrimeFromG(graph, graphPrime);
    // we need to pay attention to the start, because the cut must start form v'odd
    auto result = sequential::FordFulkerson::minCutMaxFlow(graphPrime, rGraphPrime, sourcePrime+1, toPrime);
    //auto result = sequential::FordFulkerson::minCutMaxFlow(graphPrime, rGraphPrime, sourcePrime+1, toPrime);

    // print result
    std::cout << "Nodes D are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << "(" << from / 2 << ") : " << from << " -/-> " << to << std::endl;
    }

    std::cout << "|D|: " << result.size() << std::endl;
}
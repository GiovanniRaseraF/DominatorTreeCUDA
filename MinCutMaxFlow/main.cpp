// Author: Giovanni Rasera

#include <iostream>
#include "mincut.hpp"

void justInitGraph(Graph &graph, Graph &rGraph) {
    std::cout << "Init G and rG with V: " << V << std::endl;
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < V; j++){
            graph[i].push_back(0);
            rGraph[i].push_back(0);
        }
    }
}

int main(){
    Graph rGraph(V);
    Graph graph(V);

    justInitGraph(graph, rGraph);

    auto result = sequential::Default::minCutMaxFlow(graph, rGraph, 0, V-1);

    // print result
    std::cout << "Edges to remove are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from << " --> " << to << std::endl;
    }
}
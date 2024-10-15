#include "mincut.cuh"
#include <iostream>

void justInitGraph(Graph &graph, Graph &rGraph) {
    std::cout << "Init G and rG with V: " << graph.size() << std::endl;
    for(int i = 0; i < graph.size(); i ++){
        for(int j = 0; j < graph.size(); j++){
            graph[i].push_back(0);
            rGraph[i].push_back(0);
        }
    }
}

int main(){
    Graph G(10);
    Graph Gf(10);
    ExcessFlow e(10);
    Height h(10);
    justInitGraph(G, Gf);

    parallel::GoldbergTarjan::MinCutMaxFlow(G, Gf, e, h); 
}
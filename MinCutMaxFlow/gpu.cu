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
    int V = 10;
    Graph G(V);
    Graph Gf(V);
    ExcessFlow e(V);
    Height h(V);
    justInitGraph(G, Gf);
    int source = 0;
    int to = V-1;

    parallel::GoldbergTarjan::MinCutMaxFlow(G, Gf, e, h, source, to); 
}
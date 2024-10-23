// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.cuh"

#ifndef V
#define V 10
#endif

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

void test1(){
    std::cout << "\nTest with V elements and a connection from 0 to V-1" << std::endl;
    int source = 0;
    int to = V-1;
    Graph rGraph(V);
    Graph graph(V);
    ExcessFlow e(V);
    Height h(V);

    justInitGraph(graph, rGraph);
    generateFromStartToFinish(graph);

    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to); 
}
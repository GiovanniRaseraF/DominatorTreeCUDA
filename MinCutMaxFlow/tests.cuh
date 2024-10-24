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

void print(const Graph &graph) {
    std::cout << std::endl;
    for(int i = 0; i < graph.size(); i ++){
        for(int j = 0; j < graph.size(); j++){
            std::cout << graph[i][j] << " ";
        }
        std::cout << std::endl;
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

    print(graph);
    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to); 
}

void test2(){
    std::cout << "\nTest with custom graph design with Professor" << std::endl;
    const int numberOfNodes = 7;
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);

    int source = 0;
    int to = 6;
    //
    graph[source][1] = 1;
    graph[source][3] = 1;
    graph[source][4] = 1;
    graph[source][5] = 1;

    graph[5][2] = 1;
    graph[1][2] = 1;
    graph[3][2] = 1;
    graph[4][2] = 1;

    graph[2][to] = 1;
    //

    ExcessFlow e(numberOfNodes);
    Height h(numberOfNodes);

    print(graph);
    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to); 
}


void test3(){
    std::cout << "\nTest with custom graph with more then connecition from" << std::endl;
    const int numberOfNodes = 10;
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);

    int source = 0;
    int to = numberOfNodes-1;
    //
    graph[source][1] = 1;
    graph[source][3] = 1;
    graph[source][4] = 1;
    graph[source][5] = 1;

    graph[5][2] = 1;
    graph[1][2] = 1;
    graph[3][2] = 1;
    graph[4][2] = 1;

    graph[2][to] = 2;
    graph[2][6] = 1;
    graph[6][7] = 1;
    graph[7][8] = 1;
    graph[8][to] = 1;
    //

    ExcessFlow e(numberOfNodes);
    Height h(numberOfNodes);

    print(graph);
    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to);
}

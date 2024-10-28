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

    ExcessFlow e(numberOfNodes, 0);
    Height h(numberOfNodes, 0);

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
    graph[source][to] = 100;
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

    ExcessFlow e(numberOfNodes, 0);
    Height h(numberOfNodes, 0);

    print(graph);
    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to);
}

void test4(){
    std::cout << "\nTest with custom graph with more then connecition from" << std::endl;
    const int numberOfNodes = 6;
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);

    int source = 0;
    int to = numberOfNodes-1;
    //
    // Creating above shown flow network 
    graph[0][1] =  16;
    graph[0][2] =  13;
    graph[1][2] =  10;
    graph[2][1] =  4;
    graph[1][3] =  12;
    graph[2][4] =  14;
    graph[3][2] =  9;
    graph[3][5] =  20;
    graph[4][3] =  7;
    graph[4][5] =  4;
    //

    ExcessFlow e(numberOfNodes, 0);
    Height h(numberOfNodes, 0);

    print(graph);
    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to);
}

void test7(){
    std::cout << "\nTest paper graph" << std::endl;
    const int numberOfNodes = 7;
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);

    int source = 0;
    int to = numberOfNodes-1;
    //
    graph[source][1] = 3;
    graph[source][2] = 9;
    graph[source][3] = 5;
    graph[source][4] = 6;
    graph[source][5] = 2;

    graph[1][2] = 3;
    graph[2][3] = 3;
    graph[2][1] = 3;
    graph[3][2] = 3;
    graph[4][3] = 4;
    graph[5][4] = 1;


    graph[1][to] = 10;
    graph[2][to] = 2;
    graph[3][to] = 1;
    graph[4][to] = 8;
    graph[5][to] = 9;

    ExcessFlow e(numberOfNodes, 0);
    Height h(numberOfNodes, 0);

    parallel::GoldbergTarjan::minCutMaxFlow(graph, rGraph, e, h, source, to);

    // print result
    //std::cout << "Edges to remove are: " << std::endl;
    // for(auto r : result){
    //     int from = std::get<0>(r);
    //     int to = std::get<1>(r);

    //     std::cout << from << " --> " << to << std::endl;
    // }
}
// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.hpp"

constexpr int V = 1024;

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
    Graph rGraph(V);
    Graph graph(V);

    justInitGraph(graph, rGraph);
    generateFromStartToFinish(graph);

    auto result = sequential::FordFulkerson::minCutMaxFlow(graph, rGraph, 0, V-1);

    // print result
    std::cout << "Edges to remove are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from << " --> " << to << std::endl;
    }
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

    auto result = sequential::FordFulkerson::minCutMaxFlow(graph, rGraph, source, to);

    // print result
    std::cout << "Edges to remove are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from << " --> " << to << std::endl;
    }
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

    auto result = sequential::FordFulkerson::minCutMaxFlow(graph, rGraph, source, to);

    // print result
    std::cout << "Edges to remove are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from << " --> " << to << std::endl;
    }
}

void test4(){
    std::cout << "\nTest with double V" << std::endl;
    const int numberOfNodes = 7;
    // original
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);
    int source = 0;
    int to = numberOfNodes-1;

    // G'
    Graph rGraphPrime(numberOfNodes*2);
    Graph graphPrime(numberOfNodes*2);
    justInitGraph(graphPrime, rGraphPrime);
    int sourcePrime = source; 
    int toPrime = to*2; 
    
    // building the original
    graph[source][1] = 1;
    graph[source][3] = 1;
    graph[source][4] = 1;
    graph[source][5] = 1;

    graph[5][2] = 1;
    graph[1][2] = 1;
    graph[3][2] = 1;
    graph[4][2] = 1;

    graph[2][to] = 1;

    // build G'
    sequential::FordFulkerson::buildGPrimeFromG(graph, graphPrime);
    // we need to pay attention to the start, because the cut must start form v'odd
    auto result = sequential::FordFulkerson::minCutMaxFlow(graphPrime, rGraphPrime, sourcePrime+1, toPrime);

    // print result
    std::cout << "Nodes to remove in G are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from / 2  << std::endl; 
    }
}

void test5(){
    std::cout << "\nTest with modified original graph" << std::endl;
    const int numberOfNodes = 10;
    // original
    Graph rGraph(numberOfNodes);
    Graph graph(numberOfNodes);
    justInitGraph(graph, rGraph);
    int source = 0;
    int to = numberOfNodes-1;

    // G'
    Graph rGraphPrime(numberOfNodes*2);
    Graph graphPrime(numberOfNodes*2);
    justInitGraph(graphPrime, rGraphPrime);
    int sourcePrime = source; 
    int toPrime = to*2; 
    
    // building the original
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

    // build G'
    sequential::FordFulkerson::buildGPrimeFromG(graph, graphPrime);
    // we need to pay attention to the start, because the cut must start form v'odd
    auto result = sequential::FordFulkerson::minCutMaxFlow(graphPrime, rGraphPrime, sourcePrime+1, toPrime);

    // print result
    std::cout << "Nodes to remove in G are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from / 2 << std::endl; 
    }
}
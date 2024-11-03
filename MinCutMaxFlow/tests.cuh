// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.cuh"

constexpr int V = 7;

void test1(){
    std::cout << "Test 1" << std::endl;
    Graph G(V, std::vector<int>(V, 0));
    parallel::GoldbergTarjan::minCutMaxFlow(G, 0, 3);
}
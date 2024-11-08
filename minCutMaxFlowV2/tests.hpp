// Author: Giovanni Rasera

#pragma once
#include <iostream>
#include "mincut.hpp"

constexpr int V = 7;

void test1(){
    std::cout << "Test 1" << std::endl;
    Graph G(V, std::vector<int>(V, 0));
    int from = 0;
    int to = 0;
    std::cout << "source:"; std::cin >> from;
    std::cout << "to:"; std::cin >> to;
    parallel::GoldbergTarjan::minCutMaxFlow(G, from, to);
}
// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <deque>
#include <queue>
#include <list>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include "graph.hpp"

namespace sequential{
    // Karger's Algorithm
    namespace Kalger{
        // Source: Algorithm 2: Initialize(G)
        void Initialize(Graph &G){ // G = (V, E)
            // pre
            auto &V = G.V;
            auto &E = G.E;
            
            // implementation
            std::cout << "sequential::Kalger::Initialize(G)" << std::endl;
            std::set<Supernode> Tau{};  // the set of supernodes
            std::set<Superedge> F{};    // the set of superedges

            for(auto &v : V){
                auto v_ = Supernode();
                // TODO: V(v_) = {v};
                Tau.insert(v_); // TODO: supernode must be comparable
            }

            for(auto &e : E){
                //
            }
        }
    }
};
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
#include <tuple>
#include "graph.hpp"

namespace sequential{
    // Karger's Algorithm
    namespace Kalger{
        // Source: Algorithm 2: Initialize(G)
        std::tuple<std::set<Supernode>, std::set<Superedge>> Initialize(Graph &G){ // G = (V, E)
            // pre
            auto &V = G.V;
            auto &E = G.E;
            
            // implementation
            std::cout << "sequential::Kalger::Initialize(G)" << std::endl;
            std::set<Supernode> Tau{};  // the set of supernodes
            std::set<Superedge> F{};    // the set of superedges

            for(auto &v : V){
                auto v_ = Supernode(v);     // u_       <- new supernode
                v_.insert(v);               // V(u_)    <- {v} //TODO: is insertion or assignment in the set of v_ ? 
                Tau.insert(v_);             // Tau      <- Tau U {v_}
            }

            for(auto &e : E){
                auto Euv = Superedge(e);    // Euv      <- {(u, v)}
                Euv.insert(e);              // 
                F.insert(Euv);              // F        <- F U {(u, v)}
            }

            return {Tau, F};
        }
    }
};
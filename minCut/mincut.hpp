// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
// Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
// 

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
            std::set<Supernode> Tau{};      // the set of supernodes
            std::set<Superedge> F{};        // the set of superedges

            for(auto &v : V){
                auto v_ = Supernode();      // u_       <- new supernode
                v_.insert(v);               // V(u_)    <- {v} //TODO: is insertion or assignment in the set of v_ ? 
                Tau.insert(v_);             // Tau      <- Tau U {v_}
            }

            for(auto &e : E){
                auto Euv = Superedge();     // Euv      <- {(u, v)}
                Euv.insert(e);              // 
                F.insert(Euv);              // F        <- F U {(u, v)}
            }

            return {Tau, F};
        }

        // Source: Algorithm 3: Merge(a, b, Tau) // Tau is the set of supernodes with a, b £ Tau
        Superedge Merge(Supernode a, Supernode b, std::set<Supernode> &Tau, Graph &G){
            // pre

            // implementation
            //auto x = Supernode();
            x.V = a.V;
            x.V.merge(b.V); // V(x) <- V(a) U V(b)

            //// get outs
            auto outFroma = G.out(a);
            auto outFromb = G.out(b);
            Superedge Exd, Ead, Ebd;

            Tau.extract(a); Tau.extract(b);
            //for(auto d : Tau){
                //Ead = outFroma.to(d);
                //Ebd = outFromb.to(d);
                //Exd.Euv.merge(Ead.Euv);
                //Exd.Euv.merge(Ebd.Euv);
            //}

            //Tau.insert(x);
            return Exd;
        }
    }
};
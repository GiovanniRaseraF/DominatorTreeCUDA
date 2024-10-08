// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
// Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
// Help From: https://www.baeldung.com/cs/minimum-cut-graphs
// 

#include <deque>
#include <queue>
#include <list>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>
#include <tuple>
#include "graph.hpp"

namespace sequential{
    // Karger's Algorithm
    namespace Kalger{
        // Source: Algorithm 1: IntuitiveKarger(G)
        std::vector<Edge> IntuitiveKarger(Graph &G){

        }

        // Source: Algorithm 2: Initialize(G)
        std::tuple<std::vector<Supernode>, std::vector<Superedge>> Initialize(Graph &G){ // G = (V, E)
            // pre
            auto &V = G.V;
            auto &E = G.E;

            std::vector<Supernode> Tau;
            std::vector<Superedge> F;

            for(auto &v : V){
                auto v_ = Supernode();
                v_.insert(v);
                Tau.push_back(v_);
            }

            for(auto &uv : E){
                auto Euv = Superedge();
                Euv.insert(uv);
                F.push_back(Euv);
            }

            return {Tau, F};
        }

        // Source: Algorithm 3: Merge(a, b, Tau) // Tau is the set of supernodes with a, b £ Tau
        void Merge(Supernode a, Supernode b, std::vector<Supernode> &Tau, Graph &G){
            // pre

            // implementation
            auto x = Supernode();
            x.V = a.V;
            x.insert(b); // V(x) <- V(a) U V(b)
            Tau.push_back(x);
            
        }                

        // Source: Algorithm 2: Initialize(G)
        //std::tuple<std::unordered_set<Supernode>, std::unordered_set<Superedge>> Initialize(Graph &G){ // G = (V, E)
            //// pre
            //auto &V = G.V;
            //auto &E = G.E;
            
            //// implementation
            //std::cout << "sequential::Kalger::Initialize(G)" << std::endl;
            //std::unordered_set<Supernode> Tau{};      // the set of supernodes
            //std::unordered_set<Superedge> F{};        // the set of superedges

            //for(auto &v : V){
                //auto v_ = Supernode();      // u_       <- new supernode
                //v_.insert(v);               // V(u_)    <- {v} //TODO: is insertion or assignment in the set of v_ ? 
                //Tau.insert(v_);             // Tau      <- Tau U {v_}
            //}

            //for(auto &e : E){
                //auto Euv = Superedge();     // Euv      <- {(u, v)}
                //Euv.insert(e);              // 
                //F.insert(Euv);              // F        <- F U {(u, v)}
            //}

            //return {Tau, F};
        //}

        // Source: Algorithm 3: Merge(a, b, Tau) // Tau is the set of supernodes with a, b £ Tau
        //Superedge Merge(Supernode a, Supernode b, std::unordered_set<Supernode> &Tau, Graph &G){
            //// pre

            //// implementation
            //auto x = Supernode();
            //x.V = a.V;
            //x.V.merge(b.V); // V(x) <- V(a) U V(b)

            ////// get outs
            //auto outFroma = G.out(a);
            //auto outFromb = G.out(b);
            //Superedge Exd, Ead, Ebd;

            //Tau.extract(a); Tau.extract(b);
            //for(auto d : Tau){
                //Ead = outFroma.to(d);
                //Ebd = outFromb.to(d);
                //Exd.Euv.merge(Ead.Euv);
                //Exd.Euv.merge(Ebd.Euv);
            //}

            //Tau.insert(x);
            //return Exd;
        //}
    }
};
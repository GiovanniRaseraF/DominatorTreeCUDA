// Author: Giovanni Rasera

#include <iostream>
#include "mincut.hpp"

void print(std::vector<Supernode> &Tau, std::vector<Superedge> &F){
for(auto t : Tau){
        for(auto v : t.V){
            std::cout << v.id << ", ";
        }

        std::cout << std::endl;
    }

    for(auto f : F){
        auto Euv = f.Euv;

        for(auto e : Euv){
            std::cout << "(" << e.from.id << ", " << e.to.id << "), ";
        }

        std::cout << std::endl;
    }
}

int main(){
    Graph G;

    // Nodes
    Node s = Node(0);
    Node n1 = Node(1), n2 = Node(2), n3 =  Node(3), n4 = Node(4), n5 = Node(5);
    Node t = Node(6);
    G.insertNode(s);
    G.insertNode(n1);
    G.insertNode(n2);
    G.insertNode(n3);
    G.insertNode(n4);
    G.insertNode(n5);
    G.insertNode(Node(9));
    G.insertNode(t);

    // edges
    G.insertEdge(Edge(s, n1, 1));
    G.insertEdge(Edge(s, n3, 1));
    G.insertEdge(Edge(s, n4, 1));
    G.insertEdge(Edge(s, n5, 1));

    G.insertEdge(Edge(n5, n2, 1));
    G.insertEdge(Edge(n1, n2, 1));
    G.insertEdge(Edge(n3, n2, 1));
    G.insertEdge(Edge(n4, n2, 1));

    G.insertEdge(Edge(n2, t, 1));

    // Initialize
    auto resInitialize = sequential::Kalger::Initialize(G);

    std::vector<Supernode> Tau = std::get<0>(resInitialize);
    auto F = std::get<1>(resInitialize);

    print(Tau, F);

    // algo
    while(Tau.size() > 2){
        auto a = Supernode();
        auto b = Supernode();
        a.insert(Node(0));
        b.insert(Node(1));
        sequential::Kalger::Merge(a, b, Tau, G);

        print(Tau, F);
        std::cin.ignore();
    }
}
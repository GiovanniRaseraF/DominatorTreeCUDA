// Author: Giovanni Rasera

#include <iostream>
#include "mincut.hpp"

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

    auto Tau = std::get<0>(resInitialize);
    auto F = std::get<1>(resInitialize);

    for(auto t : Tau){
        std::cout << t.u.id << std::endl;
    }

    for(auto f : F){
        std::cout << "(" << f.uv.from.id << ", " << f.uv.to.id << ")" << std::endl;
    }
}
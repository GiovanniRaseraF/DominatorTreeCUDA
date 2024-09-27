// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <set>
#include <vector>

// (id)
struct Node{
    int id;

    // TODO: implement compare
    bool operator< (const Node& other) const {
        return (id < other.id);
    }
};

// (from) -capacity-> (to)
struct Edge{
    Node from;
    Node to;
    int capacity;

    // TODO: implement compare
};

// G = (V, E)
struct Graph{
    std::set<Node> V = {};
    std::set<Edge> E = {};
};

// The set of nodes within a supernode u as V (u)
struct Supernode{
    Node u;
    std::set<Node> V;

    // TODO: implement compare
    bool operator< (const Supernode& op2) const {
        return (u < op2.u);
    }
};

// The set of edges between two supernodes u, v as Euv
struct Superedge{
    Edge uv;
    std::set<Edge> Euv;

    // TODO: implement compare
};
// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <set>
#include <vector>

// (id)
struct Node{
    int id;

    Node(int i) : id{i}{}

    // Ordere by id
    bool operator< (const Node& other) const {
        return (id < other.id);
    }
};

// (from) -capacity-> (to)
struct Edge{
    Node from;
    Node to;
    int capacity;

    Edge(Node f, Node t, int c) : from{f}, to{t}, capacity{c}{}

    // Ordered by capacity
    bool operator< (const Edge& other) const {
        return (from < other.from) || (to < other.to);
    }
};

// G = (V, E)
struct Graph{
    std::set<Node> V = {};
    std::set<Edge> E = {};

    void insertNode(Node n){
        V.insert(n);
    }

    void insertEdge(Edge e){
        E.insert(e);
    }
};

// The set of nodes within a supernode u as V (u)
struct Supernode{
    Node u;
    std::set<Node> V;

    Supernode(Node Nu) : u{Nu}{}

    // V(u_) <- {v};
    void insert(Node v){
        V.insert(v);
    }

    // Ordered by Node
    bool operator< (const Supernode& other) const {
        return (u < other.u);
    }
};

// The set of edges between two supernodes u, v as Euv
struct Superedge{
    Edge uv;
    std::set<Edge> Euv;

    Superedge(Edge Nuv) : uv{Nuv}{}

    // Euv <- {(u, v)};
    void insert(Edge Nuv){
        Euv.insert(Nuv);
    }

    // Ordered by Edge
    bool operator< (const Superedge& other) const {
        return (uv < other.uv);
    }
};
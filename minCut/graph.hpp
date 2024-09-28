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

// The set of nodes within a supernode u as V(u)
struct Supernode{
    std::set<Node> V;

    Supernode(){}

    // V(u_) <- {v};
    void insert(Node v){
        V.insert(v);
    }

    bool operator< (const Supernode& other) const {
        bool ret = true;
        
        for(auto itThis = V.begin(), itOther = other.V.begin(); itThis != V.end() && itOther != other.V.begin(); ++itThis, ++itOther){
            if ((*itThis) < (*itOther)){
                return true;
            }
        }

        return ret;
    }
};

// The set of edges between two supernodes u, v as Euv
struct Superedge{
    std::set<Edge> Euv;

    Superedge(){}

    // Euv <- {(u, v)};
    void insert(Edge Nuv){
        Euv.insert(Nuv);
    }

    bool operator< (const Superedge& other) const {
        bool ret = true;
        
        for(auto itThis = Euv.begin(), itOther = other.Euv.begin(); itThis != Euv.end() && itOther != other.Euv.begin(); ++itThis, ++itOther){
            if ((*itThis) < (*itOther)){
                return true;
            }
        }

        return ret;
    }
};
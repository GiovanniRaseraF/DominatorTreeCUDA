// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <vector>

// (id)
struct Node{
    int id;

    Node(int i) : id{i}{}
};

// (from) -capacity-> (to)
struct Edge{
    Node from;
    Node to;
    int capacity;

    Edge(Node f, Node t, int c) : from{f}, to{t}, capacity{c}{}
};

// The std::vector of nodes within a supernode u as V(u)
struct Supernode{
    std::vector<Node> V;

    Supernode(){}

    // V(u_) <- {v};
    void insert(Node v){
        V.push_back(v);
    }

    void insert(const Supernode &sn){
        for(auto v : sn.V){
            insert(v);
        }
    }
};

// The std::vector of edges between two supernodes u, v as Euv
struct Superedge{
    std::vector<Edge> Euv;

    Superedge(){}

    // Euv <- {(u, v)};
    void insert(Edge Nuv){
        Euv.push_back(Nuv);
    }

    // to node
    Superedge to(Node v){
        std::vector<Edge> ret;
        std::copy_if(Euv.begin(), Euv.end(), std::inserter(ret, std::next(ret.begin())), 
            [&](Edge e){ return e.to.id == v.id;});
        
        Superedge se;
        se.Euv = ret;
        return se;
    }
};

// G = (V, E)
struct Graph{
    std::vector<Node> V = {};
    std::vector<Edge> E = {};

    void insertNode(Node n){
        V.push_back(n);
    }

    void insertEdge(Edge e){
        E.push_back(e);
    }

    // out edges 
    Superedge out(Node n){
        std::vector<Edge> ret;
        std::copy_if(E.begin(), E.end(), std::inserter(ret, std::next(ret.begin())), 
            [&](Edge e){ return e.from.id == n.id;});

        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge out(Supernode sn){
        std::vector<Edge> ret;

        for(auto v : sn.V){
            auto outv = out(v);
            for(auto e : outv.Euv){
                ret.push_back(e);
            }
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    // int edges 
    Superedge in(Node n){
        std::vector<Edge> ret;
        std::copy_if(E.begin(), E.end(), std::inserter(ret, std::next(ret.begin())), 
            [&](Edge e){ return e.to.id == n.id;});

        Superedge se;
        se.Euv = ret;
        return se;
    }

    // all edges
    Superedge all(Node n){
        auto retOut = out(n);
        auto retIn = in(n);

        Superedge se;
        se.Euv = retOut.Euv;
        return se;
    }
};


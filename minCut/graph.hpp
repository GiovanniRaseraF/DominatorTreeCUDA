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
        
        for(auto itThis = V.begin(), itOther = other.V.begin(); 
                itThis != V.end() && itOther != other.V.begin(); 
                ++itThis, ++itOther){
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

    // to node
    Superedge to(Node v){
        std::set<Edge> ret;
        std::copy_if(Euv.begin(), Euv.end(), std::inserter(ret, std::next(ret.begin())), 
            [&](Edge e){ return e.to.id == v.id;});
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge to(Supernode v){
        std::set<Edge> ret;

        for(auto v : v.V){
            auto outv = to(v);
            ret.merge(outv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    bool operator< (const Superedge& other) const {
        bool ret = true;
        
        for(auto itThis = Euv.begin(), itOther = other.Euv.begin(); 
                itThis != Euv.end() && itOther != other.Euv.begin(); 
                ++itThis, ++itOther){
            if ((*itThis) < (*itOther)){
                return true;
            }
        }

        return ret;
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

    // out edges 
    Superedge out(Node n){
        std::set<Edge> ret;
        std::copy_if(E.begin(), E.end(), std::inserter(ret, std::next(ret.begin())), 
            [&](Edge e){ return e.from.id == n.id;});

        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge out(Supernode n){
        std::set<Edge> ret;

        for(auto v : n.V){
            auto outv = out(v);
            ret.merge(outv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge in(Supernode n){
        std::set<Edge> ret;

        for(auto v : n.V){
            auto inv = in(v);
            ret.merge(inv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    // int edges 
    Superedge in(Node n){
        std::set<Edge> ret;
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
        retOut.Euv.merge(retIn.Euv);

        Superedge se;
        se.Euv = retOut.Euv;
        return se;
    }
};


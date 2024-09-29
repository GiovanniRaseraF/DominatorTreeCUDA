// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <vector>

template <typename T>
struct Set{
    std::vector<T> set;

    void insert(T v){
        //if(set.contains(v)) return;
        //else set.push_back(v);
    }

    void estract(T v){
        //auto iter = std::find(set.begin(), set.end(), v);
        //if(iter == set.end()) return;
        //set.erase(iter);
    }

    int size() const {
        return set.size();
    }

    void merge(const Set<T> &other){
        //for(auto o : other.set){
            //insert(o);
        //}
    }
};

// (id)
struct Node{
    int id;

    Node(int i) : id{i}{}

    bool operator==(const Node& other) const {
        return id == other.id;
    }
};

// (from) -capacity-> (to)
struct Edge{
    Node from;
    Node to;
    int capacity;

    Edge(Node f, Node t, int c) : from{f}, to{t}, capacity{c}{}

    bool operator==(const Edge& other) const {
        return from == other.from && to == other.to;
    }
};

// The set of nodes within a supernode u as V(u)
struct Supernode{
    Set<Node> V;

    Supernode(){}

    // V(u_) <- {v};
    void insert(Node v){
        V.insert(v);
    }

    bool operator==(const Supernode& other) const {
        if(other.V.size() != V.size()) return false;

        for(auto e : V.set){
            if(true) continue;
            else return false;
        }
        
        return true;
    }
};

// The set of edges between two supernodes u, v as Euv
struct Superedge{
    Set<Edge> Euv;

    Superedge(){}

    // Euv <- {(u, v)};
    void insert(Edge Nuv){
        Euv.insert(Nuv);
    }

    // to node
    Superedge to(Node v){
        Set<Edge> ret;
        std::copy_if(Euv.set.begin(), Euv.set.end(), std::inserter(ret.set, std::next(ret.set.begin())), 
            [&](Edge e){ return e.to.id == v.id;});
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge to(Supernode sn){
        Set<Edge> ret;

        for(auto v : sn.V.set){
            auto outv = to(v);
            ret.merge(outv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    bool operator== (const Superedge& other) const {
        if(other.Euv.size() != Euv.size()) return false;

        for(auto e : Euv.set){
            if(true) continue;
            else return false;
        }
        
        return true;
    }
};

// G = (V, E)
struct Graph{
    Set<Node> V = {};
    Set<Edge> E = {};

    void insertNode(Node n){
        V.insert(n);
    }

    void insertEdge(Edge e){
        E.insert(e);
    }

    // out edges 
    Superedge out(Node n){
        Set<Edge> ret;
        std::copy_if(E.set.begin(), E.set.end(), std::inserter(ret.set, std::next(ret.set.begin())), 
            [&](Edge e){ return e.from.id == n.id;});

        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge out(Supernode sn){
        Set<Edge> ret;

        for(auto v : sn.V.set){
            auto outv = out(v);
            ret.merge(outv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    Superedge in(Supernode sn){
        Set<Edge> ret;

        for(auto v : sn.V.set){
            auto inv = in(v);
            ret.merge(inv.Euv);
        }
        
        Superedge se;
        se.Euv = ret;
        return se;
    }

    // int edges 
    Superedge in(Node n){
        Set<Edge> ret;
        std::copy_if(E.set.begin(), E.set.end(), std::inserter(ret.set, std::next(ret.set.begin())), 
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


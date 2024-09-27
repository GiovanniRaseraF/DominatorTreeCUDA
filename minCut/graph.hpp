// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf

#include <vector>

struct Node{ // (id)
    int id;
};

struct Edge{ // (from) ----capacity----> (to)
    Node from;
    Node to;
    int capacity;
};

class Graph{ // G = (V, E)
    std::vector<Node> V = {};
    std::vector<Edge> E = {};
};
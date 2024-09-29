#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unordered_set>

struct Node{
    int id = 0;
    Node(int i) : id{i}{}

    bool operator==(const Node &other) const{
        return id == other.id;
    }

    
};
struct NodeHash{
    size_t operator()(const Node &n) const{
        return n.id;
    }
};

int main(){
    std::unordered_set<int> tau;
    tau.insert(1);
    tau.insert(2);
    tau.insert(3);
    tau.insert(4);

    for(auto t : tau){
        std::cout << t << " ";
    }
    std::cout << std::endl;


    std::unordered_set<Node, NodeHash> nodes;
    nodes.insert((Node(0)));
    nodes.insert((Node(2)));
    nodes.insert((Node(2)));

    for(auto n : nodes){
        std::cout << "Node("<< n.id<<") ";
    }
}
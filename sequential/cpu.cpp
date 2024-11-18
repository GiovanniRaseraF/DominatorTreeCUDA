// Author: Giovanni Rasera

#include <iostream>
#include "mincut.hpp"
#include "tests.hpp"

int main(int argc, char **argv){
    std::string filename;
    int source = 1;
    int sink = 1;

    if(argc < 3){
        std::cout << "./" << argv[0] << " ./path_to_graph/graph.txt source sink" << std::endl;
        return EXIT_FAILURE;
    }

    filename = argv[1];

    try{
        source = atoi(argv[2]);
        sink = atoi(argv[3]);
    }catch(...){
        std::cout << "./" << argv[0] << " ./path_to_graph/graph.txt source sink" << std::endl;
        return EXIT_FAILURE;
    }

    testFile(filename, source, sink);
}
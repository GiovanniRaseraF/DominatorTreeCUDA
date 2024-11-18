// Author: Giovanni Rasera

#include <iostream>
#include <string>
#include <sstream>

int main(){
    std::string first = "";
    std::getline(std::cin, first);
    std::cout << first << std::endl;

    std::string line;

    while (std::getline(std::cin, line)) {
        std::stringstream ss{line};

        int u, v, c;
        ss >> u >> v >> c;
        std::cout << u  << " " << v  << " " << 1 << std::endl;
    }
}
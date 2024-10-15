// Author: Giovanni Rasera

#include <iostream>
#include "mincut.hpp"
#include "tests.hpp"

int main(){
    std::cout << "Sequential" << std::endl;
    std::cout << "-----Tests usign G-----" << std::endl;
    test1();
    test2();
    test3();

    std::cout << "\n\n-----Tests usign G'-----" << std::endl;
    test4();
    test5();
    test6();
}
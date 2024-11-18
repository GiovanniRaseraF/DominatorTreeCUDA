#include <thread>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <functional>
#include <execution>
#include <iostream>
#include <random>

/*
template< class ExecutionPolicy, class ForwardIt, class UnaryPredicate >
bool none_of( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
              UnaryPredicate p );
*/



int main(){
    std::cout << "TODO: naive solution" << std::endl;

    std::vector<bool> vec(1000000, false);
    std::cout << "vec.size(): " << vec.size() << std::endl;

    // none
    auto ret = 
    std::none_of(
        vec.begin(), vec.end(), 
        [&](bool val){ 
            return val == true;
        }
    );
    // // // 

    std::cout << "result: " << std::boolalpha << ret << std::endl;

    return 0;
}
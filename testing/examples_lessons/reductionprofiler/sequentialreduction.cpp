#include <iostream>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>

#define threads 512
#define blocks 1024
#define N threads * blocks

int main(){
    std::array<int, N> ret;

    for(int i = 0; i < N; i++){
        ret[i] = 1;
    }

    auto sum = std::accumulate(ret.begin(), ret.end(), 0);
    std::cout << "Sum: " << sum;
}
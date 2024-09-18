// Author: Giovanni Rasera

#include <iostream>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>
using namespace std::chrono_literals;

#define N threads * blocks * 2
std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
std::chrono::time_point<std::chrono::steady_clock> end   = std::chrono::steady_clock::now();

int main(){
    std::array<int, N> ret;
    int sum = 0;
    
    // make threads=1024 blocks=512 OPTIM=O3 sequentialreduction
    for(int i = 0; i < N; i++) ret[i] = rand();
    start = std::chrono::steady_clock::now();
    //auto sum = std::accumulate(ret.begin(), ret.end(), 0);
    for(int i = 0; i < N; i++) sum += ret[i];
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " nanos C style for loop took \n";

    for(int i = 0; i < N; i++) ret[i] = rand();
    start = std::chrono::steady_clock::now();
    sum = std::accumulate(ret.begin(), ret.end(), 0);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " nanos accumulate loop took \n";
}
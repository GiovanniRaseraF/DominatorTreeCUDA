#include <iostream>
#include <memory>
#include <thread>

int main(){
    std::cout << "cuda compile" << std::endl;

    int *a;
    cudaMalloc(&a, 10);
    cudaFree(a);

    std::cout << "result: " << "passed" << std::endl;

    return 0;
}
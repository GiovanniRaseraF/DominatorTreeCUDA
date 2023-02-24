#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(){
    std::cout << "cuda compile" << std::endl;

    std::shared_ptr<std::vector<int>> vet;
    vet = std::make_shared<std::vector<int>>();
    vet->push_back(10);

    int *a;
    cudaMalloc(&a, vet->at(0));
    cudaFree(a);

    std::cout << "result: " << "passed" << std::endl;

    return 0;
}
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

const int N = 1;

__global__ void kernel(unsigned long thid){
    printf("th calling: %lu\n", thid);
}

void launch_kernel(){
    float *data;
    cudaMalloc(&data, N * sizeof(float));

    std::thread::id id = std::this_thread::get_id();
    std::cout << id << std::endl;
    std::stringstream ss{""};

    ss << id;
    unsigned long kid = 0;
    ss >> kid;
    
    kernel<<<1, 1>>>(kid);

    cudaDeviceSynchronize();
}

int main(){
    const int numth = 24;
    std::vector<std::thread> thvet;

    for(int i = 0; i < numth; i++){
        thvet.push_back(std::thread{launch_kernel});
    }

    for(auto &t : thvet){
        t.join();
    }
}
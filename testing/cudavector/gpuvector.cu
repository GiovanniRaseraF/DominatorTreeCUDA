#include <iostream>
#include <iomanip>
#include "gvector.cuh"
template <typename T = int>
__global__ void hello(const T *a, T *c, int len){
    unsigned int tid = threadIdx.x;
    if(tid < len)
        c[tid] = a[tid] + tid;
}

int main(){
    std::cout << "CUDA: cuda::vector<T>" << std::endl;

    int N = 100;

    cuda::vector<int> v{N, 1};
    cuda::vector<int> c{N, 0};

    v.mirror(cudaMemcpyHostToDevice);
    c.mirror(cudaMemcpyHostToDevice);

    hello<int><<<1, 100>>>(v.get(), c.get(), 100);

    c.mirror(cudaMemcpyDeviceToHost);

    // print
    for(int i = 0; i < c.size(); i++){
        std::cout << c[i] << std::endl;
    }

    std::cout << "end" << std::endl;
    return 0;
}
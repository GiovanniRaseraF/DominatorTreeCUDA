#include <iostream>
#include <stdio.h>
#include <assert.h>

// Author: Giovanni Rasera

#define N 10000
#define NumThPerBlock 256
#define NumBlocks 256

__global__ void vector_sum(long long *A, long long *B, long long *C){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) C[thread_id] = A[thread_id] + B[thread_id];
}

int main(){
    long long a[N], b[N], c[N];
    long long *dev_a, *dev_b, *dev_c;

    // static memory allocation
    cudaMalloc((void**)&dev_a, N*sizeof(long long));
    cudaMalloc((void**)&dev_b, N*sizeof(long long));
    cudaMalloc((void**)&dev_c, N*sizeof(long long));

    // host inits values
    for(long long i = 0; i < N; i++){
        a[i] = i+1;
        b[i] = i+1;
    }

    // copy memory to gpu
    cudaMemcpy(dev_a, a, N * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(long long), cudaMemcpyHostToDevice);

    // run code on gpu
    vector_sum<<<NumBlocks, NumThPerBlock>>>(dev_a, dev_b, dev_c);

    // read result from gpu to cpu
    cudaMemcpy(c, dev_c, N * sizeof(long long), cudaMemcpyDeviceToHost);

    // results
    std::cout << "[";
    for(int i = 0; i < N-1; i++){
        if(i % 1000 == 0) std::cout << c[i] << ", " << std::endl;
    }
    std::cout << c[N-1];
    std::cout << " ]" << std::endl;;

}
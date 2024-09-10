#include <iostream>
#include <stdio.h>

// Author: Giovanni Rasera

#define N 100
#define NumThPerBlock 256
#define NumBlocks 1

__global__ void vector_sum(int *A, int *B, int *C){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) C[thread_id] = A[thread_id] + B[thread_id];
}

int main(){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // static memory allocation
    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    // host inits values
    for(int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i * i;
    }

    // copy memory to gpu
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // run code on gpu
    vector_sum<<<NumBlocks, NumThPerBlock>>>(dev_a, dev_b, dev_c);

    // read result from gpu to cpu
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // results
    std::cout << "[";
    for(int i = 0; i < N-1; i++){
        std::cout << c[i] << ", ";
    }
    std::cout << c[N-1];
    std::cout << "]";
}
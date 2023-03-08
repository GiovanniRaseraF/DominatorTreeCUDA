#include "book.h"
#include <iostream>
#include <iomanip>

constexpr int N = 100;
constexpr int NumThPerBlock = 256;
constexpr int NumBlocks = 1;

__global__ void vetadd(float *A, float *B, float *C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // need to check if i is bigger than the vector length
    if (i < N)
        // this divergence is good because we have 1 block
        C[i] = A[i]+ B[i];
}

int main(){
#ifdef MAPPED_ZERO_COPY
    std::cout << "CUDA: add PAGE LOCKED + MAPPED ZERO COPY" << std::endl;
#else
    std::cout << "CUDA: add PAGE LOCKED" << std::endl;
#endif

    //on host
    float *h_A, *h_B, *h_C;
    float *A, *B, *C;

#ifdef MAPPED_ZERO_COPY
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&h_A, N * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_B, N * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_C, N * sizeof(float), cudaHostAllocMapped);
#else
    cudaHostAlloc(&h_A, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_B, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_C, N * sizeof(float), cudaHostAllocDefault);
#endif

    for(int i = 0; i < N; i++){
        h_A[i] = i;
        h_B[i] = i*i;
    }

    cudaHostGetDevicePointer(&A, h_A, 0);
    cudaHostGetDevicePointer(&B, h_B, 0);
    cudaHostGetDevicePointer(&C, h_C, 0);

    // create blocks of threads

    // execute kernel 
    vetadd<<<NumBlocks, NumThPerBlock>>>(A, B, C);

    for(int i = 0; i < N; i++){
        //std::cout << std::setw(4) << std::left << h_A[i] << " + " << std::setw(5) << std::right << h_B[i] << ": " << h_C[i] << std::endl;
    }

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
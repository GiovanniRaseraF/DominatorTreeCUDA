#include "book.h"
#include <iostream>
#include <iomanip>
#include <vector>

template<int a, int b>
struct imin{
    static const int value = (a < b ? a : b);
};

constexpr int N = 128*254;
constexpr int threadsPerBlock = 254;
constexpr int blocksPerGrid = imin< 32, (N+threadsPerBlock-1) / threadsPerBlock >::value;

__global__ void dot(float *A, float *B, float *C){
    // create shared mamory between thread of the same block
    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;

    while(tid < N){
        temp += A[tid] * B[tid];

        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    C[cacheIndex] = temp;
}

float dot_serialized(float *A, float *B, float *C){
    float sum = 0;

    for(int i = 0; i < N; i++){
        C[i] = A[i] * B[i];
        sum += C[i];
    }

    return sum;
}

int main(){
    std::cout << "CUDA: dot product" << std::endl;

    float h_A[N], h_B[N], h_C[threadsPerBlock];
    float *A, *B, *C;

    for(int i = 0; i < N; i++){
        h_A[i] = 2; h_B[i] = 25;
    }
    
    // execution
    HANDLE_ERROR( cudaMalloc((void**)&A, N*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&B, N*sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&C, threadsPerBlock*sizeof(float)) );

    // copy
    HANDLE_ERROR( cudaMemcpy(A, h_A, N*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(B, h_B, N*sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(C, h_C, threadsPerBlock*sizeof(float), cudaMemcpyHostToDevice) );

    // execute kernel 
    dot<<<128, threadsPerBlock>>>(A, B, C);

    // print from gpu
    cudaDeviceSynchronize();

    // retreve result   
    HANDLE_ERROR( cudaMemcpy(h_C, C, threadsPerBlock*sizeof(float), cudaMemcpyDeviceToHost) );

    // print
    for(int i = 0; i < threadsPerBlock; i++){
        std::cout << h_C[i] << std::endl;
    }
    //cuda

    cudaFree(C);
    cudaFree(A);
    cudaFree(B);

    return 0;
}
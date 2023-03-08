#include "book.h"
#include <iostream>
#include <iomanip>
#include <vector>

template<int a, int b>
struct imin{
    static const int value = (a < b ? a : b);
};

constexpr int N = 512 * 1024;
constexpr int threadsPerBlock = 512;
constexpr int blocksPerGrid = imin<32, (N + threadsPerBlock - 1) / threadsPerBlock >::value;

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

    // here threades calculated their part of the multiplication and sums
    cache[cacheIndex] = temp;

    // we have to syncronize them before procedent to create all sums
    __syncthreads();

    // at this point we have to sum blockdim.x value
    // we can do a naive implementation like this:
#ifdef  NAIVE_SUM_DOT
    int sum = 0;

    for(int i = 0; i < threadsPerBlock; i++){
        sum += cache[i];
    }
    __syncthreads();

    C[blockIdx.x] = sum;
#else
    // this called reductions, threadsPerBlock must be a power of 2 
    // this allow to do log2(blockdim.x) operations instead of blockDim.x 
    // of the naive implementation
    // the optimization is here
    int i = blockDim.x / 2;
    while(i != 0){
        if(cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        C[blockIdx.x] = cache[0];
#endif
}

float dot_serialized(float *A, float *B, float *C){
    float sum = 0;

    for(int i = 0; i < N; i++){
        sum += A[i] * B[i];
    }

    return sum;
}

int main(){
#ifdef  NAIVE_SUM_DOT
    std::cout << "CUDA: dot product: with NAIVE sum" << std::endl;
#else
    std::cout << "CUDA: dot product" << std::endl;
#endif
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
    dot<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);
    // retreve result   
    HANDLE_ERROR( cudaMemcpy(h_C, C, threadsPerBlock*sizeof(float), cudaMemcpyDeviceToHost) );
    // finish sum on the cpu
    int d = 0;
    for(int i = 0; i < blocksPerGrid; i++) d += h_C[i];

    // print
    std::cout << std::setw(15) << std::left << "parallel dot: " << d << std::endl;
    std::cout << std::setw(15) << std::left << "serial dot: " << (int)dot_serialized(h_A, h_B, h_C) << std::endl;
    //}
    //cuda

    cudaFree(C);
    cudaFree(A);
    cudaFree(B);

    return 0;
}
#include "book.h"
#include <iostream>
#include <iomanip>

constexpr int NumThPerBlock = 256;
constexpr int NumBlocks = 256;
constexpr int N = NumBlocks * NumThPerBlock;

__global__ void vetadd(float *A, float *B, float *C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // need to check if i is bigger than the vector length
    if (i < N)
        // this divergence is good because we have 1 block
        C[i] = A[i]+ B[i];
}

int main(){
    std::cout << "CUDA: mat add" << std::endl;
    //on host
    float h_A[N], h_B[N], h_C[N];

    // on device
    // device memory cannot be deferenced
    // so we nell cudaMalloc and cudaMemcpy 
    // to know result and allocate memory
    float *A, *B, *C;

    // init
    for(int i = 0; i < N; i++){
        h_A[i] = -i;
        h_B[i] = i*i;
        h_C[i] = 0;
    }
    
    // malloc mem on device
    // the memory is STATIC in the sense that
    // the memory is copied before the kernel 
    // execution
    HANDLE_ERROR(
        cudaMalloc(
            (void**)&A,
            N*sizeof(float))
    );
    HANDLE_ERROR(
        cudaMalloc(
            (void**)&B,
            N*sizeof(float))
    );
    HANDLE_ERROR(
        cudaMalloc(
            (void**)&C,
            N*sizeof(float))
    );

    // copy
    HANDLE_ERROR(
        cudaMemcpy(
            A,
            h_A,
            N*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    HANDLE_ERROR(
        cudaMemcpy(
            B,
            h_B,
            N*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    HANDLE_ERROR(
        cudaMemcpy(
            C,
            h_C,
            N*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    // create blocks of threads

    // execute kernel 
    vetadd<<<NumBlocks, NumThPerBlock>>>(A, B, C);

    // retreve result   
    HANDLE_ERROR(
        cudaMemcpy(
            h_C,
            C,
            N*sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    //for(int i = 0; i < N; i++){
        //std::cout << std::setw(4) << std::left << h_A[i] << " + " << std::setw(5) << std::right << h_B[i] << ": " << h_C[i] << std::endl;
    //}

    cudaFree(C);
    cudaFree(A);
    cudaFree(B);

    return 0;
}
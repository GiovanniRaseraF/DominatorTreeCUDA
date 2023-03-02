#include "book.h"
#include <iostream>
#include <iomanip>

constexpr int N = 32*4;

__global__ void matadd(float *A, float *B, float *C){
    int i = threadIdx.x;
    //int j = threadIdx.y;

    C[i] = A[i]+ B[i];
}

int main(){
    std::cout << "CUDA: mat add" << std::endl;
    //on host
    float h_A[N];
    float h_B[N];
    float h_C[N];

    
    // on device
    float *A, *B, *C;

    // init
    for(int i = 0; i < N; i++){
        h_A[i] = 1;
        h_B[i] = 1;
        h_C[i] = 0;
    }
    
    // malloc mem on device
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
    int numBlocks = 1;
    dim3 threadPerBlock(N);

    // execute kernel 
    matadd<<<numBlocks, threadPerBlock>>>(A, B, C);

    // retreve result   
    HANDLE_ERROR(
        cudaMemcpy(
            h_C,
            C,
            N*sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    for(int i = 0; i < N; i++){
        std::cout << std::setw(3) << h_C[i];
    }

    cudaFree(C);
    cudaFree(A);
    cudaFree(B);

    return 0;
}
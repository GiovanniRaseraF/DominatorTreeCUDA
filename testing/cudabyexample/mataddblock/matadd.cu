#include "book.h" // compile with nvcc -I../common/
#include <iostream>
#include <iomanip>

constexpr int N = 16;

__global__ void matadd(float A[N][N], float B[N][N], float C[N][N]){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main(){
    std::cout << "CUDA: mat add block: " << __FILE__ << std::endl;

    //on host
    float h_A[N][N];
    float h_B[N][N];
    float h_C[N][N];

    float A[N][N];
    float B[N][N];
    float C[N][N];

    // create blocks of threads
    dim3 threadPerBlock(N, N);
    dim3 numBlocks(N / threadPerBlock.x, N / threadPerBlock.y);

    // execute kernel 
    matadd<<<numBlocks, threadPerBlock>>>(A, B, C);

    cudaFree(C);
    cudaFree(A);
    cudaFree(B);

    return 0;
}
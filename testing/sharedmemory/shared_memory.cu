#include <iostream>
#include <iomanip>

constexpr int N = 512;

__global__ void reverse(float *A, int n){
#ifndef DINAMIC_SHARED_MEM
    __shared__ float shm[N];
#else
    extern __shared__ float shm[];
#endif

    int t = threadIdx.x;
    int trv = n - t - 1;

    shm[t] = A[t];

    // need sync to prevent race condition on shared memory on Streaming Multiprocessor
    // all threads in the same block will barrier here
    __syncthreads();

    A[trv] = shm[t];
}

int main(){
#ifndef DINAMIC_SHARED_MEM
    std::cout << "CUDA: Reverse with shared STATIC memory allocation" << std::endl;
#else
    std::cout << "CUDA: Reverse with shared DINAMIC memory allocation" << std::endl;
#endif

    //on host
    float *h_A, *A;
    
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&h_A, N * sizeof(float), cudaHostAllocMapped);
    for(int i = 0; i < N; i++) h_A[i] = i;

    cudaHostGetDevicePointer(&A, h_A, 0);

    // execute kernel 
#ifndef DINAMIC_SHARED_MEM
    reverse<<<1, N>>>(A, N);
#else
    reverse<<<1, N, N * sizeof(float)>>>(A, N);
#endif

    cudaFreeHost(h_A);

    return 0;
}
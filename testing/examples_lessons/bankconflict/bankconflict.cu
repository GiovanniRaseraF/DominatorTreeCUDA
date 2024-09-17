// static memory
#include <iostream>
#include <stdlib.h>
#include <iomanip>

// Author: Giovanni Rasera
// date: 2024 09 17

// comes from outside
const int N = (threads * blocks);

static void HandleError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        std::cerr << (cudaGetErrorString(err)) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void staticReverse(int *d, int n){
    __shared__ int s[threads];
    int tid = threadIdx.x;
    int t = (blockIdx.x * blockDim.x) + tid;
    int tr = n-t-1;
    s[tid] = d[t];
    __syncthreads();
    d[tr] = s[tid];
}

__global__ void dynamicReverse(int *d, int n){
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int t = (blockIdx.x * blockDim.x) + tid;
    int tr = n-t-1;
    s[tid] = d[t];
    __syncthreads();
    d[tr] = s[tid];
}


bool checkResult(int *h_A){
    bool isGood = true;
    for(int i = 1; i < N; i++){
        if((h_A[i-1] < h_A[i])){
            isGood = false;
        }
    }
    return isGood;
}

int main(){
    // host
    int h_A[N];

    // device
    int *A;
    
    // generate data
    for(int i = 0; i < N; i++) h_A[i] = i;
    
    // create memory on device
    cudaMalloc((void**)&A, N*sizeof(int));

    // copy the memory to the device 
    cudaMemcpy(A, h_A, N*(sizeof(int)), cudaMemcpyHostToDevice);
    
    // Compute
#ifdef DYNAMIC
    dynamicReverse<<<blocks, threads, threads>>>(A, N);
#ifdef cudaSync
    cudaDeviceSynchronize();
    std::cout << "\nSynched\n";
#endif
    std::cout << "\nDynamic Allocation\n";
#else 
    staticReverse<<<blocks, threads>>>(A, N);
#ifdef cudaSync
    cudaDeviceSynchronize();
    std::cout << "\nSynched\n";
#endif
    std::cout << "\nStatic Allocation\n";
#endif

    // retreive result
    HANDLE_ERROR(cudaMemcpy(h_A, A, N*(sizeof(int)), cudaMemcpyDeviceToHost));
    auto check = checkResult(h_A);
    std::cout << "Reversed? " << std::boolalpha << check << std::endl;
}


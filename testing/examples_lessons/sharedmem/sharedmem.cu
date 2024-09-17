// static memory
#include <iostream>
#include <stdlib.h>

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
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n-t-1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n){
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int t = (blockIdx.x * blockDim.x) + tid;
    int tr = n-t-1;
    // printf("dix:%d, dm: %d, tid: %d, t: %d, tr: %d\n", blockIdx.x, blockDim.x, tid, t, tr);
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

int main(){
    // host
    int h_A[N];

    // device
    int *A;
    
    // generate data
    for(int i = 0; i < N; i++) h_A[i] = i;
    std::cout << "\n";
    std::cout << "[";
    for(int i = 0; i < N-1; i++){
        if((i % blocks) == 0) std::cout << h_A[i] << ", ";
    }
    std::cout << h_A[N-1];
    std::cout << "]";

    // create memory on device
    cudaMalloc((void**)&A, N*sizeof(int));

    // copy the memory to the device 
    cudaMemcpy(A, h_A, N*(sizeof(int)), cudaMemcpyHostToDevice);
    
    // Compute
    dynamicReverse<<<blocks, threads, N>>>(A, N);
    //cudaDeviceSynchronize();

    // retreive result
    HANDLE_ERROR(cudaMemcpy(h_A, A, N*(sizeof(int)), cudaMemcpyDeviceToHost));

    std::cout << "\n";
    std::cout << "[";
    for(int i = 0; i < N; i++){
        if((i % blocks) == 0) std::cout << h_A[i] << ", ";
    }
    std::cout << h_A[N-1];
    std::cout << "]";
}

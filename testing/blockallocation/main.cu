#include <stdlib.h>

__global__ void mallocTest(){
    __shared__ int *data;

    print("blockDim.x: %d\n", blockDim.x);

    if(threadIdx.x == 0){
        size_t size = blockDim.x * 64;
        data = (int *) malloc(sizeof(int) * size);
    }

    __syncthreads();

    // Check dor failure
    if(data == NULL) return;

    // Thread index into the memory
    int * ptr = data;
    for(int i = 0; i < 64; ++i){
        ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;
    }

    __syncthreads();

    if(threadIdx.x == 0){
        free(data);
    }
}

int main(){
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    mallocTest<<<4, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
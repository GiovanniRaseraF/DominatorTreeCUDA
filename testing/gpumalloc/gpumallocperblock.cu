#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>

__global__ void gpumalloc(){
    size_t size = 64;
    // shared in the streaming multiprocessor
    // this is private in the block
    __shared__ int* ptr_in_block;

    // only one thread allocates memory
    if(threadIdx.x == 0)
        ptr_in_block = (int*)malloc(blockDim.x * sizeof(int));

    // all threads start from here their execution 
    __syncthreads();
    
    // check for failure
    if(ptr_in_block == NULL) return;

    // Actual parallel code from here
    // value in the block
    
    int *ptr = ptr_in_block; // optimization
    // ptr will be in a register
    // ptr_in_block will be a shared variable
    ptr[threadIdx.x] = threadIdx.x * (blockIdx.x + 1);

    printf("th: %d\n", ptr[threadIdx.x]);

    __syncthreads();

    // free all mememory
    if(threadIdx.x == 0) free(ptr_in_block);
}

int main()
{
    std::cout << "CUDA: gpu malloc" << std::endl;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    gpumalloc<<<64, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
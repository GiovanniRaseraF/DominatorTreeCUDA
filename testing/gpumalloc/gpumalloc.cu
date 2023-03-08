#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>

__global__ void gpumalloc(){
    size_t size = 160;

    // in this case every thread allocates memory for
    // himself
    char *ptr = (char*)__nv_aligned_device_malloc(size, 32);
    memset(ptr, 0, size);

    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);

    free(ptr);
}

int main(int argc, char*argv[])
{
    std::cout << "CUDA: gpu malloc" << std::endl;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    gpumalloc<<<1, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}
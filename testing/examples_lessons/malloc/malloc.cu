#include <iostream>
#include <stdlib.h>

__global__ void mallocTest(){
    size_t size = 123;
    char* ptr = (char*) malloc(size);

    memset(ptr, 0, size);

    printf("Thread %d pointing to %p", threadIdx.x, ptr);
    //std::cout << "Thread " << threadIdx.x << " got pointer " << ptr << std::endl;

    free(ptr);
}

int main(){
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}

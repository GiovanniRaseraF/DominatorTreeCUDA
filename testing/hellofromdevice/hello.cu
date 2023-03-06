#include <iostream>
#include <stdio.h>
#include <cuda.h>

__global__ void hellokernel(void){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("I'm: %d\n", tid);
    // cannot use std::cout
}

int main(){
    hellokernel<<<10, 32>>>();

    // to see output
    cudaDeviceSynchronize();

    std::cout << "Hello World !" << std::endl;

    return 0;
}
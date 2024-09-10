#include <iostream>
#include <stdio.h>

// Author: Giovanni Rasera

__global__ void mykernel(void){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int block_index = blockIdx.x;

    printf("Im thread: %d in block %d\n ", thread_id, block_index);
}

int main(){
    mykernel<<<2,32>>>();
    std::cout << ("Hello World!\n");
}
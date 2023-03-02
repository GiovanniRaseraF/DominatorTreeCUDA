#include "book.h"
#include <iostream>

__global__ void add(int a, int b, int *c){
    *c = a + b;
}

int main(){
    std::cout << "CUDA: 3.2.3 Passing Parameters" << std::endl;

    int c = 0;
    int *dev_c;

    // first allocate memory for the result in the GPU(device)
    // do it by using malloc on GPU
    auto h_error = cudaMalloc(
        (void **)&dev_c, 
        sizeof(int)
    );
    // error check
    HANDLE_ERROR(h_error);

    // create the kernel
    // <<< number of blocks, threads per block>>>
    // thread per block: dim3
    add<<<1, 1>>>(2, 99, dev_c);

    // copy the result from GPU to CPU
    h_error = cudaMemcpy(
        &c,
        dev_c,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );
    HANDLE_ERROR(h_error);

    // print results
    std::cout << "host: 2 + 99: " << c << std::endl;

    // free the memory in the GPU
    h_error = cudaFree(dev_c);

    return 0;
}
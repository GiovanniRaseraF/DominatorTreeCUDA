#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <memory>
#include <thread>

int main(){
    std::cout << "cuda compile" << std::endl;

    int *a;
    cudaMalloc(&a, 10);
    cudaFree(a);

    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "thread num: " << omp_get_thread_num() << std::endl;
        }
    }

    std::cout << "result: " << "passed" << std::endl;

    return 0;
}
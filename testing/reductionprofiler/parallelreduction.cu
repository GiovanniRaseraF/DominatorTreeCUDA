#include <iostream>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "gpu_reduce.cuh"

float reduce(float *h_Results, float *A, float *Results);

int main(){
    std::cout << "N: " << N << " -" << reduce_type << " reduce"<< std::endl;

    //on host
    float h_A[N], h_Results[N];
    float *A, *Results;

    for(int i = 0; i < N; i++){
        h_A[i] = 1;
        h_Results[i] = 0;
    }

    cudaMalloc(
        (void**)&A,
        N*sizeof(float));
    cudaMalloc(
        (void**)&Results,
        N*sizeof(float));

    // copy
    cudaMemcpy(
        A,
        h_A,
        N*sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        Results,
        h_Results,
        N*sizeof(float),
        cudaMemcpyHostToDevice
    );

    auto sum = reduce(h_Results, A, Results);

    std::cout << "result= " << sum << ", correct?" << std::boolalpha << " " << (sum == N) << std::endl;

    cudaFree(A);
    cudaFree(Results);

    return 0;
}
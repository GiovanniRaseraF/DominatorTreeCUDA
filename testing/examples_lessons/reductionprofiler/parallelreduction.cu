#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>

#define threads 512
#define blocks 1024
#define N threads * blocks

int factor = 1;
__global__ void reduceGPU(float *A, float *Results){
    __shared__ float shared_mem[threads];

    // indexing
    unsigned int tid = threadIdx.x;
    unsigned int index_in_global = (blockIdx.x * blockDim.x) + tid;
    int sum = 0;

    // load to shared mem
    shared_mem[tid] = A[index_in_global];
    __syncthreads();

    if(tid == 0){
        for(unsigned int s = 0; s < blockDim.x; s++){
            sum += shared_mem[s];
        }

        Results[blockIdx.x] = sum;
    }
    __syncthreads();
}

float reduce(float *h_Results, float *A, float *Results);
float reduce(float *h_Results, float *A, float *Results){
    reduceGPU<<<blocks, threads / factor>>>(A, Results);

    // retreve result   
    cudaMemcpy(
        h_Results,
        Results,
        N*sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::vector<float> res{};
    res.assign(h_Results, h_Results+N);
    auto sum = std::accumulate(res.begin(), res.end(), 0);

    return sum;
}

int main(){
    //std::cout << "N: " << N << " -" << reduce_type << " reduce"<< std::endl;

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
    
    auto sum = reduce(h_Results, A, Results);

    cudaMemcpy(
        Results,
        h_Results,
        N*sizeof(float),
        cudaMemcpyHostToDevice
    );


    std::cout << "result= " << sum << ", correct?" << std::boolalpha << " " << (sum == N) << std::endl;

    cudaFree(A);
    cudaFree(Results);

    return 0;
}
// Author: Giovanni Rasera

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>
#include <chrono>

#define N threads * blocks

#if defined(NAIVE_REDUCE)
int factor = 1;
__global__ void reduceGPU(int *A, int *Results){
    __shared__ int shared_mem[threads];

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

#elif defined(FIRST_ATTEMPT_REDUCE)
int factor = 1;
extern "C"{
    std::string reduce_type{"FIRST_ATTEMPT"};
}

__global__ void reduceGPU(int *A, int *Results){
    __shared__ int shared_mem[threads];

    // indexing
    unsigned int tid = threadIdx.x;
    unsigned int index_in_global = (blockIdx.x * blockDim.x) + tid;

    shared_mem[tid] = A[index_in_global];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if(tid % (s * 2) == 0){
            shared_mem[tid] += shared_mem[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0) Results[blockIdx.x] = shared_mem[0];
}

#elif defined(SECOND_REDUCE)
int factor = 1;
extern "C"{
    std::string reduce_type{"SECOND_REDUCE"};
}

__global__ void reduceGPU(int *A, int *Results){
    __shared__ int shared_mem[threads];

    // indexing
    unsigned int tid = threadIdx.x;
    unsigned int index_in_global = (blockIdx.x * blockDim.x) + tid;

    shared_mem[tid] = A[index_in_global];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        int index = 2 * s * tid;

        if(index < blockDim.x){
            shared_mem[tid] += shared_mem[index + s];
        }

        __syncthreads();
    }

    if(tid == 0) Results[blockIdx.x] = shared_mem[0];
}
#elif defined(THIRD_REDUCE)
int factor = 1;
extern "C"{
    std::string reduce_type{"THIRD_REDUCE"};
}

__global__ void reduceGPU(int *A, int *Results){
    __shared__ int shared_mem[threads];

    // indexing
    unsigned int tid = threadIdx.x;
    unsigned int index_in_global = (blockIdx.x * blockDim.x) + tid;

    shared_mem[tid] = A[index_in_global];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            shared_mem[tid] += shared_mem[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0) Results[blockIdx.x] = shared_mem[0];
}
#elif defined(SPEED_REDUCE)
int factor = 2;

__global__ void reduceGPU(int *A, int *Results){
    __shared__ int shared_mem[threads];

    // indexing
    unsigned int tid = threadIdx.x;
    unsigned int index_in_global = (blockIdx.x * (2 * blockDim.x)) + tid;

    shared_mem[tid] = A[index_in_global] + A[index_in_global + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            shared_mem[tid] += shared_mem[tid + s];
        }

        __syncthreads();
    }

    if(tid == 0) Results[blockIdx.x] = shared_mem[0];
}
#else
extern "C"{
    std::string reduce_type{"NOT_IMPLEMENTED_REDUCE"};
}
__global__ void reduceGPU(int *A, int *Results){
    print("Error on call");
    exit(-1);
}
#endif

int reduce(int *h_Results, int *A, int *Results);
int reduce(int *h_Results, int *A, int *Results){
    reduceGPU<<<dim3(blocks), dim3(threads / factor)>>>(A, Results);

    // retreve result   
    cudaMemcpy(
        h_Results,
        Results,
        N*sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::vector<int> res{};
    res.assign(h_Results, h_Results+N);
    auto sum = std::accumulate(res.begin(), res.end(), 0);

    return sum;
}

int main(){
    //on host
    int h_A[N], h_Results[N];
    int *A, *Results;
    srand(time(NULL));

    for(int i = 0; i < N; i++){
        h_A[i] = (rand() % 100) + 10;
        h_Results[i] = 0;
    }

    cudaMalloc(
        (void**)&A,
        N*sizeof(int));
    cudaMalloc(
        (void**)&Results,
        N*sizeof(int));

    // copy
    cudaMemcpy(
        A,
        h_A,
        N*sizeof(int),
        cudaMemcpyHostToDevice
    );
    
    auto sum = reduce(h_Results, A, Results);

    cudaMemcpy(
        Results,
        h_Results,
        N*sizeof(int),
        cudaMemcpyHostToDevice
    );

    std::vector<int> res{};
    res.assign(h_A, h_A+N);
    auto actual_sum = std::accumulate(res.begin(), res.end(), 0);
    std::cout << std::endl << ": result= " << sum << " | expected= " << actual_sum << ", correct?" << std::boolalpha << " " << (sum == actual_sum) << std::endl;

    cudaFree(A);
    cudaFree(Results);

    return 0;
}
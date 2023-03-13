#pragma once
#include <string>

#define threads 512
#define blocks 1024
#define N threads * blocks

#if defined(NAIVE_REDUCE)
int factor = 1;
extern "C" {
    std::string reduce_type{"NAIVE"};
}

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

#elif defined(FIRST_ATTEMPT_REDUCE)
int factor = 1;
extern "C"{
    std::string reduce_type{"FIRST_ATTEMPT"};
}

__global__ void reduceGPU(float *A, float *Results){
    __shared__ float shared_mem[threads];

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

__global__ void reduceGPU(float *A, float *Results){
    __shared__ float shared_mem[threads];

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

__global__ void reduceGPU(float *A, float *Results){
    __shared__ float shared_mem[threads];

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
extern "C"{
    std::string reduce_type{"SPEED_REDUCE"};
}

__global__ void reduceGPU(float *A, float *Results){
    __shared__ float shared_mem[threads];

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
__global__ void reduceGPU(float *A, float *Results){
    exit(-1);
}
#endif

float reduce(float *h_Results, float *A, float *Results){
    reduceGPU<<<blocks, threads / factor>>>(A, Results);

    // retreve result   
    cudaMemcpy(
        h_Results,
        Results,
        N*sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::vector<float> res;
    res.assign(h_Results, h_Results+N);
    auto sum = std::accumulate(res.begin(), res.end(), 0);

    return sum;
}
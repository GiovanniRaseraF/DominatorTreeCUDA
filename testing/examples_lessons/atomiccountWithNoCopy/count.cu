// Author: Giovanni Rasera

#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>

#define MAX_TH = threads

#define NPerStream ((threads)*(blocks))
#define N ((threads)*(blocks) * scalar)

std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
std::chrono::time_point<std::chrono::steady_clock> end   = std::chrono::steady_clock::now();

__global__ void countNum(int *vetPart, int numToCount, int *res){
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int val = vetPart[tid];

    if (val == numToCount) atomicAdd(res, 1);
}

int main(){
     //on host
    int *h_A = (int *)malloc(N * sizeof(int));
    int *A;
    int *res = (int*) malloc(sizeof(int)), *resNoCopy = (int*) malloc(sizeof(int));
    int *dev_Res, *dev_ResNoCopy;
    *res = 0;
    *resNoCopy = 0;
    int toCount = 0;
    srand(0);

    // start copy
    for(int i = 0; i < N; i++){
        h_A[i] = (rand() % 1000);
    }

    // Cuda No Copy 
    cudaMalloc((void**)&A, (N)*sizeof(int));
    cudaMalloc((void**)&dev_ResNoCopy, sizeof(int));
    cudaMemcpy(A, h_A, (N)*sizeof(int), cudaMemcpyHostToDevice);
    start = std::chrono::steady_clock::now();
    for(int streamIndex = 0; streamIndex < scalar; streamIndex++){
        //count
        countNum<<<blocks, threads>>>(A+(streamIndex * NPerStream), toCount, dev_ResNoCopy);
    }
    cudaMemcpy(resNoCopy, dev_ResNoCopy, sizeof(int), cudaMemcpyDeviceToHost);
    end = std::chrono::steady_clock::now();
    auto countCudaNoCopy = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaFree(A);

    // Cuda With Copy 
    start = std::chrono::steady_clock::now();
    cudaMalloc((void**)&A, (N)*sizeof(int));
    cudaMalloc((void**)&dev_Res, sizeof(int));
    cudaMemcpy(A, h_A, (N)*sizeof(int), cudaMemcpyHostToDevice);

    for(int streamIndex = 0; streamIndex < scalar; streamIndex++){
        //count
        countNum<<<blocks, threads>>>(A+(streamIndex * NPerStream), toCount, dev_Res);
    }
    cudaMemcpy(res, dev_Res, sizeof(int), cudaMemcpyDeviceToHost);
    end = std::chrono::steady_clock::now();
    auto countCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaFree(A);

    // normal calculation
    std::vector<int> values{};
    values.assign(h_A, h_A+N);

    start = std::chrono::steady_clock::now();
    auto counts = std::count(values.begin(), values.end(), toCount);
    end = std::chrono::steady_clock::now();
    auto countStd = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // print
    std::cout << *resNoCopy << ", " << *res << ", " << counts << ", " << N << ", " << NPerStream << ", " << threads << ", " << blocks << ", " << scalar << "," << countCudaNoCopy << ", " << countCuda << ", " << countStd << "\n";

}
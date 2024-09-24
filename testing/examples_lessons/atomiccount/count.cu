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
    int *res = (int*) malloc(sizeof(int));
    int *dev_Res;
    *res = 0;

    int toCount = 0;

    srand(0);

    for(int i = 0; i < N; i++){
        h_A[i] = (rand() % 1000);
    }

    cudaMalloc((void**)&A, (NPerStream)*sizeof(int));
    cudaMalloc((void**)&dev_Res, sizeof(int));

    start = std::chrono::steady_clock::now();
    for(int streamIndex = 0; streamIndex < scalar; streamIndex++){
        // copy
        cudaMemcpy(A, h_A+(streamIndex * NPerStream), (NPerStream)*sizeof(int), cudaMemcpyHostToDevice);

        //count
        countNum<<<blocks, threads>>>(A, toCount, dev_Res);

        // get result
        cudaMemcpy(res, dev_Res, sizeof(int), cudaMemcpyDeviceToHost);
    }
    end = std::chrono::steady_clock::now();
    
    // print
    //std::cout << std::endl << "" << toCount << "" << (*res) << " took ";
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    auto countCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    // normal calculation
    std::vector<int> values{};
    values.assign(h_A, h_A+N);

    start = std::chrono::steady_clock::now();
    auto counts = std::count(values.begin(), values.end(), toCount);
    end = std::chrono::steady_clock::now();

    //std::cout << "StdCount(  " << toCount << " ): " << (counts) << " took ";
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    auto countStd = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //std::cout << "Difference (countCuda - countStd): " << (countCuda - countStd) << std::endl;
    //std::cout << "Ratio (countCuda / countStd): cuda is " << (((float)countStd / (float)countCuda)) << " times better" << std::endl;

    std::cout << N << ", " << NPerStream << ", " << threads << ", " << blocks << ", " << countCuda << ", " << countStd << "\n";

    cudaFree(A);
}
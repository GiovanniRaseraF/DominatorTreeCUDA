#include <iostream>
#include <iomanip>

#define N 100
__global__ void vectorSum(float A[N], float B[N], float C[N]){
    int index = threadIdx.x;
    if(index < N)
        C[index] = A[index] + B[index];
}

int main(){ 
    std::cout << "Hello From main Program" << std::endl;
}
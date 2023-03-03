#include <iostream>
#include <omp.h>
#include <cstring>

constexpr size_t N = 32;

// c implementation
void add(int *A, int *B, int *C, size_t size){
    // this is 1 cpu
    // no optimization
    for(size_t i = 0; i < size; i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    std::cout << __FILE__ << ": " << "1 core cpu add" << std::endl;

    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];

    // all zeros to be shore
    std::memset(A, 0, N*sizeof(int));
    std::memset(B, 0, N*sizeof(int));
    std::memset(C, 0, N*sizeof(int));

    for(size_t i = 0; i < N; i++){
        A[i] = i;
        B[i] = i * 2;
    }

    add(A, B, C, N);

    for(size_t i = 0; i < N; i++){
        std::cout << A[i] << " + " << B[i] << ": " << C[i] << std::endl;
    }

    return 0;
}
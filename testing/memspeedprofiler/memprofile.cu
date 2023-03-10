#include <iostream>
#include <iomanip>

int main(){
    std::cout << "N: " << N << " float allocation"<< std::endl;
    //on host
    float h_A[N];
    float *A, *B, *C;

    for(int i = 2; i < 1024 * 512 - N; i*=2){
        cudaMalloc(
            (void**)&A,
            N*sizeof(float));

        // copy
        cudaMemcpy(
            A,
            h_A,
            N*sizeof(float),
            cudaMemcpyHostToDevice
        );

        // retreve result   
        cudaMemcpy(
            h_A,
            A,
            N*sizeof(float),
            cudaMemcpyDeviceToHost
        );

        cudaFree(A);
    }

    return 0;
}
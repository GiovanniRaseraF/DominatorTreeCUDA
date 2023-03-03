#include "book.h"
#include <iostream>
#include <memory>
#include <bitset>
#include <climits>
#include <cstring>

int main(){
    std::cout << __FILE__ << ": select gpu" << std::endl;

    cudaDeviceProp prop;
    int device;

    // device number, in my case 0 because i have 1 gpu
    HANDLE_ERROR(
        cudaGetDevice(&device)
    );
    std::cout << "ID of current CUDA device: " << device << std::endl;

    // reset prop variable memory with all zeros
    std::memset(&prop, 0, sizeof(cudaDeviceProp));

    // set CUDA version
    prop.major = 5;
    prop.minor = 7;

    HANDLE_ERROR(
        cudaChooseDevice(
            &device,
            &prop
        )
    );
    HANDLE_ERROR(
        cudaGetDeviceProperties(
            &prop, 
            device)
    );

    // print results
    std::cout << "CUDA device ID better for revision 5.7: GPU" << device << std::endl;
    std::cout << " - Name: " << prop.name << std::endl;

    return 0;
}
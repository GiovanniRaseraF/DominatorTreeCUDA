#include "book.h"
#include <iostream>

__global__ void kernel(void){

}

int main(){
    kernel<<<1, 1>>>();

    std::cout << "CUDA: Hello Kernel!" << std::endl;
    return 0;
}
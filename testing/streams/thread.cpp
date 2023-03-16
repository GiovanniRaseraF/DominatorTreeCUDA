#include <thread>
#include <iostream>
#include <vector>

//const int N = 1;

// __global__ void kernel(){

// }

void launch_kernel(){
    std::cout << "id: " << std::this_thread::get_id() << std::endl;

    float *data;
    //cudaMalloc(&data, N * sizeof(float));

    //kernel<<<1, 64>>>();
}

int main(){
    const int numth = 8;
    std::vector<std::thread> thvet;

    for(int i = 0; i < numth; i++){
        thvet.push_back(std::thread{launch_kernel});
    }

    for(auto &t : thvet){
        t.join();
    }
}
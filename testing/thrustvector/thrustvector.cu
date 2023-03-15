#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// Cuda example implementation
struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return a * x + y;
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(){
    std::cout << "CUDA: thrust vector testing" << std::endl;
    {
        thrust::host_vector<float> host_X{128*1024*1024}; 

        for(int i = 0; i < host_X.size(); i++){
            host_X[i] = host_X.size() - i;
        }
        
        thrust::device_vector<float> dv_X{host_X.begin(), host_X.end()};
        thrust::sort(dv_X.begin(), dv_X.end());

        std::cout << "done sorting" << std::endl;
    }

    {
        std::cout << "thrust saxpy" << std::endl;
        thrust::device_vector<float> dv_X{1024*1024};
        thrust::device_vector<float> dv_Y{1024*1024};

        saxpy_fast(123, dv_X, dv_Y);
        saxpy_slow(123, dv_X, dv_Y);
    }

    {
        std::cout << "thrust reduce" << std::endl;

        // this is 4 Gb of memory that is the maximum for int rappresentation
        // 2 ^ 32 = 4 GB
        // float are 32 bits
        thrust::host_vector<float> host_X{1024*1024*1024}; 

        for(int i = 0; i < host_X.size(); i++){
            host_X[i] = i;
        }

        thrust::device_vector<float> dv_X{host_X.begin(), host_X.end()};

        auto sum = thrust::reduce(dv_X.begin(), dv_X.end());

        std::cout << "sum: " << sum << std::endl;
    }

    return 0;
}
#include "for_each.cuh"
#include "reduce.cuh"
#include <cuda.h>
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>

template <typename T>
class mark
{
    public:
    __host__ __device__ void operator()(T x){}
};

template <typename T>
class sums 
{
    public:
    __host__ __device__ T operator()(T a, T b){ return a + b;}
};
int main(){
    std::vector<int> a(100, 1);
    thrust::device_vector<int> da(a.begin(), a.end());
    mark<int> func;

    // for each
    parallel::for_each(da.begin(), da.end(), func);
    seq::for_each(a.begin(), a.end(),func); 
    
    // reduce
    auto ret = seq::reduce(a.begin(), a.end(), 0, [](int x, int y){return x + y;});
    std::cout << "seq::reduce> " << ret << std::endl;

    ret = parallel::reduce(da.begin(), da.end(), 0, sums<int>{});
    cudaDeviceSynchronize();
    std::cout << "parallel::reduce> " << ret << std::endl;
}
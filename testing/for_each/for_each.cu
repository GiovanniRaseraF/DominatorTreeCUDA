#include <vector>
#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>

namespace parallel
{
    template <
        typename InputIterator,
        typename UnaryFunction>
    __global__
    void for_each_kernel(
        InputIterator first,
        InputIterator last,
        UnaryFunction unifunc)
    {
        const int grid_size = blockDim.x * gridDim.x;

        first += blockIdx.x * blockDim.x + threadIdx.x;

        while (first < last)
        {
            unifunc(*first);
            first += grid_size;
        }
    }

    // for each
    template <
        typename InputIterator,
        typename UnaryFunction>
    void for_each(
        InputIterator first,
        InputIterator last,
        UnaryFunction unifunc)
    {
        if (first >= last)
            return;

        const std::size_t BLOCK_SIZE = 256;
        const size_t MAX_BLOCKS = 1024;
        const size_t NUM_BLOCKS = std::min(MAX_BLOCKS, ((last - first) + (BLOCK_SIZE - 1)) / BLOCK_SIZE);

        for_each_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(first, last, unifunc);
    }
}

template <typename T>
class mark
{
    public:
    __host__ __device__ void operator()(T x){printf("%d\n", x);}
};

#define N 100000

int main()
{
    std::cout << "CUDA: for_each" << std::endl;
    std::array<int, N> hh = {};
    for(int i = 0; i < N; i++) hh[i] = i;

    thrust::device_vector<int> test{hh.begin(), hh.end()};
    mark<int> s;
    //parallel::for_each(test.begin(), test.end(), s);

    cudaDeviceSynchronize();
}
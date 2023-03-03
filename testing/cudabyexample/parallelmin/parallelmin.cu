#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <limits>
#include <array>

#include <cstring>
#include <cmath>

#include <omp.h>

constexpr size_t N = 10;
constexpr int MAX_THREAD = 2;

// parallel implementation depending on the fisical thread num 
int parallelmin(const std::vector<int> &A){
    std::array<int, MAX_THREAD> array = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};

    const int maxthreads = MAX_THREAD;
    const int range = std::floor(A.size() / maxthreads);

    #pragma omp parallel 
    {
        const int thread_num = omp_get_thread_num();
        const int start = thread_num * range;
        int end = start + range;

        if(MAX_THREAD == thread_num+1)
            end = A.size();

        for(int i = start; i < end; i++){
            if(A[i] < array[thread_num]){
                array[thread_num] = A[i];
                //std::cout << array[thread_num] << std::endl;
            }
        }

    }

    return *std::min(array.begin(), array.end());
}

int main(){
    std::cout << __FILE__ << ": " << MAX_THREAD << " cpu add" << std::endl;

    // openmp setup
    const int hostcpu_maxthreads = omp_get_max_threads();
    if(MAX_THREAD > hostcpu_maxthreads){
        std::cerr << "cannot use " << MAX_THREAD << " threads, you have " << hostcpu_maxthreads << " that you can use" << std::endl;
        exit(1);    
    }

    // setting number of threads
    omp_set_num_threads(MAX_THREAD);
    std::cout << "setted num_threads to " << MAX_THREAD << std::endl;

    // random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(40, 60);
    
    // allocate memory
    std::vector<int> A(N, 0);
    // init random
    for(auto &v : A){
        v = distrib(gen);
    }

    // perform min
    const int min = parallelmin(A);

    for(auto v : A) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << std::setw(25) << std::left << "parallel min" << ": " << min << std::endl;    
    std::cout << std::setw(25) << std::left << "sequential min" << ": " << (*std::min(A.begin(), A.end())) << std::endl;

    return 0;
}
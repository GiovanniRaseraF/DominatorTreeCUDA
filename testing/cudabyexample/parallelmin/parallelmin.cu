#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <limits>
#include <array>
#include <ranges>
#include <execution>

#include <cstring>
#include <cmath>

#include <omp.h>

#include "test.hpp"

constexpr size_t N = 99999999;
constexpr int MAX_THREAD = 16;

// parallel implementation depending on the fisical thread num 
int parallelmin(const std::vector<int> &A){
    std::array<int, MAX_THREAD> array;
    for(auto &v : array) v = std::numeric_limits<int>::max();

    const int maxthreads = MAX_THREAD;
    const int range = std::floor(A.size() / maxthreads);

    #pragma omp parallel 
    {
        const int thread_num = omp_get_thread_num();
        const int start = thread_num * range;
        int end = start + range;

        if(MAX_THREAD == thread_num+1)
            end = A.size();

        for(int i = start; i < end; i++)
            if(A[i] < array[thread_num])
                array[thread_num] = A[i];

    }

    //for(auto a : array) std::cout << a << " ";
    auto min = std::min_element(array.begin(), array.end());

    return *min;
}

int mysecmin(const std::vector<int> &A){
    int min = A[0];

    for(int i = 0; i < A.size(); i++)
        if(A[i] < min)
            min = A[i];
    
    return min;
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
    std::uniform_int_distribution<int> distrib(2, N*5);
    
    // allocate memory
    std::vector<int> A(N, 0);
    // init random
    for(auto &v : A){
        v = distrib(gen);
    }

    // perform min
    int min, my_sec_min, sec_min, exec_par_min, exec_par_unseq_min;

    CHRONO_TEST(
        min = parallelmin(A),
        "find min in parallel"
    )

    CHRONO_TEST(
        my_sec_min = mysecmin(A),
        "find min with for loop"
    )
    CHRONO_TEST(
        sec_min = *std::min_element(A.begin(), A.end()),
        "find min sequential with std::min_element"
    )

    CHRONO_TEST(
	    exec_par_min = *std::min_element(std::execution::par, A.begin(), A.end()),
	    "find min using std::exec::par"
    )

    CHRONO_TEST(
	    exec_par_unseq_min = *std::min_element(std::execution::par_unseq, A.begin(), A.end()),
	    "find min using std::exec::par_unseq"
    )

    // output
    std::cout << std::setw(25) << std::left << "parallel min" << ": " << min << std::endl;    
    std::cout << std::setw(25) << std::left << "my sec min min" << ": " << my_sec_min << std::endl;    
    std::cout << std::setw(25) << std::left << "sequential min" << ": " << sec_min << std::endl;
    std::cout << std::setw(25) << std::left << "std::exec::par min" << ": " << exec_par_min << std::endl;
    std::cout << std::setw(25) << std::left << "std::exec::par_unseq min" << ": " << exec_par_unseq_min << std::endl;

    return 0;
}

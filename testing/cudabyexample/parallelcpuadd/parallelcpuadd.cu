#include <iostream>
#include <omp.h>
#include <cstring>

constexpr size_t N = 999;
constexpr int MAX_THREAD = 2;

// parallel implementation depending on the fisical thread num 
void paralleladd(int *A, int *B, int *C, size_t size){
    #pragma omp parallel
    {
        const int threadnum = omp_get_thread_num();

        for(size_t i = (size_t)threadnum; i < size; i+=MAX_THREAD){
            C[i] = A[i] + B[i];
        }
    }
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

    // allocate memory
    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];

    // all zeros to be shore
    std::memset(A, 0, N*sizeof(int));
    std::memset(B, 0, N*sizeof(int));
    std::memset(C, 0, N*sizeof(int));

    for(size_t i = 0; i < N; i++){
        A[i] = 0;
        B[i] = 1;
    }

    paralleladd(A, B, C, N);

    std::cout << "Done." << std::endl;    

    return 0;
}
#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <memory>
#include <thread>
#include <vector>
#include <math.h>

int main(){
    std::cout << "omp test cudaversion: "  << std::endl;
    std::vector<int> vet(99999999, 1); 
    int partialsum = 0;
    int totalsum = 0;
    const int maxthreads = omp_get_max_threads();
    const int range = std::floor(vet.size() / maxthreads);
    
    std::cout << "maxthreads: " << maxthreads << std::endl;
    std::cout << "range: " << range << std::endl;

    #pragma omp parallel private(partialsum) shared(totalsum)
    {
        partialsum = 0;
        int start = omp_get_thread_num() * range;
        int end = start + range;

        if(omp_get_max_threads() == omp_get_thread_num()+1)
            end = vet.size();

        for(int i = start; i < end; i++){
            partialsum += vet[i];
        }
        
        #pragma omp critical
        {
            totalsum += partialsum;
        }
    }

    std::cout << "result: " << totalsum << std::endl;

    return 0;
}
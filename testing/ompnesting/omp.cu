#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <memory>
#include <thread>
#include <vector>
#include <math.h>

using namespace std::chrono_literals;

void print_t(int sect, int level){
    std::cout << "this: " << omp_get_thread_num() << " S: " << sect << " level: " << level << " -team threads: " 
        << omp_get_num_threads() << std::endl;
}

int main(){
    std::cout << "omp nesting " << std::endl;
    // setting nested parallel threads
    omp_set_nested(1);

    #pragma omp parallel num_threads(2)
    {
        #pragma omp critical
        {
            print_t(1, 0);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                #pragma omp parallel num_threads(3)
                {
                    #pragma omp critical
                    {
                        print_t(2, 1);
                    }
                }
            }

            #pragma omp section
            {
                #pragma omp parallel num_threads(2)
                {
                    #pragma omp critical
                    {
                        print_t(2, 2);
                    }
                }
            }
        }
    }

    // flush
    // flush means to force memory coherence
    // between barriers no memory coherence can be found

    // if you have to force it you need to use flush
    
    // example
    // simple producer consumer
    bool ready = false;
    int numitems = 0;
    bool temp_ready = false;
    #pragma omp parallel sections shared(numitems)
    {
        #pragma omp section
        {
            while(true){
                // create something
                if(numitems <= 0) numitems++;

                #pragma omp flush
                #pragma omp atomic write
                ready = true; // need to comunicate ready state
                #pragma omp flush(ready)
            }
        }

        #pragma omp section
        {
            while(true){
                while(true){
                    #pragma omp flush(ready)
                    {
                        #pragma omp atomic read
                        temp_ready = ready;

                        if (temp_ready) break;
                    }
                }

                // can consume data
                if (numitems > 0){
                    numitems--;
                    std::cout << numitems << std::endl;

                    std::this_thread::sleep_for(1s);
                }
            }
                
        }
    }

}
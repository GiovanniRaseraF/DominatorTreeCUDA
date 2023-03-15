#pragma once

#include <iostream>
#include <algorithm>
#include <memory>
#include <iomanip>

namespace cuda{
    template <typename T>
    class vector{
        private:
        T *host_vector_ptr;
        T *device_vector_ptr;
        std::size_t num_elems;

        public:
        /*
        No copy is going to be done here
        use mirror to copy hostToDevice and 
        deviceToHost
        */
        vector(int size, T defvalue) : num_elems(size){
            int size_on_gpu = size * sizeof(T);

            // host allocation
            host_vector_ptr = new T[size];

            // init
            for(int i = 0; i < this->size(); i++)
                host_vector_ptr[i] = defvalue;

            // device allocation
            cudaMalloc(
                (void **) &device_vector_ptr,
                size_on_gpu
            );
        }

        ~vector(){
            if(device_vector_ptr != nullptr)
                cudaFree(device_vector_ptr);

            if(host_vector_ptr != nullptr)
                delete [] host_vector_ptr;
        }

       
        /*
        Copy memory from device to host
        and host to device
        */ 
        cudaError_t mirror(enum cudaMemcpyKind kind){
            int size_on_gpu = this->size() * sizeof(T);
            cudaError_t ret;

            switch (kind){
            case cudaMemcpyKind::cudaMemcpyHostToDevice:
                ret = cudaMemcpy(
                    device_vector_ptr,
                    host_vector_ptr,
                    size_on_gpu,
                    cudaMemcpyHostToDevice
                );

                break;
                
            case cudaMemcpyKind::cudaMemcpyDeviceToHost:
                ret = cudaMemcpy(
                    host_vector_ptr,
                    device_vector_ptr,
                    size_on_gpu,
                    cudaMemcpyDeviceToHost
                );
                break;

            default:
                break;
            }

            return ret;
        }

        inline std::size_t size() const {
            return num_elems;
        }

        /*
        [] operator, you need to mirror the informations
        to the device if you want to see changes
        */ 
        T& operator[](int index){
            if (index >= this->size()) exit(0);

            return host_vector_ptr[index];
        }

        // ON DEVICE 
        T* get(){
            return device_vector_ptr;
        }
    };
}

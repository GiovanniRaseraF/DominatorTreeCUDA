#include "book.h"
#include <iostream>
#include <iomanip>

void printinfo(std::ostream &ost, const cudaDeviceProp &prop, int num){
    ost << "GPU" << num << " info:" << std::endl;

    ost << std::setw(10) << std::left << "Name" << prop.name << std::endl;
    ost << std::setw(10) << "" << prop.totalGlobalMem << std::endl;
    ost << std::setw(10) << "" << prop.sharedMemPerBlock << std::endl;
    ost << std::setw(10) << "" << prop.regsPerBlock << std::endl;
    ost << std::setw(10) << "" << prop.warpSize << std::endl;
    ost << std::setw(10) << "" << prop.memPitch << std::endl;
    ost << std::setw(10) << "" << prop.maxThreadsPerBlock << std::endl;
    ost << std::setw(10) << "" << prop.maxThreadsDim << std::endl;
    ost << std::setw(10) << "" << prop.maxGridSize << std::endl;
    ost << std::setw(10) << "" << prop.totalConstMem << std::endl;
    ost << std::setw(10) << "" << prop.major << "." << prop.minor << std::endl;
    ost << std::setw(10) << "" << prop.clockRate << std::endl;
    ost << std::setw(10) << "" << prop.textureAlignment << std::endl;
    ost << std::setw(10) << "" << prop.deviceOverlap << std::endl;
    ost << std::setw(10) << "" << prop.multiProcessorCount << std::endl;
    ost << std::setw(10) << "" << prop.kernelExecTimeoutEnabled << std::endl;
    ost << std::setw(10) << "" << prop.integrated << std::endl;
    ost << std::setw(10) << "" << prop.canMapHostMemory << std::endl;
    ost << std::setw(10) << "" << prop.computeMode << std::endl;
    ost << std::setw(10) << "" << prop.maxTexture1D << std::endl;
    ost << std::setw(10) << "" << prop.maxTexture2D << std::endl;
    ost << std::setw(10) << "" << prop.maxTexture3D << std::endl;
    ost << std::setw(10) << "" << prop.concurrentKernels << std::endl;
}

int main(){
    std::cout << __FILE__ << ": gpu informtions" << std::endl;
    cudaDeviceProp prop;

    // get the number of nvidia gpus in the system
    int fisicaldevicecount;
    HANDLE_ERROR(
        cudaGetDeviceCount(
            &fisicaldevicecount
        )
    );

    for(int i = 0; i < fisicaldevicecount; i++){
        // get the actual informations of the gpu
        HANDLE_ERROR(
            cudaGetDeviceProperties(
                &prop,
                i
            )
        );

        // display info
        printinfo(std::cout, prop, i);
    }
}
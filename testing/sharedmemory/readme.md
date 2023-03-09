# Profiler
Profiling reversing a vector with 512 floats allocating a shared memory in STATIC e DINAMIC

```bash
make
```
# Output
## CUDA: Reverse with shared STATIC memory allocation
```
CUDA: Reverse with shared STATIC memory allocation
==205018== NVPROF is profiling process 205018, command: ./build/shared_static_alloc.out
==205018== Profiling application: ./build/shared_static_alloc.out
==205018== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.4480us         1  4.4480us  4.4480us  4.4480us  reverse(float*, int)
      API calls:   99.14%  85.785ms         1  85.785ms  85.785ms  85.785ms  cudaHostAlloc
                    0.33%  288.29us         1  288.29us  288.29us  288.29us  cudaFreeHost
                    0.23%  201.99us         1  201.99us  201.99us  201.99us  cuLibraryLoadData
                    0.22%  188.05us       114  1.6490us     230ns  76.520us  cuDeviceGetAttribute
                    0.04%  32.606us         1  32.606us  32.606us  32.606us  cuDeviceGetName
                    0.02%  20.907us         1  20.907us  20.907us  20.907us  cudaLaunchKernel
                    0.01%  7.2290us         1  7.2290us  7.2290us  7.2290us  cuDeviceGetPCIBusId
                    0.00%  2.5390us         3     846ns     420ns  1.4790us  cuDeviceGetCount
                    0.00%  1.8490us         1  1.8490us  1.8490us  1.8490us  cudaHostGetDevicePointer
                    0.00%     910ns         2     455ns     260ns     650ns  cuDeviceGet
                    0.00%     470ns         1     470ns     470ns     470ns  cuDeviceTotalMem
                    0.00%     460ns         1     460ns     460ns     460ns  cuModuleGetLoadingMode
                    0.00%     340ns         1     340ns     340ns     340ns  cuDeviceGetUuid

```

## CUDA: Reverse with shared DINAMIC memory allocation
```
CUDA: Reverse with shared DINAMIC memory allocation
==205038== NVPROF is profiling process 205038, command: ./build/shared_dinamic_alloc.out
==205038== Profiling application: ./build/shared_dinamic_alloc.out
==205038== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.4150us         1  4.4150us  4.4150us  4.4150us  reverse(float*, int)
      API calls:   99.19%  92.406ms         1  92.406ms  92.406ms  92.406ms  cudaHostAlloc
                    0.31%  291.81us         1  291.81us  291.81us  291.81us  cudaFreeHost
                    0.23%  211.13us         1  211.13us  211.13us  211.13us  cuLibraryLoadData
                    0.20%  185.38us       114  1.6260us     240ns  74.200us  cuDeviceGetAttribute
                    0.03%  30.446us         1  30.446us  30.446us  30.446us  cuDeviceGetName
                    0.02%  20.497us         1  20.497us  20.497us  20.497us  cudaLaunchKernel
                    0.01%  5.9190us         1  5.9190us  5.9190us  5.9190us  cuDeviceGetPCIBusId
                    0.00%  2.3990us         3     799ns     420ns  1.3690us  cuDeviceGetCount
                    0.00%  1.6590us         1  1.6590us  1.6590us  1.6590us  cudaHostGetDevicePointer
                    0.00%     780ns         2     390ns     270ns     510ns  cuDeviceGet
                    0.00%     560ns         1     560ns     560ns     560ns  cuModuleGetLoadingMode
                    0.00%     490ns         1     490ns     490ns     490ns  cuDeviceTotalMem
                    0.00%     340ns         1     340ns     340ns     340ns  cuDeviceGetUuid
```
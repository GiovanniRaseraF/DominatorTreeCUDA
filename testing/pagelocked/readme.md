# Profiler
```bash
make
```
# Output
## CUDA: add PAGE LOCKED
```bash
CUDA: add PAGE LOCKED
nvprof ./build/pagelocked.out
CUDA: add PAGE LOCKED
==161816== NVPROF is profiling process 161816, command: ./build/pagelocked.out
==161816== Profiling application: ./build/pagelocked.out
==161816== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.4480us         1  4.4480us  4.4480us  4.4480us  vetadd(float*, float*, float*)
      API calls:   98.98%  95.256ms         3  31.752ms  3.2400us  95.249ms  cudaHostAlloc
                    0.48%  457.21us         3  152.40us  3.2800us  359.50us  cudaFreeHost
                    0.24%  234.22us         1  234.22us  234.22us  234.22us  cuLibraryLoadData
                    0.22%  208.51us       114  1.8290us     270ns  83.981us  cuDeviceGetAttribute
                    0.05%  47.641us         1  47.641us  47.641us  47.641us  cuDeviceGetName
                    0.02%  21.020us         1  21.020us  21.020us  21.020us  cudaLaunchKernel
                    0.01%  7.9010us         1  7.9010us  7.9010us  7.9010us  cuDeviceGetPCIBusId
                    0.00%  2.8700us         3     956ns     430ns  1.6400us  cuDeviceGetCount
                    0.00%  2.2100us         3     736ns     480ns  1.2000us  cudaHostGetDevicePointer
                    0.00%  1.0600us         2     530ns     280ns     780ns  cuDeviceGet
                    0.00%     550ns         1     550ns     550ns     550ns  cuDeviceTotalMem
                    0.00%     520ns         1     520ns     520ns     520ns  cuModuleGetLoadingMode
                    0.00%     380ns         1     380ns     380ns     380ns  cuDeviceGetUuid
```
## CUDA: add PAGE LOCKED + MAPPED ZERO COPY
```bash
./build/pagelocked_mapped_zero_copy.out
CUDA: add PAGE LOCKED + MAPPED ZERO COPY
nvprof ./build/pagelocked_mapped_zero_copy.out
CUDA: add PAGE LOCKED + MAPPED ZERO COPY
==161836== NVPROF is profiling process 161836, command: ./build/pagelocked_mapped_zero_copy.out
==161836== Profiling application: ./build/pagelocked_mapped_zero_copy.out
==161836== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.3520us         1  4.3520us  4.3520us  4.3520us  vetadd(float*, float*, float*)
      API calls:   99.22%  79.977ms         3  26.659ms  3.0300us  79.969ms  cudaHostAlloc
                    0.37%  300.62us         3  100.21us  3.2600us  288.36us  cudaFreeHost
                    0.17%  134.72us       114  1.1810us     120ns  57.151us  cuDeviceGetAttribute
                    0.15%  118.33us         1  118.33us  118.33us  118.33us  cuLibraryLoadData
                    0.05%  40.111us         1  40.111us  40.111us  40.111us  cuDeviceGetName
                    0.03%  20.630us         1  20.630us  20.630us  20.630us  cudaLaunchKernel
                    0.01%  9.6710us         1  9.6710us  9.6710us  9.6710us  cuDeviceGetPCIBusId
                    0.00%  2.1900us         3     730ns     500ns  1.1800us  cudaHostGetDevicePointer
                    0.00%  2.1800us         3     726ns     300ns  1.5300us  cuDeviceGetCount
                    0.00%     860ns         2     430ns     200ns     660ns  cuDeviceGet
                    0.00%     520ns         1     520ns     520ns     520ns  cuDeviceTotalMem
                    0.00%     510ns         1     510ns     510ns     510ns  cuModuleGetLoadingMode
                    0.00%     250ns         1     250ns     250ns     250ns  cuDeviceGetUuid

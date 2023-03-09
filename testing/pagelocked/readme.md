# Profiler
Profiling with 256 Blocks and 256 ThreadsPerBlock

```bash
make
```
# Output
## CUDA: add PAGE LOCKED
```
CUDA: add PAGE LOCKED
nvprof ./build/pagelocked.out
CUDA: add PAGE LOCKED
==195463== NVPROF is profiling process 195463, command: ./build/pagelocked.out
==195463== Profiling application: ./build/pagelocked.out
==195463== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  51.776us         1  51.776us  51.776us  51.776us  vetadd(float*, float*, float*)
      API calls:   99.13%  88.374ms         3  29.458ms  2.3500us  88.368ms  cudaHostAlloc
                    0.33%  292.94us         3  97.646us  2.3300us  285.09us  cudaFreeHost
                    0.23%  206.55us         1  206.55us  206.55us  206.55us  cuLibraryLoadData
                    0.21%  190.49us       114  1.6700us     230ns  77.910us  cuDeviceGetAttribute
                    0.05%  46.130us         1  46.130us  46.130us  46.130us  cuDeviceGetName
                    0.02%  20.570us         1  20.570us  20.570us  20.570us  cudaLaunchKernel
                    0.01%  8.1200us         1  8.1200us  8.1200us  8.1200us  cuDeviceGetPCIBusId
                    0.00%  2.7000us         3     900ns     390ns  1.7000us  cuDeviceGetCount
                    0.00%  2.2900us         3     763ns     490ns  1.3000us  cudaHostGetDevicePointer
                    0.00%     920ns         2     460ns     250ns     670ns  cuDeviceGet
                    0.00%     530ns         1     530ns     530ns     530ns  cuDeviceTotalMem
                    0.00%     520ns         1     520ns     520ns     520ns  cuModuleGetLoadingMode
                    0.00%     340ns         1     340ns     340ns     340ns  cuDeviceGetUuid
```

## CUDA: add PAGE LOCKED + MAPPED ZERO COPY
```
CUDA: add PAGE LOCKED + MAPPED ZERO COPY
nvprof ./build/pagelocked_mapped_zero_copy.out
CUDA: add PAGE LOCKED + MAPPED ZERO COPY
==195484== NVPROF is profiling process 195484, command: ./build/pagelocked_mapped_zero_copy.out
==195484== Profiling application: ./build/pagelocked_mapped_zero_copy.out
==195484== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  52.799us         1  52.799us  52.799us  52.799us  vetadd(float*, float*, float*)
      API calls:   99.05%  84.532ms         3  28.177ms  2.2700us  84.526ms  cudaHostAlloc
                    0.34%  289.85us         3  96.616us  2.3700us  282.31us  cudaFreeHost
                    0.27%  232.03us         1  232.03us  232.03us  232.03us  cuLibraryLoadData
                    0.24%  208.69us       114  1.8300us     270ns  83.880us  cuDeviceGetAttribute
                    0.05%  46.000us         1  46.000us  46.000us  46.000us  cuDeviceGetName
                    0.02%  20.319us         1  20.319us  20.319us  20.319us  cudaLaunchKernel
                    0.01%  5.5700us         1  5.5700us  5.5700us  5.5700us  cuDeviceGetPCIBusId
                    0.00%  2.7100us         3     903ns     440ns  1.6800us  cuDeviceGetCount
                    0.00%  2.2100us         3     736ns     490ns  1.2300us  cudaHostGetDevicePointer
                    0.00%     920ns         2     460ns     300ns     620ns  cuDeviceGet
                    0.00%     580ns         1     580ns     580ns     580ns  cuModuleGetLoadingMode
                    0.00%     520ns         1     520ns     520ns     520ns  cuDeviceTotalMem
                    0.00%     390ns         1     390ns     390ns     390ns  cuDeviceGetUuid
```

## CUDA: add cudaMalloc
```
CUDA: mat add
==195583== NVPROF is profiling process 195583, command: build/gpuvetadd.out
==195583== Profiling application: build/gpuvetadd.out
==195583== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   72.90%  64.640us         3  21.546us  21.472us  21.696us  [CUDA memcpy HtoD]
                   23.06%  20.448us         1  20.448us  20.448us  20.448us  [CUDA memcpy DtoH]
                    4.04%  3.5840us         1  3.5840us  3.5840us  3.5840us  vetadd(float*, float*, float*)
      API calls:   99.27%  96.648ms         3  32.216ms  1.8300us  96.643ms  cudaMalloc
                    0.21%  199.73us         1  199.73us  199.73us  199.73us  cuLibraryLoadData
                    0.20%  195.12us       114  1.7110us     230ns  80.010us  cuDeviceGetAttribute
                    0.18%  173.69us         4  43.422us  32.630us  57.689us  cudaMemcpy
                    0.07%  72.680us         3  24.226us  1.9100us  66.460us  cudaFree
                    0.03%  31.470us         1  31.470us  31.470us  31.470us  cuDeviceGetName
                    0.02%  22.270us         1  22.270us  22.270us  22.270us  cudaLaunchKernel
                    0.01%  7.5000us         1  7.5000us  7.5000us  7.5000us  cuDeviceGetPCIBusId
                    0.00%  2.7500us         3     916ns     350ns  1.7300us  cuDeviceGetCount
                    0.00%     970ns         2     485ns     240ns     730ns  cuDeviceGet
                    0.00%     440ns         1     440ns     440ns     440ns  cuDeviceTotalMem
                    0.00%     430ns         1     430ns     430ns     430ns  cuModuleGetLoadingMode
                    0.00%     330ns         1     330ns     330ns     330ns  cuDeviceGetUuid
```

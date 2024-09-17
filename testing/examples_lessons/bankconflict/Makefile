
globalmemory: 
	nvcc sharedmem.cu -D$(alloc) -D$(cudaSync) -Dthreads=$(threads) -Dblocks=$(blocks) -o sharedmem.out

test:
	make alloc=DYNAMIC cudaSync=cudaSyncNone threads=64 blocks=128 && nvprof ./sharedmem.out
	make alloc=STATIC cudaSync=cudaSyncNone threads=64 blocks=128 && nvprof ./sharedmem.out

testsync:
	make alloc=DYNAMIC cudaSync=cudaSync  threads=64 blocks=128 && nvprof ./sharedmem.out
	make alloc=STATIC cudaSync=cudaSync threads=64 blocks=128 && nvprof ./sharedmem.out
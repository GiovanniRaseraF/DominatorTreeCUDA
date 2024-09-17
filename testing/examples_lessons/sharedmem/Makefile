
globalmemory: 
	nvcc sharedmem.cu -D$(alloc) -Dthreads=$(threads) -Dblocks=$(blocks) -o sharedmem.out

test:
	make alloc=DYNAMIC threads=64 blocks=128 && nvprof ./sharedmem.out
	make alloc=STATIC threads=64 blocks=128 && nvprof ./sharedmem.out

streams: 
	nvcc streams.cu -D$(alloc) -D$(cudaSync) -Dthreads=$(threads) -Dblocks=$(blocks) -o streams.out

test:
	make alloc=DYNAMIC cudaSync=cudaSyncNone threads=1024 blocks=1024 && nvprof ./streams.out
	make alloc=STATIC cudaSync=cudaSyncNone threads=1024 blocks=1024 && nvprof ./streams.out
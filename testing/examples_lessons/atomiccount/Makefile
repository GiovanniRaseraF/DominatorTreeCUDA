name = parallelreduction

# Author: Giovanni Rasera

count:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $@.cu -std=c++11 -Dscalar=$(scalar) -Dthreads=$(threads) -Dblocks=$(blocks)  -o ./build/nvcc_$@.o
	g++ -o ./build/$@.out ./build/nvcc_$@.o -L/usr/local/cuda/lib64 -Dscalar=$(scalar) -Dthreads=$(threads) -Dblocks=$(blocks) -lcusparse -lcudart

	./build/$@.out
	#nvprof ./build/$@.out --benchmark

test: 
	make threads=1024 blocks=1024 scalar=950 OPTIM=3	
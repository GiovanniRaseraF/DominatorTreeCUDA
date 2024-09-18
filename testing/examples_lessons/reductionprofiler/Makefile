name = parallelreduction

# Author: Giovanni Rasera

parallelreduction:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $@.cu -std=c++11 -DNAIVE_REDUCE -Dthreads=$(threads) -Dblocks=$(blocks) -o ./build/nvcc_$@.o
	g++ -o ./build/$@.out ./build/nvcc_$@.o -L/usr/local/cuda/lib64 -Dthreads=$(threads) -Dblocks=$(blocks) -lcusparse -lcudart

	nvprof ./build/$@.out

# make threads=1024 blocks=512 OPTIM=O0 sequentialreduction
# make threads=1024 blocks=512 OPTIM=O1 sequentialreduction
# make threads=1024 blocks=512 OPTIM=O2 sequentialreduction
# make threads=1024 blocks=512 OPTIM=O3 sequentialreduction
# make threads=1024 blocks=512 OPTIM=Os sequentialreduction
sequentialreduction:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	g++ -o ./build/$@_seq.out ./$@.cpp -Dthreads=$(threads) -Dblocks=$(blocks) -$(OPTIM)
	
	./build/$@_seq.out


test: 
	make threads=1024 blocks=512 OPTIM=O0 parallelreduction 
	make threads=1024 blocks=512 OPTIM=O1 parallelreduction
	make threads=1024 blocks=512 OPTIM=O2 parallelreduction
	make threads=1024 blocks=512 OPTIM=O3 parallelreduction
	make threads=1024 blocks=512 OPTIM=Os parallelreduction

	make threads=1024 blocks=512 OPTIM=O0 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O1 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O2 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O3 sequentialreduction
	make threads=1024 blocks=512 OPTIM=Os sequentialreduction
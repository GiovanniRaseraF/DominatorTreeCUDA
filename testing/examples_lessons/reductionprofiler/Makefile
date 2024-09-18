name = parallelreduction

# Author: Giovanni Rasera

parallelreduction:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $@.cu -std=c++11 -Dthreads=$(threads) -Dblocks=$(blocks) -D$(REDUCTION) -o ./build/nvcc_$@.o
	g++ -o ./build/$@.out ./build/nvcc_$@.o -L/usr/local/cuda/lib64 -Dthreads=$(threads) -Dblocks=$(blocks) -lcusparse -lcudart

	./build/$@.out
	# nvprof ./build/$@.out -benchmark -numdevices=1 -i=3

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
# 	make threads=1024 blocks=512 OPTIM=O0 REDUCTION=SPEED_REDUCE parallelreduction
 	#make threads=1024 blocks=512 OPTIM=O1 REDUCTION=SPEED_REDUCE parallelreduction
 	#make threads=1024 blocks=512 OPTIM=O2 REDUCTION=SPEED_REDUCE parallelreduction
 	#make threads=1024 blocks=512 OPTIM=O3 REDUCTION=SPEED_REDUCE parallelreduction
 	#make threads=1024 blocks=512 OPTIM=Os REDUCTION=SPEED_REDUCE parallelreduction

	make threads=1024 blocks=512 OPTIM=O0 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O1 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O2 sequentialreduction
	make threads=1024 blocks=512 OPTIM=O3 sequentialreduction
	make threads=1024 blocks=512 OPTIM=Os sequentialreduction
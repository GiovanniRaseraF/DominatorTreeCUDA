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
	make threads=1024 blocks=1024 scalar=3 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=4 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=5 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=6 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=8 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=10 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=12 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=15 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=18 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=22 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=27 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=33 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=40 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=48 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=58 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=70 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=84 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=101 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=122 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=147 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=177 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=213 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=256 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=308 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=370 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=444 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=533 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=640 OPTIM=3 --silent
	make threads=1024 blocks=1024 scalar=768 OPTIM=3 --silent
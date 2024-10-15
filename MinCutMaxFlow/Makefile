# Author: Giovanni Rasera
V=1024

cpu:
	#if [ -d "./build" ]; then rm -r build; fi
	#mkdir build
	g++ -o ./build/$@_sequential.out ./$@.cpp -DV=$V -std=c++17

	./build/$@_sequential.out

gpu:
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $@.cu -std=c++11 -Dthreads=$(threads) -Dblocks=$(blocks) -D$(REDUCTION) -o ./build/nvcc_$@.o
	g++ -o ./build/$@_parallel.out ./build/nvcc_$@.o -L/usr/local/cuda/lib64 -Dthreads=$(threads) -Dblocks=$(blocks) -lcusparse -lcudart

	./build/$@_parallel.out
	nvprof ./build/$@_parallel.out --benchmark

test:
	make cpu STRUCT=USEVECTOR V=1024
	make cpu STRUCT=USEVECTOR V=2024
	make cpu STRUCT=USEVECTOR V=4024
	make cpu STRUCT=USEVECTOR V=8024
	make cpu STRUCT=USEVECTOR V=16024
	make cpu STRUCT=USEVECTOR V=32024
name = parallelreduction

name = parallelreduction

parallelreduction:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $@.cu -std=c++11 -DNAIVE_REDUCE -o ./build/nvcc_$@.o
	g++ -o ./build/$@.out ./build/nvcc_$@.o -L/usr/local/cuda/lib64 -lcusparse -lcudart

	nvprof ./build/$@.out
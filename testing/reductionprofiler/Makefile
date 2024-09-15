name = parallelreduction

name = parallelreduction

parallelreduction:
	nvcc $@.cu -std=c++17 -lcusparse -lcudart -DNAIVE_REDUCE -o ./build/$@.out
	./build/$@.out
	
	nvprof ./build/$@.out
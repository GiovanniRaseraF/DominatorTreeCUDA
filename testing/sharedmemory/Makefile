basename = shared_memory
name = shared_static_alloc
name2 = shared_dinamic_alloc

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build

	# nvcc $@.cu  -I../cudabyexample/common/ -o ./build/$@.out
	# ./build/$@.out
	# nvprof ./build/$@.out

	nvcc -c -I/usr/local/cuda/include -I../cudabyexample/common/ $(basename).cu -o ./build/nvcc_$(name).o
	g++ -DDEBUG -o ./build/$(name).out ./build/nvcc_$(name).o -L/usr/local/cuda/lib64 -lcusparse -lcudart
	
	nvcc -c -DDINAMIC_SHARED_MEM -I/usr/local/cuda/include -I../cudabyexample/common/ $(basename).cu -o ./build/nvcc_$(name2).o
	g++ -DDEBUG -o ./build/$(name2).out ./build/nvcc_$(name2).o -L/usr/local/cuda/lib64 -lcusparse -lcudart

	echo "\nCompiled\n -- Profiling --"
	./build/$(name).out
	nvprof ./build/$(name).out

	./build/$(name2).out
	nvprof ./build/$(name2).out

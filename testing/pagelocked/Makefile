name = pagelocked

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	nvcc -c -DMAPPED_ZERO_COPY -I/usr/local/cuda/include -I../cudabyexample/common/ $(name).cu -o ./build/nvcc_$(name)_mapped_zero_copy.o
	g++ -DDEBUG -o ./build/$(name)_mapped_zero_copy.out ./build/nvcc_$(name)_mapped_zero_copy.o -L/usr/local/cuda/lib64 -lcusparse -lcudart
	
	nvcc -c -I/usr/local/cuda/include -I../cudabyexample/common/ $(name).cu -o ./build/nvcc_$(name).o
	g++ -DDEBUG -o ./build/$(name).out ./build/nvcc_$(name).o -L/usr/local/cuda/lib64 -lcusparse -lcudart

	echo "\nCompiled\n -- Profiling --"
	./build/$(name).out
	nvprof ./build/$(name).out
	./build/$(name)_mapped_zero_copy.out
	nvprof ./build/$(name)_mapped_zero_copy.out

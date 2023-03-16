name = threadedstreams

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	nvcc -c -I/usr/local/cuda/include -I../cudabyexample/common/ $(name).cu -o ./build/nvcc_$(name).o
	g++ -DDEBUG -o ./build/$(name).out ./build/nvcc_$(name).o -L/usr/local/cuda/lib64 -lcusparse -lcudart
	

	echo "\nCompiled\n -- Profiling --"
	./build/$(name).out


name = parallelreduction

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	nvcc -c 	-DNAIVE_REDUCE -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../cudabyexamples/common/
	g++ 		-o ./build/$(name).out 		./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64
	
	nvprof ./build/$(name).out

	nvcc -c 	-DFIRST_ATTEMPT_REDUCE -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../cudabyexamples/common/
	g++ 		-o ./build/$(name).out 		./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64

	nvprof ./build/$(name).out
	
	nvcc -c 	-DSECOND_REDUCE -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../cudabyexamples/common/
	g++ 		-o ./build/$(name).out 		./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64

	nvprof ./build/$(name).out

	nvcc -c 	-DTHIRD_REDUCE -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../cudabyexamples/common/
	g++ 		-o ./build/$(name).out 		./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64

	nvprof ./build/$(name).out
	
	nvcc -c 	-DSPEED_REDUCE -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../cudabyexamples/common/
	g++ 		-o ./build/$(name).out 		./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64

	nvprof ./build/$(name).out
vectorsum: 
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build

	nvcc $@.cu  -o ./build/$@.out
	./build/$@.out
	nvprof ./build/$@.out
name = for_each

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	# from cuda documentation
	nvcc -c -I/usr/local/cuda/include $(name).cu -o ./build/nvcc_$(name).o
	g++ -o ./build/$(name).out ./build/nvcc_$(name).o -L/usr/local/cuda/lib64 -lcusparse -lcudart

	echo "\nCompiled\n -- Starting program --"
	./build/*.out

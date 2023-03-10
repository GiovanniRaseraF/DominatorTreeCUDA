name = memprofile

all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	for N in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288; do\
		nvcc -c 	-DN=$$N -o ./build/nvcc_$(name).o 	 	$(name).cu 				-I/usr/local/cuda/include -I../common/; \
		g++ 		-DN=$$N -o ./build/$(name)_$$N.out 	./build/nvcc_$(name).o 	-lcusparse -lcudart -L/usr/local/cuda/lib64; \
		nvprof --csv ./build/$(name)_$$N.out; \
	done

graph:
	if [ -d "./build" ]; then echo "build exists"; else mkdir build; fi

	open ./plotresults.png &
	
	g++ dataparse.cpp -o ./build/a.out && cat profiled.md | ./build/a.out  > results.md

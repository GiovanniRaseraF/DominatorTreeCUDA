# Author: Giovanni Rasera
gpu:
	@ nvcc -c -I/usr/local/cuda/include $@.cu -std=c++17 -o ./build/nvcc_$@.o
	@ g++ -o  ./build/$@_parallel.out ./build/nvcc_$@.o -std=c++23 -L/usr/local/cuda/lib64 -lcusparse -lcudart

test:
	@ make gpu
	@ ./build/gpu_parallel.out ./graphs/bigone.txt 1 50

p:
	git add . && git commit -m "automatic: data exchange" && git push
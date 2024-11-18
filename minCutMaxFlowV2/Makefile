# Author: Giovanni Rasera
gpu:
	@ nvcc -c -I/usr/local/cuda/include $@.cu -std=c++17 -DBLS=2 -DTHS=512 -o ./build/nvcc_$@.o
	@ g++ -o  ./build/$@_parallel.out ./build/nvcc_$@.o -std=c++23 -DBLS=2 -DTHS=512 -L/usr/local/cuda/lib64 -lcusparse -lcudart

custom_params:
	@ nvcc -c -I/usr/local/cuda/include gpu.cu -std=c++17 -DBLS=${BLS} -DTHS=${THS} -o ./build/nvcc_gpu.o
	@ g++ -o  ./build/gpu_parallel.out ./build/nvcc_gpu.o -std=c++23 -DBLS=${BLS} -DTHS=${THS} -L/usr/local/cuda/lib64 -lcusparse -lcudart

test:
	# BLS=2 THS=512
	@ make gpu
	@ ./build/gpu_parallel.out ./graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ./graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ./graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ./graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ./graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ./graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ./graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ./graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ./graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ./graphs/bigone.txt   1 32688

	# BLS=4 THS=256
	@ make custom_params BLS=4 THS=256
	@ ./build/gpu_parallel.out ./graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ./graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ./graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ./graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ./graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ./graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ./graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ./graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ./graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ./graphs/bigone.txt   1 32688

create:
	@ python3 graph_generator.py --nodes    70 --edges    6000 --output ./graphs/basic1.txt --weight-range 1
	@ python3 graph_generator.py --nodes   130 --edges   10000 --output ./graphs/basic2.txt --weight-range 1
	@ python3 graph_generator.py --nodes   260 --edges   20000 --output ./graphs/basic3.txt --weight-range 1
	@ python3 graph_generator.py --nodes   520 --edges   50000 --output ./graphs/basic4.txt --weight-range 1
	@ python3 graph_generator.py --nodes  1030 --edges  100000 --output ./graphs/basic5.txt --weight-range 1
	@ python3 graph_generator.py --nodes  2050 --edges  200000 --output ./graphs/nanoone.txt --weight-range 1
	@ python3 graph_generator.py --nodes  4100 --edges  400000 --output ./graphs/microone.txt --weight-range 1
	@ python3 graph_generator.py --nodes  8200 --edges  800000 --output ./graphs/minione.txt --weight-range 1
	@ python3 graph_generator.py --nodes 16400 --edges 1600000 --output ./graphs/midone.txt --weight-range 1
	@ python3 graph_generator.py --nodes 32800 --edges 3200000 --output ./graphs/bigone.txt --weight-range 1

p:
	git add . && git commit -m "automatic: data exchange" && git push
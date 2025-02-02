# Author: Giovanni Rasera
gpu:
	@ nvcc -c -I/usr/local/cuda/include $@.cu -std=c++17 -DBLS=2 -DTHS=512 -o ./build/nvcc_$@.o
	@ g++ -o  ./build/$@_parallel.out ./build/nvcc_$@.o -std=c++23 -DBLS=2 -DTHS=512 -L/usr/local/cuda/lib64 -lcusparse -lcudart

custom_params:
	@ nvcc -c -I/usr/local/cuda/include gpu.cu -std=c++17 -DBLS=${BLS} -DTHS=${THS} -o ./build/nvcc_gpu.o
	@ g++ -o  ./build/gpu_parallel.out ./build/nvcc_gpu.o -std=c++23 -DBLS=${BLS} -DTHS=${THS} -L/usr/local/cuda/lib64 -lcusparse -lcudart

custom_params_node_cut:
	@ nvcc -c -I/usr/local/cuda/include gpu.cu -std=c++17 -DBLS=${BLS} -DTHS=${THS} -DNODECUT -o ./build/nvcc_gpu.o
	@ g++ -o  ./build/gpu_parallel.out ./build/nvcc_gpu.o -std=c++23 -DBLS=${BLS} -DTHS=${THS} -DNODECUT -L/usr/local/cuda/lib64 -lcusparse -lcudart

custom_params_node_cut_print:
	@ nvcc -c -I/usr/local/cuda/include gpu.cu -std=c++17 -DBLS=${BLS} -DTHS=${THS} -DNODECUT -DPRINTNODECUT -o ./build/nvcc_gpu.o
	@ g++ -o  ./build/gpu_parallel.out ./build/nvcc_gpu.o -std=c++23 -DBLS=${BLS} -DTHS=${THS} -DNODECUT -DPRINTNODECUT -L/usr/local/cuda/lib64 -lcusparse -lcudart

test:
	@ make custom_params_node_cut_print BLS=1 THS=1024
	@ ./build/gpu_parallel.out ../graphs/example3.txt  0 6

	# BLS=1 THS=1024
	@ make custom_params BLS=1 THS=1024
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs/bigone.txt   1 32688

	# BLS=2 THS=512
	@ make custom_params BLS=2 THS=512
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs/bigone.txt   1 32688

	# BLS=4 THS=256
	@ make custom_params BLS=4 THS=256
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs/bigone.txt   1 32688

	# BLS=8 THS=128
	@ make custom_params BLS=8 THS=128
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs/bigone.txt   1 32688

	# BLS=16 THS=64
	@ make custom_params BLS=16 THS=64
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs/bigone.txt   1 32688

test_2:
	# BLS=1 THS=1024
	@ make custom_params BLS=1 THS=1024
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs2/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs2/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs2/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs2/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs2/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs2/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs2/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs2/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs2/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs2/bigone.txt   1 32688

test_seed_1:
	# BLS=1 THS=1024
	@ make custom_params BLS=1 THS=1024
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs_seed_1/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs_seed_1/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs_seed_1/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs_seed_1/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs_seed_1/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs_seed_1/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs_seed_1/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs_seed_1/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs_seed_1/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs_seed_1/bigone.txt   1 32688

test_seed_max_2:
	# BLS=1 THS=1024
	@ make custom_params BLS=1 THS=1024
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs_seed_max_2/bigone.txt   1 32688

test_seed_max_3:
	# BLS=1 THS=1024
	@ make custom_params BLS=1 THS=1024
    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/basic1.txt   1 64
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/basic2.txt   1 128
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/basic3.txt   1 256
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/basic4.txt   1 512
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/basic5.txt   1 1024
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/nanoone.txt  1 2048
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/microone.txt  1 4096
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/minione.txt  1 8172
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/midone.txt   1 16344
	@ ./build/gpu_parallel.out ../graphs_seed_max_3/bigone.txt   1 32688

create_min:
	@ python3 graph_generator.py --nodes    70 --edges    600 --output ../graphs2/basic1.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   130 --edges   1000 --output ../graphs2/basic2.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   260 --edges   2000 --output ../graphs2/basic3.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   520 --edges   5000 --output ../graphs2/basic4.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  1030 --edges  10000 --output ../graphs2/basic5.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  2050 --edges  20000 --output ../graphs2/nanoone.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  4100 --edges  40000 --output ../graphs2/microone.txt  --weight-range 1
	@ python3 graph_generator.py --nodes  8200 --edges  80000 --output ../graphs2/minione.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes 16400 --edges 160000 --output ../graphs2/midone.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes 32800 --edges 320000 --output ../graphs2/bigone.txt 	--weight-range 1

create_min:
	@ python3 graph_generator.py --nodes    70 --edges    600 --output ../graphs2/basic1.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   130 --edges   1000 --output ../graphs2/basic2.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   260 --edges   2000 --output ../graphs2/basic3.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes   520 --edges   5000 --output ../graphs2/basic4.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  1030 --edges  10000 --output ../graphs2/basic5.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  2050 --edges  20000 --output ../graphs2/nanoone.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes  4100 --edges  40000 --output ../graphs2/microone.txt  --weight-range 1
	@ python3 graph_generator.py --nodes  8200 --edges  80000 --output ../graphs2/minione.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes 16400 --edges 160000 --output ../graphs2/midone.txt 	--weight-range 1
	@ python3 graph_generator.py --nodes 32800 --edges 320000 --output ../graphs2/bigone.txt 	--weight-range 1

create_seed_1:
	@ python3 graph_generator.py --seed 1 --nodes    70 --edges    600 --output ../graphs_seed_1/basic1.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes   130 --edges   1000 --output ../graphs_seed_1/basic2.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes   260 --edges   2000 --output ../graphs_seed_1/basic3.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes   520 --edges   5000 --output ../graphs_seed_1/basic4.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes  1030 --edges  10000 --output ../graphs_seed_1/basic5.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes  2050 --edges  20000 --output ../graphs_seed_1/nanoone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes  4100 --edges  40000 --output ../graphs_seed_1/microone.txt   --weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes  8200 --edges  80000 --output ../graphs_seed_1/minione.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes 16400 --edges 160000 --output ../graphs_seed_1/midone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 1 --nodes 32800 --edges 320000 --output ../graphs_seed_1/bigone.txt 	--weight-range 1

create_seed_max_2:
	@ python3 graph_generator.py --seed 2 --nodes    70 --edges    6000 --output ../graphs_seed_max_2/basic1.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes   130 --edges   10000 --output ../graphs_seed_max_2/basic2.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes   260 --edges   20000 --output ../graphs_seed_max_2/basic3.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes   520 --edges   50000 --output ../graphs_seed_max_2/basic4.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes  1030 --edges  100000 --output ../graphs_seed_max_2/basic5.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes  2050 --edges  200000 --output ../graphs_seed_max_2/nanoone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes  4100 --edges  400000 --output ../graphs_seed_max_2/microone.txt   --weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes  8200 --edges  800000 --output ../graphs_seed_max_2/minione.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes 16400 --edges 1600000 --output ../graphs_seed_max_2/midone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 2 --nodes 32800 --edges 3200000 --output ../graphs_seed_max_2/bigone.txt 	--weight-range 1

create_seed_max_3:
	@ python3 graph_generator.py --seed 3 --nodes    70 --edges    6000 --output ../graphs_seed_max_3/basic1.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes   130 --edges   10000 --output ../graphs_seed_max_3/basic2.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes   260 --edges   20000 --output ../graphs_seed_max_3/basic3.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes   520 --edges   50000 --output ../graphs_seed_max_3/basic4.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes  1030 --edges  100000 --output ../graphs_seed_max_3/basic5.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes  2050 --edges  200000 --output ../graphs_seed_max_3/nanoone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes  4100 --edges  400000 --output ../graphs_seed_max_3/microone.txt   --weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes  8200 --edges  800000 --output ../graphs_seed_max_3/minione.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes 16400 --edges 1600000 --output ../graphs_seed_max_3/midone.txt 	--weight-range 1
	@ python3 graph_generator.py --seed 3 --nodes 32800 --edges 3200000 --output ../graphs_seed_max_3/bigone.txt 	--weight-range 1

p:
	git add . && git commit -m "automatic: data exchange" && git push
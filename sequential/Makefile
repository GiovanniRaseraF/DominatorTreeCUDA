# Author: Giovanni Rasera
cpu:
	#if [ -d "./build" ]; then rm -r build; fi
	#mkdir build
	g++ -o ./build/$@_sequential.out ./$@.cpp -std=c++23

test:
	@ make 
	@ ./build/cpu_sequential.out ../graphs/example3.txt  0 6

    # nanos, micros, millis, V, E, source, sink, maxFlow
	@ ./build/cpu_sequential.out ../graphs/basic1.txt   1 64
	@ ./build/cpu_sequential.out ../graphs/basic2.txt   1 128
	@ ./build/cpu_sequential.out ../graphs/basic3.txt   1 256
	@ ./build/cpu_sequential.out ../graphs/basic4.txt   1 512
	@ ./build/cpu_sequential.out ../graphs/basic5.txt   1 1024
	@ ./build/cpu_sequential.out ../graphs/nanoone.txt  1 2048
	@ ./build/cpu_sequential.out ../graphs/microone.txt  1 4096
	@ ./build/cpu_sequential.out ../graphs/minione.txt  1 8172
	@ ./build/cpu_sequential.out ../graphs/midone.txt   1 16344
	@ ./build/cpu_sequential.out ../graphs/bigone.txt   1 32688

p:
	git add . && git commit -m "automatic: data exchange" && git push
# Author: Giovanni Rasera

main:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	g++ -o ./build/$@.out ./$@.cpp -std=c++17

	./build/$@.out

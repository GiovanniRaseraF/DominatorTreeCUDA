# Author: Giovanni Rasera

main:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	g++ -o ./build/$@.out ./$@.cpp -std=c++17

	./build/$@.out

cpu: 
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	g++ -o ./build/$@.out ./$@.cpp -std=c++17 -D$(STRUCT) -DV=$(V)

	./build/$@.out

test:
#make cpu STRUCT=USEARRAY V=1024
	make cpu STRUCT=USEVECTOR V=1024
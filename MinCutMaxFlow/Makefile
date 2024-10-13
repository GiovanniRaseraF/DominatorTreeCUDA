# Author: Giovanni Rasera
V=1024

main:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build
	
	g++ -o ./build/$@.out ./$@.cpp -DV=$V -std=c++17

	./build/$@.out

test:
	make main STRUCT=USEVECTOR V=1024
	make main STRUCT=USEVECTOR V=2024
	make main STRUCT=USEVECTOR V=4024
	make main STRUCT=USEVECTOR V=8024
	make main STRUCT=USEVECTOR V=16024
	make main STRUCT=USEVECTOR V=32024
all:
	if [ -d "./build" ]; then rm -r build; fi
	mkdir build && cd build && cmake .. && make	
	echo "\nCompiled\n -- Starting program --"
	./build/*.out
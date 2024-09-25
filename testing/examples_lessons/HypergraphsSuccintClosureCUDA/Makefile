generator:
	if [ -d "./eseguibile" ]; then rm -r eseguibile; fi
	mkdir eseguibile 

	nvcc ./generator.cpp -o ./eseguibile/generator.out
	nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=1 -D MAX_BLOCKS_AI=1 -D NTHR=1 -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 ./progetto_G.cu -D HIDE -o ./eseguibile/progetto_1_1_16.out
	nvcc -rdc=true -D FILE_OUT -D DEBUG  -D CPU -D NTHR=1 -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 ./progetto_G.cu -D HIDE -o ./eseguibile/progetto_16_CPU.out
	nvcc -rdc=true -D FILE_OUT -D DEBUG  -D DYNAMIC -D NTHR=8 -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 ./progetto_G.cu -D HIDE -o ./eseguibile/progetto_16_DYNO.out
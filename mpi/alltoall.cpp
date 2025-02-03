#include <mpi.h>
#include <iostream>

int main(){
    MPI_Init(NULL, NULL);

    int size; 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 3){
        std::cout << "Devi lanciare 3 thread !!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int myvalues[3];
    for(int i = 0; i < 3; i++){
        myvalues[i] = myrank * 300 + i *100;
    }

    std::cout << "Processo: " << myrank << " Valori: " << 
    myvalues[0] << " " << myvalues[1] << " " << myvalues[2] << std::endl;

    int bufferrecv[3];

    MPI_Alltoall(
        &myvalues, 1, MPI_INT, 
        bufferrecv, 1, MPI_INT, MPI_COMM_WORLD
    );

    std::cout << "Valori dal processo " << myrank << " Valori: " << 
    bufferrecv[0] << " " << bufferrecv[1] << " " << bufferrecv[2] << std::endl;

    MPI_Finalize();
}
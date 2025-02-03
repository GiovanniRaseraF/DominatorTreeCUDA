#include <mpi.h>
#include <iostream>

int main(int argc, char **argv){
    MPI_Init(NULL, NULL);
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int number;

    if(rank == 0){
        number = -123;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }else if(rank == 1){
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Ho ricevuto: " << number << std::endl;
    }else{
        std::cout << "Not a validi thread: " << rank << std::endl;
    }

    MPI_Finalize();
}
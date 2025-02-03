#include <mpi.h>
#include <iostream>

int main(int argc, char **argv){
    MPI_Init(NULL, NULL);
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int a;
    int b;

    if(rank == 0){
        MPI_Send(&a, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Send(&b, 1, MPI_INT, 1, 2, MPI_COMM_WORLD);
    }else if(rank == 1){
        MPI_Recv(&b, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&a, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }else{
        std::cout << "Not a validi thread: " << rank << std::endl;
    }

    MPI_Finalize();
}
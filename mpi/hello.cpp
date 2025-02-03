#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]){
    int tutti, myrank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &tutti);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::cout << "Hello dal thread: " << myrank << " / " << tutti << std::endl;

    MPI_Finalize();
}
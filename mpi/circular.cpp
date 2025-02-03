#include <mpi.h>
#include <iostream>

int main(int argc, char **argv){
    MPI_Init(NULL, NULL);
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int a[10], b[10], P, r;

    if(rank % 2 == 0){
        MPI_Send(a, 10, MPI_INT, (r+1) %P, 1, MPI_COMM_WORLD);
        MPI_Recv(b, 10, MPI_INT, (r-1+P) %P, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }else{
        MPI_Recv(b, 10, MPI_INT, (r+1) %P, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(a, 10, MPI_INT, (r-1+P) %P, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
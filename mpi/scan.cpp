#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int total;
    // somma i valodi dei precedenti
    MPI_Scan(&myrank, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "MPI process " << myrank << " Totale=" << total << std::endl;

    int myranks[3], totals[3];
    myranks[0] = myrank;
    myranks[1] = 10*myrank;
    myranks[2] = myrank+10;


    MPI_Scan(myranks, totals, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "MPI process " << myranks[0] << " Totale=" << totals[0] 
    << " " << totals[1] << " " << totals[2] << std::endl;

    MPI_Finalize();

    return 0;
}
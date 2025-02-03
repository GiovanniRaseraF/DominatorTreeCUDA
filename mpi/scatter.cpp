#include <mpi.h>
#include <iostream>

int main(){
    MPI_Init(NULL, NULL);
    int root = 0;

    MPI_Comm comm = MPI_COMM_WORLD;
    int gsize, *sendarray;
    int myrank, rbuf[10]; 

    // settaggio delle agrandezza
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &myrank);

    if(myrank == root){
        MPI_Comm_size(comm, &gsize);

        // la root alloca il buffer necessario
        sendarray = (int *)malloc(gsize*10*sizeof(int));

        for(int s = 0; s < gsize; s++){
            for(int i = 0; i < 10; i++){
                sendarray[s*10+i] = s*10+i;
            }
        }

        std::cout << "To Send: " << std::endl;
        for(int s = 0; s < gsize; s++){
            for(int i = 0; i < 10; i++){
                std::cout << sendarray[s*10+i] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Questo comando invia al root tutto il buffer da tutti i processi
    MPI_Scatter(sendarray, 10, MPI_INT, rbuf, 10, MPI_INT, root, comm);

    std::cout << "Rank: " << myrank << " -> ";
    for(int i = 0; i < 10; i++){
        std::cout << rbuf[i] << " ";
    }
    std::cout << std::endl;
    
    MPI_Finalize();
}


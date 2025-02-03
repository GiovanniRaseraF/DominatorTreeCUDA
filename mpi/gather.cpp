#include <mpi.h>
#include <iostream>

int main(){
    MPI_Init(NULL, NULL);
    int root = 0;

    MPI_Comm comm = MPI_COMM_WORLD;
    int gsize, sendarray[10];
    int myrank, 
    *rbuf; // questo è il receiving buffer

    // settaggio delle agrandezza
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &myrank);

    // init del send array
    for(int i = 0; i < 10; i++){
        sendarray[i] = myrank+i;
    }

    if(myrank == root){
        MPI_Comm_size(comm, &gsize);

        // la root alloca il buffer necessario
        rbuf = (int *)malloc(gsize*10*sizeof(int));
    }

    // Questo comando invia al root tutto il buffer da tutti i processi
    MPI_Gather(sendarray, 10, MPI_INT, rbuf, 10, MPI_INT, root, comm);
    
    if(myrank == root){
        for(int s = 0; s < gsize; s++){
            for(int i = 0; i < 10; i++){
                std::cout << rbuf[s*10+i] << " ";
            }
            std::cout << std::endl;
        }
    }

    
    MPI_Finalize();
}


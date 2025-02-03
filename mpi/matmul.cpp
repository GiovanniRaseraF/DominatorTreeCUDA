#include <mpi.h>
#include <iostream>

int main(){
    MPI_Init(NULL, NULL);
    int root = 0;

    MPI_Comm comm = MPI_COMM_WORLD;
    int gsize, *A;
    int myrank;

    // settaggio delle agrandezza
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &myrank);

    int Aparz[gsize]; // A parziale
    int b[gsize] = {1, 1, 1}; // vettore b
    int *x; // risultato

    if(myrank == root){
        MPI_Comm_size(comm, &gsize);

        // la root alloca il buffer necessario
        A = (int *)malloc(gsize*gsize*sizeof(int));
        x = (int *)malloc(gsize*sizeof(int));

        for(int s = 0; s < gsize; s++){
            for(int i = 0; i < gsize; i++){
                A[s*gsize+i] = s*gsize+i;
            }
        }

        for(int i = 0; i < gsize; i++){
            x[i] = 0;
        }
    }

    // Questo comando invia al root tutto il buffer da tutti i processi
    MPI_Scatter(A, gsize, MPI_INT, Aparz, gsize, MPI_INT, root, comm);

    // A questo punto ho deiviso i dati
    int xparz[1] = {0};
    for(int i = 0; i < gsize; i++){
        xparz[0] += (Aparz[i] * b[i]);
    }
    
    std::cout << "id " << myrank << " xparz: " << xparz[0] << std::endl;

    MPI_Gather(xparz, 1, MPI_INT, x, 1, MPI_INT, root, comm);

    std::cout << "id: " << myrank << " Aparz -> ";
    for(int i = 0; i < gsize; i++){
        std::cout << Aparz[i] << " ";
    }
    std::cout << std::endl;

    if(myrank == root){
        std::cout << "b -> ";
        for(int i = 0; i < gsize; i++){
            std::cout << b[i] << " ";
        }

        std::cout << "x result -> ";
        for(int i = 0; i < gsize; i++){
            std::cout << x[i] << " ";
        }
    }
    
    MPI_Finalize();
}


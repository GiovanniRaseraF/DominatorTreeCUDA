#include <mpi.h>
#include <iostream>

int main(){
    MPI_Init(NULL, NULL);
    int root = 0;

    double ain[30] , aout[30];
    int ind[30];

    struct {
        double val;
        int rank;
    } in[30], out[30];

    // Creazione del rank
    int myrank = -1;
    int i;
    auto comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank);

    // Predispogno i dati
    for(i = 0; i < 30; ++i){
        in[i].val = (i + myrank) % 6;//ain[i];
        in[i].rank = myrank;
    }

    if(myrank != root){
        // Print val pre thread
        std::cout << "val: ";
        for(int i = 0; i < 30; ++i){
            std::cout << in[i].val << " ";
        }
        std::cout << std::endl; 
        std::cout << "rak: ";
        for(int i = 0; i < 30; ++i){
            std::cout << in[i].rank << " ";
        }
        std::cout << std::endl; 
    }

    // fa una riduzione con tutti come target
    MPI_Allreduce(in, out, 30, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

    std::cout << "Rank: " << myrank << std::endl;
    std::cout << "Result for reduce: " << std::endl;
    std::cout << "out: ";
    for(int i = 0; i < 30; ++i){
        std::cout << out[i].val << " ";
    }
    std::cout << std::endl; 
    std::cout << "loc: ";
    for(int i = 0; i < 30; ++i){
        std::cout << out[i].rank << " ";
    }
    std::cout << std::endl; 

    MPI_Finalize();
}
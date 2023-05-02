#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ack = 1;

    for (int j = 0; j < 20; j++)
    {
        for (int i = 0; i < size; i++)
        {
            if(rank == i){
                printf("I am procces %d\n", rank);
                MPI_Send(&ack, 1, MPI_INT, (i+1) % size, i, MPI_COMM_WORLD);
            }
            else if(rank == (i+1) % size){
                MPI_Status status;
                MPI_Recv(&ack, 1, MPI_INT, i % size, i, MPI_COMM_WORLD, &status);
                printf("I am procces %d, ack = %d\n", rank, ack);
            }
            ack += 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
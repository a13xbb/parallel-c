#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../utils.h"

int main(int argc, char* argV[]) {
    int n_rows, n_cols;
    n_rows = strtol(argV[1], NULL, 10);
    n_cols = strtol(argV[2], NULL, 10);
    int my_rank, thread_cnt;
    MPI_Init(&argc, &argV);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    double sq = sqrt(thread_cnt);
    if (sq != (int)sq) {
        if (my_rank == 0) {
            printf("Threads amount is not a square\n\n");
        }
        return -1;
    }

    int blocks_per_dim = (int)(sq);
    if (n_rows != n_cols || n_rows % blocks_per_dim != 0) {
        if (my_rank == 0) {
            printf("Matrix size is not fully divided by sqrt of num threads\n\n");
        }
        return -1;
    }

    int *matrix = (int*)malloc(n_rows * n_cols * sizeof(int));
    int *vector = (int*)malloc(n_cols * sizeof(int));
    int *product = (int*)malloc(n_rows * sizeof(int));
    int *local_product = (int*)malloc(n_rows * sizeof(int));

    initialize_matrix(matrix, n_rows, n_cols);
    initialize_vector(vector, n_cols);
    initialize_vector_with_zeros(product, n_rows);
    initialize_vector_with_zeros(local_product, n_rows);

    int block_sz = n_cols / blocks_per_dim;

    double local_time_start = MPI_Wtime();

    int row_start = (my_rank / blocks_per_dim) * block_sz;
    int row_end  = row_start + block_sz;
    int col_start = (my_rank % blocks_per_dim) * block_sz;
    int col_end = col_start + block_sz;

    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            local_product[i] += matrix[i * n_cols + j] * vector[j];
        }
    }

    MPI_Reduce(local_product, product, n_rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_time_finish = MPI_Wtime();
    double local_time_elapsed = local_time_finish - local_time_start;
    double time_elapsed;
    MPI_Reduce(&local_time_elapsed, &time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //OUTPUT
    // if (my_rank == 0) {
    //     printf("Matrix:\n");
    //     print_matrix(matrix, n_rows, n_cols);

    //     printf("Vector:\n");
    //     print_vector(vector, n_cols);

    //     printf("Product:\n");
    //     print_vector(product, n_rows);
    // }

    if (my_rank == 0) {
        printf("Blocks parallelism on %d threads runtime:\n %lf\n", thread_cnt, time_elapsed);
    }

    free(matrix);
    free(vector);
    free(product);
    free(local_product);

    MPI_Finalize();

    return 0;
}
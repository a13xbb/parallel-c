#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argV[]) {
    int n_rows, n_cols;
    n_rows = strtol(argV[1], NULL, 10);
    n_cols = strtol(argV[2], NULL, 10);
    int my_rank, thread_cnt, local_n_cols;
    MPI_Init(&argc, &argV);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int *matrix, *vector, *product, *local_product;

    vector = (int*)malloc(n_cols * sizeof(int));
    matrix = (int*)malloc(n_rows * n_cols * sizeof(int));
    product = (int*)malloc(n_rows * sizeof(int));
    local_product = (int*)malloc(n_rows * sizeof(int));
    
    for (int i = 0; i < n_rows; i++) {
        local_product[i] = 0;
        product[i] = 0;
        for (int j = 0; j < n_cols; j++) {
            matrix[i * n_cols + j] = rand() % 10;
        }
    }
    for (int i = 0; i < n_cols; i++) {
        vector[i] = rand() % 10;
    }

    int remainder = n_cols % thread_cnt;
    int *colcounts, *displs;
    int displs_sum = 0;
    colcounts = (int*)malloc(thread_cnt * sizeof(int));
    displs = (int*)malloc(thread_cnt * sizeof(int));
    for (int rank = 0; rank < thread_cnt; rank++) {
        if (remainder == 0 || rank >= remainder) {
            colcounts[rank] = (n_cols / thread_cnt);
        } else if (rank < remainder) {
            colcounts[rank] = (n_cols / thread_cnt + 1);
        }
        displs[rank] = displs_sum;
        displs_sum += colcounts[rank];
    }

    //------------------OUTPUT--------------------
    // if (my_rank == 0) {
    //     printf("Matrix:\n");
    //     for (int i = 0; i < n_rows; i++) {
    //         for (int j = 0; j < n_cols; j++) {
    //             printf("%d ", matrix[i * n_cols + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     printf("Vector:\n");
    //     for (int i = 0; i < n_cols; i++) {
    //         printf("%d ", vector[i]);
    //     }
    //     printf("\n\n");
    // }
    

    double local_time_start = MPI_Wtime();

    int start = displs[my_rank];
    int end = start + colcounts[my_rank];

    for (int col = start; col < end; col++) {
        for (int row = 0; row < n_rows; row++) {
            local_product[row] += matrix[row * n_cols + col] * vector[col];
        }
    }
      
    MPI_Reduce(local_product, product, n_rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double local_time_finish = MPI_Wtime();
    double local_time_elapsed = local_time_finish - local_time_start;
    double time_elapsed;
    MPI_Reduce(&local_time_elapsed, &time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);



    
    //------------------OUTPUT-----------------
    // if (my_rank == 0) {
    //     printf("Product:\n");
    //     for (int i = 0; i < n_rows; i++) {
    //         printf("%d ", product[i]);
    //     }
    //     printf("\n");
    // }

    MPI_Finalize();


    if (my_rank == 0) {
        printf("Cols parallelism on %d threads runtime = %lf\n", thread_cnt, time_elapsed);
    }

    return 0;
}
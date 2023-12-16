#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../utils.h"

int main(int argc, char* argV[]) {
    int n_rows, n_cols;
    n_rows = strtol(argV[1], NULL, 10);
    n_cols = strtol(argV[2], NULL, 10);
    int my_rank, thread_cnt, local_n_rows;
    MPI_Init(&argc, &argV);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int *matrix, *vector, *local_rows, *product, *local_product;
    int *sendcounts, *send_displs, *recvcounts, *recv_displs;

    sendcounts = (int*)malloc(thread_cnt * sizeof(int));
    recvcounts = (int*)malloc(thread_cnt * sizeof(int));
    send_displs = (int*)malloc(thread_cnt * sizeof(int));
    recv_displs = (int*)malloc(thread_cnt * sizeof(int));
    int remainder = n_rows % thread_cnt;
    int send_sum = 0, recv_sum = 0;
    for (int rank = 0; rank < thread_cnt; rank++) {
        if (remainder == 0 || rank >= remainder) {
            sendcounts[rank] = (n_rows / thread_cnt) * n_cols;
            recvcounts[rank] = n_rows / thread_cnt;
        } else if (rank < remainder) {
            sendcounts[rank] = (n_rows / thread_cnt + 1) * n_cols;
            recvcounts[rank] = n_rows / thread_cnt + 1;
        }
        send_displs[rank] = send_sum;
        send_sum += sendcounts[rank];
        recv_displs[rank] = recv_sum;
        recv_sum += recvcounts[rank];
    }

    if (remainder == 0 || my_rank >= remainder) {
        local_n_rows = n_rows / thread_cnt;
    } else if (my_rank < remainder) {
        local_n_rows = n_rows / thread_cnt + 1;
    }

    vector = (int*)malloc(n_cols * sizeof(int));
    matrix = (int*)malloc(n_rows * n_cols * sizeof(int));
    local_rows = (int*)malloc(local_n_rows * n_cols * sizeof(int));
    product = (int*)malloc(n_rows * sizeof(int));
    local_product = (int*)malloc(local_n_rows * sizeof(int));

    
    if (my_rank == 0) {
        initialize_matrix(matrix, n_rows, n_cols);
        initialize_vector(vector, n_cols);
    }

    MPI_Bcast(vector, n_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(matrix, sendcounts, send_displs, MPI_INT, local_rows, local_n_rows * n_cols, MPI_INT, 0, MPI_COMM_WORLD);

    double local_time_start = MPI_Wtime();

    for (int i = 0; i < local_n_rows; i++) {
        local_product[i] = 0;
        for (int j = 0; j < n_cols; j++) {
            local_product[i] += local_rows[i * n_cols + j] * vector[j];
        }
    }

    MPI_Gatherv(local_product, local_n_rows, MPI_INT, product, recvcounts, recv_displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    double local_time_finish = MPI_Wtime();
    double local_time_elapsed = local_time_finish - local_time_start;
    double time_elapsed;
    MPI_Reduce(&local_time_elapsed, &time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //------------------OUTPUT-----------------
    // if (my_rank == 0) {
    //     printf("Matrix:\n");
    //     print_matrix(matrix, n_rows, n_cols);

    //     printf("Vector:\n");
    //     print_vector(vector, n_cols);
        
    //     printf("Product:\n");
    //     print_vector(product, n_rows);
    // }

    if (my_rank == 0) {
        printf("Rows parallelism on %d threads runtime = %lf\n", thread_cnt, time_elapsed);
    }

    free(matrix);
    free(vector);
    free(local_rows);
    free(product);
    free(local_product);
    free(sendcounts);
    free(send_displs);
    free(recvcounts);
    free(recv_displs);

    MPI_Finalize();

    return 0;
}
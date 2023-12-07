#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int n_rows = 6, n_cols = 4;
    int my_rank, thread_cnt, local_n_rows;
    MPI_Init(&argc, &argv);
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

    // printf("Sendcounts:\n");
    // for (int i = 0; i < thread_cnt; i++) {
    //     printf("%d ", sendcounts[i]);
    // }
    // printf("\n");

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
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                matrix[i * n_cols + j] = rand() % 10;
            }
        }
        for (int i = 0; i < n_cols; i++) {
            vector[i] = rand() % 10;
        }

        printf("Matrix:\n");
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                printf("%d ", matrix[i * n_cols + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("Vector:\n");
        for (int i = 0; i < n_cols; i++) {
            printf("%d ", vector[i]);
        }
        printf("\n\n");
    }

    MPI_Bcast(vector, n_cols, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Scatter(matrix, local_n_rows * n_cols, MPI_INT, local_rows, local_n_rows * n_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(matrix, sendcounts, send_displs, MPI_INT, local_rows, local_n_rows * n_cols, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n_rows; i++) {
        local_product[i] = 0;
        for (int j = 0; j < n_cols; j++) {
            local_product[i] += local_rows[i * n_cols + j] * vector[j];
        }
    }

    //DEBUG
    // printf("Local submatrix on thread %d:\n", my_rank);
    // for (int i = 0; i < local_n_rows; i++) {
    //     for (int j = 0; j < n_cols; j++) {
    //         printf("%d ", local_rows[i * n_cols + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    //DEBUG
    // printf("Local product on thread %d:\n", my_rank);
    // for (int i = 0; i < local_n_rows; i++) {
    //     printf("%d ", local_product[i]);
    // }
    // printf("\n\n");

    // MPI_Gather(local_product, local_n_rows, MPI_INT, product, local_n_rows, MPI_INT, 0, MPI_COMM_WORLD);

    //DEBUG
    // printf("recvcounts:\n");
    // for (int i = 0; i < thread_cnt; i++) {
    //     printf("%d ", recvcounts[i]);
    // }
    // printf("\n\n");

    MPI_Gatherv(local_product, local_n_rows, MPI_INT, product, recvcounts, recv_displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Product:\n");
        for (int i = 0; i < n_rows; i++) {
            printf("%d ", product[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
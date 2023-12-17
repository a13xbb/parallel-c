#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "../utils.h"

int thread_cnt;
int my_rank;
int grid_size;

void DataDistribution(int *a_matrix, int *b_matrix, int *a_block, int *b_block, int size,
                      int block_size)
{
    int grid_size = (int)sqrt((int)thread_cnt);
    if (my_rank == 0)
    {
        int row_start = my_rank / grid_size;
        int row_end = row_start + block_size;
        int col_start = my_rank % grid_size;
        int col_end = col_start + block_size;
        int tmp_idx = 0;
        for (int row = row_start; row < row_end; row++)
        {
            for (int col = col_start; col < col_end; col++)
            {
                a_block[tmp_idx] = a_matrix[row * size + col];
                b_block[tmp_idx] = b_matrix[row * size + col];
                tmp_idx++;
            }
        }

        for (int rank = 1; rank < thread_cnt; rank++)
        {
            int *tmp_a, *tmp_b;
            tmp_a = (int *)malloc(block_size * block_size * sizeof(int));
            tmp_b = (int *)malloc(block_size * block_size * sizeof(int));
            int row_start = (rank / grid_size) * block_size;
            int row_end = row_start + block_size;
            int col_start = (rank % grid_size) * block_size;
            int col_end = col_start + block_size;

            int tmp_idx = 0;
            for (int row = row_start; row < row_end; row++)
            {
                for (int col = col_start; col < col_end; col++)
                {
                    tmp_a[tmp_idx] = a_matrix[row * size + col];
                    tmp_b[tmp_idx] = b_matrix[row * size + col];
                    tmp_idx++;
                }
            }
            MPI_Send(tmp_a, block_size * block_size, MPI_INT, rank, 0, MPI_COMM_WORLD);
            MPI_Send(tmp_b, block_size * block_size, MPI_INT, rank, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(a_block, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_block, block_size * block_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void BlockMultiplication(int *a_block, int *b_block, int *c_block, int block_size)
{
    // Вычисление произведения матричных блоков
    for (int i = 0; i < block_size; i++)
    {
        for (int j = 0; j < block_size; j++)
        {
            {
                int temp = 0;
                for (int k = 0; k < block_size; k++)
                {
                    temp += a_block[i * block_size + k] * b_block[k * block_size + j];
                }
                c_block[i * block_size + j] += temp;
            }
        }
    }
}

void InitialSkew(int *a_block, int *b_block, int *c_block, int block_size)
{
    int my_i = (my_rank / grid_size);
    int my_j = (my_rank % grid_size);
    if (my_i != 0)
    { // i=1, j=0   a_step = 2   dest_proc = 1 * 3 + (0 + 2) % 3 = 5
        // i=1, j=1   a_step = 2   dest_proc = 1 * 3 + (1 + 2) % 3 = 3
        // i=2, j=0   a_step = 1   dest_proc = 1 * 3 + (0 + 2) % 3 = 5
        int a_step = grid_size - my_i;
        int dest_proc = my_i * grid_size + (my_j + a_step) % grid_size;
        // i=1, j=0   a_step = 2   dest_proc = 1 * 3 + (1) % 3 = 4
        // i=1, j=1   a_step = 2   dest_proc = 1 * 3 + (1 + 1) % 3 = 5
        int source_proc = my_i * grid_size + (my_j + (grid_size - a_step)) % grid_size;
        // printf("Process %d source_proc: %d, dest_proc: %d\n", my_rank, source_proc, dest_proc);
        int *a_block_copy;
        a_block_copy = copy_matrix(a_block, block_size, block_size);
        MPI_Sendrecv(a_block_copy, block_size * block_size, MPI_INT, dest_proc, 0,
                     a_block, block_size * block_size, MPI_INT, source_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (my_j != 0)
    {
        // i=1, j=1   a_step = 2   dest_proc = (1+2)%3 + 1 = 1
        // i=2, j=2   a_step = 1   dest_proc = (2+1)%3 * 3 + 2 = 2
        int b_step = grid_size - my_j;
        int dest_proc = ((my_i + b_step) % grid_size) * grid_size + my_j;
        // i=1, j=1   a_step = 2   source_proc = (1+1)%3 * 3 + 1 = 1
        // i=2, j=2   a_step = 2   source_proc = (2+2)%3 * 3 + 2 = 5
        int source_proc = ((my_i + (grid_size - b_step)) % grid_size) * grid_size + my_j;
        // printf("Process %d source_proc: %d, dest_proc: %d\n", my_rank, source_proc, dest_proc);
        int *b_block_copy;
        b_block_copy = copy_matrix(b_block, block_size, block_size);
        MPI_Sendrecv(b_block_copy, block_size * block_size, MPI_INT, dest_proc, 1,
                     b_block, block_size * block_size, MPI_INT, source_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    BlockMultiplication(a_block, b_block, c_block, block_size);
}

void ShiftAndMultiply(int *a_block, int *b_block, int *c_block, int block_size)
{
    int my_i = (my_rank / grid_size);
    int my_j = (my_rank % grid_size);

    for (int i = 0; i < grid_size - 1; i++)
    {
        int step = grid_size - 1;
        int dest_proc = my_i * grid_size + (my_j + step) % grid_size;
        int source_proc = my_i * grid_size + (my_j + (grid_size - step)) % grid_size;
        int *a_block_copy;
        a_block_copy = copy_matrix(a_block, block_size, block_size);
        MPI_Sendrecv(a_block_copy, block_size * block_size, MPI_INT, dest_proc, 0,
                     a_block, block_size * block_size, MPI_INT, source_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        dest_proc = ((my_i + step) % grid_size) * grid_size + my_j;
        source_proc = ((my_i + (grid_size - step)) % grid_size) * grid_size + my_j;
        // printf("Process %d source_proc: %d, dest_proc: %d\n", my_rank, source_proc, dest_proc);
        int *b_block_copy;
        b_block_copy = copy_matrix(b_block, block_size, block_size);
        MPI_Sendrecv(b_block_copy, block_size * block_size, MPI_INT, dest_proc, 1,
                     b_block, block_size * block_size, MPI_INT, source_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        BlockMultiplication(a_block, b_block, c_block, block_size);
    }
}

void ResultCollection(int *c_matrix, int *c_block, int size, int block_size)
{
    int grid_size = (int)sqrt((int)thread_cnt);

    if (my_rank == 0)
    {
        int row_start = (my_rank / grid_size) * block_size;
        int row_end = row_start + block_size;
        int col_start = (my_rank % grid_size) * block_size;
        int col_end = col_start + block_size;

        for (int i = row_start; i < row_end; i++)
        {
            int block_i = i - row_start;
            for (int j = col_start; j < col_end; j++)
            {
                int block_j = j - col_start;
                c_matrix[i * size + j] = c_block[block_i * block_size + block_j];
            }
        }

        for (int rank = 1; rank < thread_cnt; rank++)
        {
            int *tmp;
            tmp = (int *)malloc(block_size * block_size * sizeof(int));
            MPI_Recv(tmp, block_size * block_size, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int row_start = (rank / grid_size) * block_size;
            int row_end = row_start + block_size;
            int col_start = (rank % grid_size) * block_size;
            int col_end = col_start + block_size;

            for (int i = row_start; i < row_end; i++)
            {
                int block_i = i - row_start;
                for (int j = col_start; j < col_end; j++)
                {
                    int block_j = j - col_start;
                    c_matrix[i * size + j] = tmp[block_i * block_size + block_j];
                }
            }
        }
    }
    else
    {
        MPI_Send(c_block, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void FreeMemory(int *a_matrix, int *b_matrix, int *c_matrix, int *a_block, int *b_block, int *c_block)
{
    if (my_rank == 0)
    {
        free(a_matrix);
        free(b_matrix);
        free(c_matrix);
    }
    free(a_block);
    free(b_block);
    free(c_block);
}

int main(int argc, char *argv[])
{

    int size;
    size = strtol(argv[1], NULL, 10);
    int *a_matrix;
    int *b_matrix;
    int *c_matrix;

    int *a_block, *b_block, *c_block;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    grid_size = (int)sqrt((double)thread_cnt);
    int block_size = size / grid_size;
    if (thread_cnt != grid_size * grid_size)
    {
        if (my_rank == 0)
        {
            printf("Number of processes must be a perfect square \n");
            return -1;
        }
    }
    else
    {

        // Выделение памяти и инициализация элементов матриц
        if (my_rank == 0)
        {
            if (size % grid_size != 0)
            {
                printf("Matrix size should be divided by grid size \n");
                return -1;
            }
        }
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        a_block = (int *)malloc(block_size * block_size * sizeof(int));
        b_block = (int *)malloc(block_size * block_size * sizeof(int));
        c_block = (int *)malloc(block_size * block_size * sizeof(int));
        initialize_matrix_with_zeros(c_block, block_size, block_size);

        if (my_rank == 0)
        {
            a_matrix = (int *)malloc(size * size * sizeof(int));
            b_matrix = (int *)malloc(size * size * sizeof(int));
            c_matrix = (int *)malloc(size * size * sizeof(int));

            initialize_matrix(a_matrix, size, size);
            initialize_matrix(b_matrix, size, size);
            initialize_matrix_with_zeros(c_matrix, size, size);
        }

        // Блочное распределение матриц между процессами
        DataDistribution(a_matrix, b_matrix, a_block, b_block, size, block_size);

        // Начальный сдвиг
        InitialSkew(a_block, b_block, c_block, block_size);

        // Параллельное умножение матриц
        double local_time_start = MPI_Wtime();

        ShiftAndMultiply(a_block, b_block, c_block, block_size);

        double local_time_finish = MPI_Wtime();
        double local_time_elapsed = local_time_finish - local_time_start;
        double time_elapsed;
        MPI_Reduce(&local_time_elapsed, &time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Сбор результирующей матрицы на ведущем процессе
        ResultCollection(c_matrix, c_block, size, block_size);

        if (my_rank == 0)
        {
            printf("Cannon algorithm on %d threads runtime:\n %lf\n", thread_cnt, time_elapsed);
        }

        // //---------------------OUTPUT----------------------
        // if (my_rank == 0) {
        //     printf("A matrix:\n");
        //     print_matrix(a_matrix, size, size);
        //     printf("B matrix:\n");
        //     print_matrix(b_matrix, size, size);
        //     printf("Result:\n");
        //     print_matrix(c_matrix, size, size);
        // }

        // printf("Process %d B block:\n", my_rank);
        // print_matrix(b_block, block_size, block_size);
    }

    FreeMemory(a_matrix, b_matrix, c_matrix, a_block, b_block, c_block);

    MPI_Finalize();
    return 0;
}
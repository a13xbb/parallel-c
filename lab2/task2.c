#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "./utils.h"

int thread_cnt;
int my_rank;
int GridSize;
int GridCoords[2];
MPI_Comm GridComm; // коммуникатор - столбец решетки
MPI_Comm ColComm;  // коммуникатор - строка решетки
MPI_Comm RowComm;

// Создание коммуникатора в виде двумерной квадратной решетки
// и коммуникаторов для каждой строки и каждого столбца решетки
void CreateGridCommunicators()
{
    int DimSize[2]; // Количество процессов в каждом измерении
    // решетки
    int Periodic[2]; // =1 для каждого измерения, являющегося
    // периодическим
    int Subdims[2]; // =1 для каждого измерения, оставляемого
    // в подрешетке
    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 0;
    Periodic[1] = 0;
    // Создание коммуникатора в виде квадратной решетки
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    // Определение координат процесса в решетке
    MPI_Cart_coords(GridComm, my_rank, 2, GridCoords);
    // Создание коммуникаторов для строк процессной решетки
    Subdims[0] = 0; // Фиксация измерения
    Subdims[1] = 1; // Наличие данного измерения в подрешетке
    MPI_Cart_sub(GridComm, Subdims, &RowComm);
    // Создание коммуникаторов для столбцов процессной решетки
    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

void DataDistribution(int *a_matrix, int *b_matrix, int *matrix_a_block, int *b_block, int size,
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
        // print_matrix(a_matrix, size, size);
        // print_matrix(b_matrix, size, size);
        for (int row = row_start; row < row_end; row++)
        {
            for (int col = col_start; col < col_end; col++)
            {
                matrix_a_block[tmp_idx] = a_matrix[row * size + col];
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
        MPI_Recv(matrix_a_block, block_size * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_block, block_size * block_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void ABlockCommunication(int iter, int *a_block,
                         int *matrix_a_block, int block_size)
{
    // Определение ведущего процесса в строке процессной решетки
    int Pivot = (GridCoords[0] + iter) % GridSize;
    // Копирование передаваемого блока в отдельный буфер памяти
    if (GridCoords[1] == Pivot)
    {
        for (int i = 0; i < block_size * block_size; i++)
        {
            a_block[i] = matrix_a_block[i];
        }
    }
    // Рассылка блока
    MPI_Bcast(a_block, block_size * block_size, MPI_INT, Pivot,
              RowComm);
}

void BlockMultiplication(int *a_block, int *b_block,
                         int *c_block, int block_size)
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

void BblockCommunication(int *b_block, int block_size)
{
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1)
        NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0)
        PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(b_block, block_size * block_size, MPI_INT,
                         NextProc, 0, PrevProc, 0, ColComm, &Status);
}

void ParallelResultCalculation(int *a_block, int *matrix_a_block, int *b_block, int *c_block, int block_size)
{
    for (int iter = 0; iter < GridSize; iter++)
    {
        // Рассылка блоков матрицы А по строкам процессной решетки
        ABlockCommunication(iter, a_block, matrix_a_block, block_size);
        // Умножение блоков
        BlockMultiplication(a_block, b_block, c_block, block_size);
        // Циклический сдвиг блоков матрицы В в столбцах процессной
        // решетки
        BblockCommunication(b_block, block_size);
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

// void ProcessTermination(int *a_matrix, int *b_matrix, int *c_matrix, int *a_block, int *b_block,
//                         int *c_block, int *matrix_a_block)
// {
//     if (my_rank == 0)
//     {
//         free(a_matrix);
//         free(b_matrix);
//         free(c_matrix);
//     }
//     free(a_block);
//     free(b_block);
//     free(c_block);
//     free(matrix_a_block);
// }

int main(int argc, char *argv[])
{

    int size;
    size = strtol(argv[1], NULL, 10);
    int *a_matrix;
    int *b_matrix;
    int *c_matrix;
    int *matrix_a_block;

    int *a_block, *b_block, *c_block;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    GridSize = (int)sqrt((double)thread_cnt);
    int block_size = size / GridSize;
    if (thread_cnt != GridSize * GridSize)
    {
        if (my_rank == 0)
        {
            printf("Number of processes must be a perfect square \n");
            return -1;
        }
    }
    else
    {
        // Создание виртуальной решетки процессов и коммуникаторов
        // строк и столбцов
        CreateGridCommunicators();
        // Выделение памяти и инициализация элементов матриц
        if (my_rank == 0)
        {
            if (size % GridSize != 0)
            {
                printf("Matrix size should be divided by grid size \n");
                return -1;
            }
        }
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        a_block = (int *)malloc(block_size * block_size * sizeof(int));
        b_block = (int *)malloc(block_size * block_size * sizeof(int));
        c_block = (int *)malloc(block_size * block_size * sizeof(int));
        matrix_a_block = (int *)malloc(block_size * block_size * sizeof(int));
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
        DataDistribution(a_matrix, b_matrix, matrix_a_block, b_block, size, block_size);

        double local_time_start = MPI_Wtime();
        // Выполнение параллельного метода Фокса
        ParallelResultCalculation(a_block, matrix_a_block, b_block,
                                  c_block, block_size);

        // Сбор результирующей матрицы на ведущем процессе
        ResultCollection(c_matrix, c_block, size, block_size);

        double local_time_finish = MPI_Wtime();
        double local_time_elapsed = local_time_finish - local_time_start;
        double time_elapsed;
        MPI_Reduce(&local_time_elapsed, &time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (my_rank == 0)
        {
            printf("Fox algorithm on %d threads runtime = %lf:\n", thread_cnt, time_elapsed);
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

        // Завершение процесса вычислений
        // ProcessTermination(a_matrix, b_matrix, c_matrix, a_block, b_block,
        //                    c_block, matrix_a_block);
    }

    MPI_Finalize();
    return 0;
}
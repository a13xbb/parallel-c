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

void ABlockCommunication(int iter, int *pAblock,
                         int *pMatrixAblock, int BlockSize)
{
    // Определение ведущего процесса в строке процессной решетки
    int Pivot = (GridCoords[0] + iter) % GridSize;
    // Копирование передаваемого блока в отдельный буфер памяти
    if (GridCoords[1] == Pivot)
    {
        for (int i = 0; i < BlockSize * BlockSize; i++)
        {
            pAblock[i] = pMatrixAblock[i];
        }
    }
    // Рассылка блока
    MPI_Bcast(pAblock, BlockSize * BlockSize, MPI_DOUBLE, Pivot,
              RowComm);
}

void BlockMultiplication(int *pAblock, int *pBblock,
                         int *pCblock, int BlockSize)
{
    // Вычисление произведения матричных блоков
    for (int i = 0; i < BlockSize; i++)
    {
        for (int j = 0; j < BlockSize; j++)
        {
            {
                int temp = 0;
                for (int k = 0; k < BlockSize; k++) {
                    temp += pAblock[i * BlockSize + k] * pBblock[k * BlockSize + j];
                }
                pCblock[i * BlockSize + j] += temp;
            }
        }
    }
}

void BblockCommunication(int *pBblock, int BlockSize)
{
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1)
        NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0)
        PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(pBblock, BlockSize * BlockSize, MPI_DOUBLE,
                         NextProc, 0, PrevProc, 0, ColComm, &Status);
}

void ParallelResultCalculation(int *pAblock, int *pMatrixAblock, int *pBblock, int *pCblock, int BlockSize)
{
    for (int iter = 0; iter < GridSize; iter++)
    {
        // Рассылка блоков матрицы А по строкам процессной решетки
        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);
        // Умножение блоков
        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);
        // Циклический сдвиг блоков матрицы В в столбцах процессной
        // решетки
        BblockCommunication(pBblock, BlockSize);
    }
}

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
                printf("Размер матриц должен быть кратен размеру сетки! \n");
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
        DataDistribution(a_matrix, b_matrix, matrix_a_block, b_block, size,
                         block_size);

        if (my_rank == 0)
        {
            printf("A matrix:\n");
            print_matrix(a_matrix, size, size);
            printf("B matrix:\n");
            print_matrix(b_matrix, size, size);
        }

        // printf("Thread %d A block:\n", my_rank);
        // print_matrix(matrix_a_block, block_size, block_size);
        // MPI_Barrier(MPI_COMM_WORLD);
        // printf("Thread %d B block:\n", my_rank);
        // print_matrix(b_block, block_size, block_size);

        // Выполнение параллельного метода Фокса
        ParallelResultCalculation(a_block, matrix_a_block, b_block,
                                  c_block, block_size);

        printf("Thread %d c_block:\n", my_rank);
        print_matrix(c_block, block_size, block_size);
        // Сбор результирующей матрицы на ведущем процессе
        // ResultCollection(pCMatrix, pCblock, Size, BlockSize);
        // Завершение процесса вычислений
        // ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock,
        //                    pCblock, pMatrixAblock);
    }

    MPI_Finalize();
}
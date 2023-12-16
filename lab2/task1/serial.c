#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../utils.h"

int main(int argc, char* argV[]) {
    int n_rows, n_cols;
    n_rows = strtol(argV[1], NULL, 10);
    n_cols = strtol(argV[2], NULL, 10);

    int *matrix, *vector, *product;


    vector = (int*)malloc(n_cols * sizeof(int));
    matrix = (int*)malloc(n_rows * n_cols * sizeof(int));
    product = (int*)malloc(n_rows * sizeof(int));

    initialize_matrix(matrix, n_rows, n_cols);
    initialize_vector(vector, n_cols);

    clock_t begin = clock();

    for (int i = 0; i < n_rows; i++) {
        product[i] = 0;
        for (int j = 0; j < n_cols; j++) {
            product[i] += matrix[i * n_cols + j] * vector[j];
        }
    }

    clock_t end = clock();

    //---------------------OUTPUT---------------------
    // printf("Matrix:\n");
    // print_matrix(matrix, n_rows, n_cols);

    // printf("Vector:\n");
    // print_vector(vector, n_cols);

    // printf("Product:\n");
    // print_vector(product, n_rows);

    printf("Serial runtime: %.6lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    free(matrix);
    free(vector);
    free(product);

    return 0;
}
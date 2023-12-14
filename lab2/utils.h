#include <stdio.h>
#include <stdlib.h>

void initialize_matrix(int *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            matrix[i * n_cols + j] = rand() % 10;
        }
    }
}

void initialize_matrix_with_zeros(int *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            matrix[i * n_cols + j] = 0;
        }
    }
}

void initialize_vector(int *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = rand() % 10;
    }
}

void initialize_vector_with_zeros(int *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = 0;
    } 
}

void print_vector(int *vector, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n\n");
}

void print_matrix(int *matrix, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%d ", matrix[i * n_cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}
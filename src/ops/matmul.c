#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../include/tensor.h"

// Matrix multiplication
void matmul(Tensor* A, Tensor* B, Tensor* C) {
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: NULL tensor in matmul\n");
        return;
    }
    
    // Validate that inputs are 2D
    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "Error: matmul requires 2D tensors\n");
        return;
    }
    
    // Validate dimensions
    if (A->shape[1] != B->shape[0]) {
        fprintf(stderr, "Error: incompatible shapes for matmul: (%d,%d) @ (%d,%d)\n",
                A->shape[0], A->shape[1], B->shape[0], B->shape[1]);
        return;
    }
    
    int m = A->shape[0];
    int k = A->shape[1];
    int n = B->shape[1];
    
    // Validate output shape
    if (C->ndim != 2 || C->shape[0] != m || C->shape[1] != n) {
        fprintf(stderr, "Error: output tensor has wrong shape\n");
        return;
    }
    
    // Naive implementation - can be optimized later with SIMD
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A->data[i * k + p] * B->data[p * n + j];
            }
            C->data[i * n + j] = sum;
        }
    }
}

// Matrix-vector multiplication
void matvec(Tensor* A, float* x, float* y, int m, int n) {
    if (A == NULL || x == NULL || y == NULL) {
        fprintf(stderr, "Error: NULL pointer in matvec\n");
        return;
    }
    
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A->data[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// Batched matrix multiplication for attention
void batched_matmul(Tensor* A, Tensor* B, Tensor* C, int batch, int m, int k, int n) {
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: NULL tensor in batched_matmul\n");
        return;
    }
    
    for (int b = 0; b < batch; b++) {
        float* A_batch = A->data + b * m * k;
        float* B_batch = B->data + b * k * n;
        float* C_batch = C->data + b * m * n;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += A_batch[i * k + p] * B_batch[p * n + j];
                }
                C_batch[i * n + j] = sum;
            }
        }
    }
}
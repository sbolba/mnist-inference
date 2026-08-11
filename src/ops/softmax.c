#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "../include/tensor.h"

// Softmax along the last dimension
// For numerical stability, we subtract the max value
void softmax(float* input, float* output, int size) {
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Softmax for 2D tensor (batch processing)
// Apply softmax to each row independently
void softmax_2d(Tensor* input, Tensor* output, int rows, int cols) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in softmax_2d\n");
        return;
    }
    
    if (input->ndim != 2 || output->ndim != 2) {
        fprintf(stderr, "Error: softmax_2d requires 2D tensors\n");
        return;
    }
    
    for (int i = 0; i < rows; i++) {
        float* input_row = input->data + i * cols;
        float* output_row = output->data + i * cols;
        softmax(input_row, output_row, cols);
    }
}

// Softmax for attention scores
// input: (batch, num_heads, seq_len, seq_len)
// Apply softmax over the last dimension
void softmax_attention(float* input, float* output, int batch, int num_heads, int seq_len) {
    int total_rows = batch * num_heads * seq_len;
    
    for (int i = 0; i < total_rows; i++) {
        float* input_row = input + i * seq_len;
        float* output_row = output + i * seq_len;
        softmax(input_row, output_row, seq_len);
    }
}

// Softmax for tensors (applies along last dimension)
void softmax_tensor(Tensor* input, Tensor* output) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in softmax_tensor\n");
        return;
    }
    
    // Compute size of last dimension
    int last_dim_size = input->shape[input->ndim - 1];
    
    // Compute number of rows
    size_t total_size = tensor_compute_size(input->shape, input->ndim);
    int num_rows = total_size / last_dim_size;
    
    // Apply softmax to each row
    for (int i = 0; i < num_rows; i++) {
        float* input_row = input->data + i * last_dim_size;
        float* output_row = output->data + i * last_dim_size;
        softmax(input_row, output_row, last_dim_size);
    }
}
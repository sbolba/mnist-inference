#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../include/tensor.h"

// Layer Normalization
// Normalizes across the last dimension (features)
// output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
// 
// gamma and beta are learnable parameters loaded from model weights

#define LAYERNORM_EPSILON 1e-5f

void layernorm(float* input, float* output, float* gamma, float* beta, 
               int batch_size, int hidden_dim) {
    
    for (int b = 0; b < batch_size; b++) {
        float* input_row = input + b * hidden_dim;
        float* output_row = output + b * hidden_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            mean += input_row[i];
        }
        mean /= hidden_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;
        
        // Normalize and apply affine transformation
        float std_inv = 1.0f / sqrtf(variance + LAYERNORM_EPSILON);
        for (int i = 0; i < hidden_dim; i++) {
            float normalized = (input_row[i] - mean) * std_inv;
            output_row[i] = gamma[i] * normalized + beta[i];
        }
    }
}

// Simplified version for single vector
void layernorm_1d(float* input, float* output, float* gamma, float* beta, int size) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    // Normalize and apply affine transformation
    float std_inv = 1.0f / sqrtf(variance + LAYERNORM_EPSILON);
    for (int i = 0; i < size; i++) {
        float normalized = (input[i] - mean) * std_inv;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

// Layer normalization for tensors
// Normalizes along the last dimension
void layernorm_tensor(Tensor* input, Tensor* output, Tensor* gamma, Tensor* beta) {
    if (input == NULL || output == NULL || gamma == NULL || beta == NULL) {
        fprintf(stderr, "Error: NULL tensor in layernorm_tensor\n");
        return;
    }
    
    // gamma and beta should be 1D tensors with size = last dimension of input
    if (gamma->ndim != 1 || beta->ndim != 1) {
        fprintf(stderr, "Error: gamma and beta must be 1D tensors\n");
        return;
    }
    
    int hidden_dim = gamma->shape[0];
    
    // Verify last dimension matches
    if (input->shape[input->ndim - 1] != hidden_dim) {
        fprintf(stderr, "Error: input last dimension must match gamma/beta size\n");
        return;
    }
    
    // Compute batch size (all dimensions except last)
    size_t total_size = tensor_compute_size(input->shape, input->ndim);
    int batch_size = total_size / hidden_dim;
    
    layernorm(input->data, output->data, gamma->data, beta->data, 
              batch_size, hidden_dim);
}
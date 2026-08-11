#include <string.h>
#include <stdio.h>
#include "../include/tensor.h"

// Element-wise addition: result = a + b
void add_vectors(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

// In-place addition: a = a + b
void add_vectors_inplace(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

// Tensor addition
void add_tensors(Tensor* a, Tensor* b, Tensor* result) {
    if (a == NULL || b == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL tensor in add_tensors\n");
        return;
    }
    
    size_t size_a = tensor_compute_size(a->shape, a->ndim);
    size_t size_b = tensor_compute_size(b->shape, b->ndim);
    size_t size_r = tensor_compute_size(result->shape, result->ndim);
    
    if (size_a != size_b || size_a != size_r) {
        fprintf(stderr, "Error: tensor sizes must match for addition\n");
        return;
    }
    
    add_vectors(a->data, b->data, result->data, size_a);
}

// Scalar multiplication: result = scalar * input
void scale_vector(float* input, float scalar, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = scalar * input[i];
    }
}

// Tensor scalar multiplication
void scale_tensor(Tensor* input, float scalar, Tensor* output) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in scale_tensor\n");
        return;
    }
    
    size_t size = tensor_compute_size(input->shape, input->ndim);
    scale_vector(input->data, scalar, output->data, size);
}

// Copy vector
void copy_vector(float* src, float* dst, int size) {
    memcpy(dst, src, size * sizeof(float));
}

// Copy tensor data
void copy_tensor_data(Tensor* src, Tensor* dst) {
    if (src == NULL || dst == NULL) {
        fprintf(stderr, "Error: NULL tensor in copy_tensor_data\n");
        return;
    }
    
    size_t size_src = tensor_compute_size(src->shape, src->ndim);
    size_t size_dst = tensor_compute_size(dst->shape, dst->ndim);
    
    if (size_src != size_dst) {
        fprintf(stderr, "Error: tensor sizes must match for copy\n");
        return;
    }
    
    copy_vector(src->data, dst->data, size_src);
}

// Element-wise multiplication: result = a * b
void multiply_vectors(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

// Tensor element-wise multiplication (Hadamard product)
void multiply_tensors(Tensor* a, Tensor* b, Tensor* result) {
    if (a == NULL || b == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL tensor in multiply_tensors\n");
        return;
    }
    
    size_t size_a = tensor_compute_size(a->shape, a->ndim);
    size_t size_b = tensor_compute_size(b->shape, b->ndim);
    size_t size_r = tensor_compute_size(result->shape, result->ndim);
    
    if (size_a != size_b || size_a != size_r) {
        fprintf(stderr, "Error: tensor sizes must match for multiplication\n");
        return;
    }
    
    multiply_vectors(a->data, b->data, result->data, size_a);
}

// Dot product
float dot_product(float* a, float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// ReLU activation (might be useful)
void relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// ReLU for tensors
void relu_tensor(Tensor* input, Tensor* output) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in relu_tensor\n");
        return;
    }
    
    size_t size = tensor_compute_size(input->shape, input->ndim);
    relu(input->data, output->data, size);
}

// Transpose a 2D matrix
// input: (m, n) -> output: (n, m)
void transpose(float* input, float* output, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            output[j * m + i] = input[i * n + j];
        }
    }
}

// Transpose 2D tensor
void transpose_tensor(Tensor* input, Tensor* output) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in transpose_tensor\n");
        return;
    }
    
    if (input->ndim != 2 || output->ndim != 2) {
        fprintf(stderr, "Error: transpose requires 2D tensors\n");
        return;
    }
    
    int m = input->shape[0];
    int n = input->shape[1];
    
    if (output->shape[0] != n || output->shape[1] != m) {
        fprintf(stderr, "Error: output tensor has wrong shape for transpose\n");
        return;
    }
    
    transpose(input->data, output->data, m, n);
}
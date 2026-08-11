#include <math.h>
#include <stdio.h>
#include "../include/tensor.h"

// GELU (Gaussian Error Linear Unit) activation function
// GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
// 
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// This is the approximation used in the original GPT-2 implementation

#define GELU_CONSTANT_A 0.7978845608028654   // sqrt(2/pi)
#define GELU_CONSTANT_B 0.044715

void gelu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = GELU_CONSTANT_A * (x + GELU_CONSTANT_B * x_cubed);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// In-place GELU
void gelu_inplace(float* data, int size) {
    gelu(data, data, size);
}

// GELU for tensors
void gelu_tensor(Tensor* input, Tensor* output) {
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL tensor in gelu_tensor\n");
        return;
    }
    
    size_t input_size = tensor_compute_size(input->shape, input->ndim);
    size_t output_size = tensor_compute_size(output->shape, output->ndim);
    
    if (input_size != output_size) {
        fprintf(stderr, "Error: input and output tensors must have same size\n");
        return;
    }
    
    gelu(input->data, output->data, input_size);
}

// In-place GELU for tensors
void gelu_tensor_inplace(Tensor* tensor) {
    if (tensor == NULL) {
        fprintf(stderr, "Error: NULL tensor in gelu_tensor_inplace\n");
        return;
    }
    
    size_t size = tensor_compute_size(tensor->shape, tensor->ndim);
    gelu_inplace(tensor->data, size);
}

// Alternative: exact GELU using error function (slower but more accurate)
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu_exact(float* input, float* output, int size) {
    const float sqrt_2_inv = 0.7071067811865476f;  // 1/sqrt(2)
    
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
    }
}
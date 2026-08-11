#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// ============================================================================
// Matrix Operations
// ============================================================================
// Matrix multiplication: C = A @ B
void matmul(Tensor* A, Tensor* B, Tensor* C);

// Matrix-vector multiplication: y = A @ x
void matvec(Tensor* A, float* x, float* y, int m, int n);

// Batched matrix multiplication
void batched_matmul(Tensor* A, Tensor* B, Tensor* C, int batch, int m, int k, int n);

// ============================================================================
// Activation Functions
// ============================================================================
// Softmax
void softmax(float* input, float* output, int size);
void softmax_2d(Tensor* input, Tensor* output, int rows, int cols);
void softmax_attention(float* input, float* output, int batch, int num_heads, int seq_len);
void softmax_tensor(Tensor* input, Tensor* output);

// GELU (Gaussian Error Linear Unit)
void gelu(float* input, float* output, int size);
void gelu_inplace(float* data, int size);
void gelu_tensor(Tensor* input, Tensor* output);
void gelu_tensor_inplace(Tensor* tensor);
void gelu_exact(float* input, float* output, int size);

// ReLU
void relu(float* input, float* output, int size);
void relu_tensor(Tensor* input, Tensor* output);

// ============================================================================
// Normalization
// ============================================================================
// Layer Normalization
void layernorm(float* input, float* output, float* gamma, float* beta, 
               int batch_size, int hidden_dim);
void layernorm_1d(float* input, float* output, float* gamma, float* beta, int size);
void layernorm_tensor(Tensor* input, Tensor* output, Tensor* gamma, Tensor* beta);

// ============================================================================
// Vector Operations
// ============================================================================
// Addition
void add_vectors(float* a, float* b, float* result, int size);
void add_vectors_inplace(float* a, float* b, int size);
void add_tensors(Tensor* a, Tensor* b, Tensor* result);

// Scalar multiplication
void scale_vector(float* input, float scalar, float* output, int size);
void scale_tensor(Tensor* input, float scalar, Tensor* output);

// Element-wise multiplication
void multiply_vectors(float* a, float* b, float* result, int size);
void multiply_tensors(Tensor* a, Tensor* b, Tensor* result);

// Copy operations
void copy_vector(float* src, float* dst, int size);
void copy_tensor_data(Tensor* src, Tensor* dst);

// Dot product
float dot_product(float* a, float* b, int size);

// Transpose
void transpose(float* input, float* output, int m, int n);
void transpose_tensor(Tensor* input, Tensor* output);

// Model-specific operations
void conv2d(const float* input, const float* kernel, const float* bias, float* output, int in_channels, int out_channels, int in_h, int in_w, int kernel_size, int stride, int padding);
void maxpool2d(const float* input, float* output, int channels, int in_h, int in_w, int pool_size, int stride);

// Multi-head attention (generica: dimensioni a runtime, non più fisse a 50x64)
// tokens/out: buffer flattened [seq_len][d_model]
// in_proj_w: [3*d_model][d_model] (Q,K,V concatenati), in_proj_b: [3*d_model]
// out_proj_w: [d_model][d_model], out_proj_b: [d_model]
void multi_head_attention(float* tokens, int seq_len, int d_model, int n_heads,
                          float* in_proj_w, float* in_proj_b,
                          float* out_proj_w, float* out_proj_b,
                          float* out);

// Feed-Forward Network (ffn) - opera sempre su d_model=64, d_ff=256 (fissi, indipendenti da patch size)
void ffn(float* input, float* output, float* w1, float* b1, float* w2, float* b2);

// Argmax
int argmax(float* logits, int size);

#endif // OPS_H

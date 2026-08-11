#include <stdio.h>
#include <stdlib.h>
#include "../include/mnist.h"
#include "../include/ops.h"
#include "../include/tensor.h"
#ifdef _WIN32
    #include "../include/mmap_loader_windows.h"
#else // Posix
    #include "../include/mmap_loader_posix.h"
#endif

// Uses the MmapLoader to open model.bin and read the weights and biases into the MnistModel struct.
int mnist_load_model(MnistModel* model, const char* path) {
    
    MmapFile* file = mmap_open(path);
    if (!file) {
        fprintf(stderr, "Error opening model file: %s\n", path);
        return -1; // Errore apertura file
    }

    // Lettura Tensori
    size_t offset = 0;
    Tensor* conv1_weight = mmap_read_tensor(file, offset, (int32_t[]){32, 1, 3, 3}, 4);
    offset += 32*1*3*3*sizeof(float);
    Tensor* conv1_bias = mmap_read_tensor(file, offset, (int32_t[]){32}, 1);
    offset += 32*sizeof(float);
    Tensor* conv2_weight = mmap_read_tensor(file, offset, (int32_t[]){64, 32, 3, 3}, 4);
    offset += 64*32*3*3*sizeof(float);
    Tensor* conv2_bias = mmap_read_tensor(file, offset, (int32_t[]){64}, 1);
    offset += 64*sizeof(float);
    Tensor* fc1_weight = mmap_read_tensor(file, offset, (int32_t[]){128, 9216}, 2);
    offset += 128*9216*sizeof(float);
    Tensor* fc1_bias = mmap_read_tensor(file, offset, (int32_t[]){128}, 1);
    offset += 128*sizeof(float);
    Tensor* fc2_weight = mmap_read_tensor(file, offset, (int32_t[]){10, 128}, 2);
    offset += 10*128*sizeof(float);
    Tensor* fc2_bias = mmap_read_tensor(file, offset, (int32_t[]){10}, 1);

    // Inserimento tensori nel modello
    model->conv1_weight = conv1_weight->data;
    model->conv1_bias = conv1_bias->data;
    model->conv2_weight = conv2_weight->data;
    model->conv2_bias = conv2_bias->data;
    model->fc1_weight = fc1_weight->data;
    model->fc1_bias = fc1_bias->data;
    model->fc2_weight = fc2_weight->data;
    model->fc2_bias = fc2_bias->data;
    model->source_handle = file; // Salva handle per chiusura futura

    printf("Succesfully loaded CNN model\n");

    return 0;
}

// ============================================================================
// Forward pass (architettura PyTorch standard MNIST CNN):
//   conv1(1→32, 3x3, pad=0): 1x28x28 → 32x26x26 + ReLU
//   conv2(32→64, 3x3, pad=0): 32x26x26 → 64x24x24 + ReLU
//   maxpool(2x2): 64x24x24 → 64x12x12
//   flatten: 64*12*12 = 9216
//   fc1: 9216 → 128 + ReLU
//   fc2: 128 → 10
//   argmax → digit predetto (0-9)
// ============================================================================
int mnist_predict(const MnistModel* model, const float* image) {

    // Conv1: 1x28x28 → 32x26x26, poi ReLU
    float conv1_out[32 * 26 * 26];
    conv2d(image, model->conv1_weight, model->conv1_bias, conv1_out,
           1, 32, 28, 28, 3, 1, 0);
    relu(conv1_out, conv1_out, 32 * 26 * 26);

    // Conv2: 32x26x26 → 64x24x24, poi ReLU
    float conv2_out[64 * 24 * 24];
    conv2d(conv1_out, model->conv2_weight, model->conv2_bias, conv2_out,
           32, 64, 26, 26, 3, 1, 0);
    relu(conv2_out, conv2_out, 64 * 24 * 24);

    // MaxPool 2x2: 64x24x24 → 64x12x12 (= 9216 valori)
    float pool_out[64 * 12 * 12];
    maxpool2d(conv2_out, pool_out, 64, 24, 24, 2, 2);

    // FC1: Linear(9216 → 128) + ReLU
    float fc1_out[128];
    for (int i = 0; i < 128; i++) {
        float sum = model->fc1_bias[i];
        for (int j = 0; j < 9216; j++) {
            sum += model->fc1_weight[i * 9216 + j] * pool_out[j];
        }
        fc1_out[i] = sum;
    }
    relu(fc1_out, fc1_out, 128);

    // FC2: Linear(128 → 10)
    float fc2_out[10];
    for (int i = 0; i < 10; i++) {
        float sum = model->fc2_bias[i];
        for (int j = 0; j < 128; j++) {
            sum += model->fc2_weight[i * 128 + j] * fc1_out[j];
        }
        fc2_out[i] = sum;
    }

    // Argmax: trova il digit con il punteggio più alto
    int predicted = 0;
    float max_val = fc2_out[0];
    for (int i = 1; i < 10; i++) {
        if (fc2_out[i] > max_val) {
            max_val = fc2_out[i];
            predicted = i;
        }
    }

    return predicted;
}

void mnist_free_model(MnistModel* model) {
    // La memoria dei tensori è gestita dal MmapFile, quindi non dobbiamo liberarla manualmente.
    if (model && model->source_handle) {
        mmap_close(model->source_handle);
        model->source_handle = NULL;
    }
}
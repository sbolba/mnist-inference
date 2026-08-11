#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
    #include "mmap_loader_windows.h"
#else
    #include "mmap_loader_posix.h"
#endif

typedef struct {
    float* conv1_weight;
    float* conv1_bias;
    float* conv2_weight;
    float* conv2_bias;
    float* fc1_weight;
    float* fc1_bias;
    float* fc2_weight;
    float* fc2_bias;
    MmapFile* source_handle;
} MnistModel;

int mnist_load_model(MnistModel* model, const char* path);

// Returns the predicted digit (0-9) for the given image
int mnist_predict(const MnistModel* model, const float* image);

void mnist_free_model(MnistModel* model);

#endif // MNIST_H
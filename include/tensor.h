#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

// Struttura tensori
typedef struct {
    float* data; // Puntatore ai dati
    int32_t* shape; // Forma del tensore (es [2, 3, 4])
    int32_t ndim; // Numero di dimensioni
    int owns_data; // Flag per indicare se il tensore possiede i dati (per gestione memoria)
} Tensor;

size_t tensor_compute_size(int32_t* shape, int32_t ndim);
Tensor* tensor_from_data(float* data, int32_t* shape, int32_t ndim, int owns_data);
void tensor_destroy(Tensor* t);

#endif // TENSOR_H
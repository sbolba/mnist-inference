#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

size_t tensor_compute_size(int32_t* shape, int32_t ndim) {
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

Tensor* tensor_from_data(float* data, int32_t* shape, int32_t ndim, int owns_data) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = data;
    t->ndim = ndim;
    t->owns_data = owns_data;
    
    t->shape = (int32_t*)malloc(ndim * sizeof(int32_t));
    for (int i=0; i<ndim; i++) {
        t->shape[i] = shape[i];
    }

    return t;
}

void tensor_destroy(Tensor* t) {
    if (!t) return;

    if (t->shape) free(t->shape);

    if (t->owns_data && t->data) {
        free(t->data);
    }

    free(t);
}
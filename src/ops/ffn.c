#include <stdio.h>
#include <stdlib.h>
#include "../include/ops.h"

//input [64] → linear1 → [256] → GELU → linear2 → [64]
//È semplicemente due layer lineari con GELU in mezzo.
//Usata con transformer
void ffn(float* input, float* output, float* w1, float* b1, float* w2, float* b2) {
    float hidden[256];
    
    // linear1: hidden = input * w1^T + b1 + GELU
    for (int i = 0; i < 256; i++) {
        hidden[i] = b1[i];
        for (int j = 0; j < 64; j++) {
            hidden[i] += input[j] * w1[i * 64 + j];
        }
        gelu(&hidden[i], &hidden[i], 1); // Applica GELU in-place
    }
    
    // linear2: output = hidden * w2^T + b2
    for (int i = 0; i < 64; i++) {
        output[i] = b2[i];
        for (int j = 0; j < 256; j++) {
            output[i] += hidden[j] * w2[i * 256 + j];
        }
    }
}
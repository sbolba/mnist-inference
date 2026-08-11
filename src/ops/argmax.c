#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ops.h"

int argmax(float* logits, int size){
    int max_idx = 0;
    for (int i = 1; i < size; i++){
        if (logits[i] > logits[max_idx]) max_idx = i;
    }
    return max_idx;
}
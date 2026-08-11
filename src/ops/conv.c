#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ops.h"

// Questa funzione è responsabile dell'estrazione delle caratteristiche (features) dall'input (come bordi, texture o forme complesse).
// Funzione complessa che coinvolge molteplici cicli annidati per iterare su canali, dimensioni dell'input e kernel, e applicare le operazioni di convoluzione.
void conv2d(const float* input, const float* kernel, const float* bias, float* output, int in_channels, int out_channels, int in_h, int in_w, int kernel_size, int stride, int padding) {
    // Dimensioni dell'output
    int out_h = (in_h - kernel_size + 2 * padding) / stride + 1;
    int out_w = (in_w - kernel_size + 2 * padding) / stride + 1;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) { 
                
                float sum = bias[oc]; 
                
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += input[ic * in_h * in_w + ih * in_w + iw] * kernel[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw];
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + oh * out_w + ow] = sum; 
            }
        }
    }
}

//Questa funzione esegue il downsampling (riduzione delle dimensioni) per rendere la rete più leggera e robusta.
void maxpool2d(const float* input, float* output, int channels, int in_h, int in_w, int pool_size, int stride) {
    // Dimensioni dell'output
    int out_h = (in_h - pool_size) / stride + 1;
    int out_w = (in_w - pool_size) / stride + 1;

    for (int i = 0; i < channels; i++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                
                float max_val = -INFINITY;
                
                for (int ph = 0; ph < pool_size; ph++) {
                    for (int pw = 0; pw < pool_size; pw++) {
                        
                        int ih = h * stride + ph;
                        int iw = w * stride + pw;
                        
                        if (ih < in_h && iw < in_w) {
                            float val = input[i * in_h * in_w + ih * in_w + iw];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
                output[i * out_h * out_w + h * out_w + w] = max_val;
            }
        }
    }
}
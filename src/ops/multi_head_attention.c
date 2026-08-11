#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ops.h"

/*
multi_head_attention è il cuore del transformer. Ogni token guarda tutti gli altri token e decide a quali prestare attenzione.
Per esempio il token CLS impara a guardare le patch più rilevanti per capire che cifra è. Il multi-head significa che lo fa in
n_heads sottospazi diversi in parallelo, ognuno cattura relazioni diverse.

Versione generica: seq_len, d_model e n_heads sono parametri a runtime (non più
fissi a 50x64x4), così la stessa funzione serve per qualunque patch size del ViT
(cambia solo la lunghezza di sequenza, non l'architettura dell'attention).
I buffer intermedi sono allocati dinamicamente in base a seq_len/d_model.
*/
void multi_head_attention(float* tokens, int seq_len, int d_model, int n_heads,
                          float* in_proj_w,  // [3*d_model][d_model]
                          float* in_proj_b,  // [3*d_model]
                          float* out_proj_w, // [d_model][d_model]
                          float* out_proj_b, // [d_model]
                          float* out) {
    int d_head = d_model / n_heads;
    float scale = 1.0f / sqrtf((float)d_head);

    // calcola Q, K, V per tutti i seq_len token
    float* Q = (float*)malloc(sizeof(float) * seq_len * d_model);
    float* K = (float*)malloc(sizeof(float) * seq_len * d_model);
    float* V = (float*)malloc(sizeof(float) * seq_len * d_model);

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float q = in_proj_b[j];
            float k = in_proj_b[d_model + j];
            float v = in_proj_b[2 * d_model + j];
            for (int kk = 0; kk < d_model; kk++) {
                float t = tokens[i * d_model + kk];
                q += t * in_proj_w[j * d_model + kk];
                k += t * in_proj_w[(d_model + j) * d_model + kk];
                v += t * in_proj_w[(2 * d_model + j) * d_model + kk];
            }
            Q[i * d_model + j] = q;
            K[i * d_model + j] = k;
            V[i * d_model + j] = v;
        }
    }

    float* head_out = (float*)malloc(sizeof(float) * seq_len * d_model);
    float* scores = (float*)malloc(sizeof(float) * seq_len * seq_len);

    // per ogni testa
    for (int h = 0; h < n_heads; h++) {
        int offset = h * d_head;

        // scores = Q * K^T / sqrt(d_head)
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float s = 0.0f;
                for (int kk = 0; kk < d_head; kk++)
                    s += Q[i * d_model + offset + kk] * K[j * d_model + offset + kk];
                scores[i * seq_len + j] = s * scale;
            }
        }

        // softmax per ogni riga
        for (int i = 0; i < seq_len; i++) {
            float max_val = scores[i * seq_len];
            for (int j = 1; j < seq_len; j++)
                if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++)
                scores[i * seq_len + j] /= sum;
        }

        // output = scores * V
        for (int i = 0; i < seq_len; i++)
            for (int kk = 0; kk < d_head; kk++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++)
                    sum += scores[i * seq_len + j] * V[j * d_model + offset + kk];
                head_out[i * d_model + offset + kk] = sum;
            }
    }

    // out_proj: proietta [seq_len][d_model] → [seq_len][d_model]
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            float sum = out_proj_b[j];
            for (int kk = 0; kk < d_model; kk++)
                sum += head_out[i * d_model + kk] * out_proj_w[j * d_model + kk];
            out[i * d_model + j] = sum;
        }
    }

    free(Q);
    free(K);
    free(V);
    free(head_out);
    free(scores);
}

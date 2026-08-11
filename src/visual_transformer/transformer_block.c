#include <stdio.h>
#include <stdlib.h>
#include "../include/ops.h"
#include "../include/transformer_embedding.h"

/*
transformer_block è un blocco completo che combina attention e FFN
con le connessioni residuali e layer norm. È l'unità che si ripete — nel nostro caso 2 volte.

Versione generalizzata: la lunghezza di sequenza (cfg->seq_len) dipende dalla
patch size scelta, quindi i buffer temporanei sono allocati dinamicamente invece
che fissati a 50 token. d_model resta 64 (fisso, indipendente dalla patch size),
quindi il buffer per l'output della FFN può restare di dimensione fissa.
*/
//tokens[seq_len][d_model] → layernorm → multi-head attention → res1 → layernorm → ffn + res2 → tokens[seq_len][d_model]
void transformer_block(float* tokens, const ViTConfig* cfg,
                       float* norm1_w, float* norm1_b,
                       float* in_proj_w, float* in_proj_b,
                       float* out_proj_w, float* out_proj_b,
                       float* norm2_w, float* norm2_b,
                       float* ff_w1, float* ff_b1,
                       float* ff_w2, float* ff_b2) {
    int seq_len = cfg->seq_len;
    int d = cfg->d_model;

    float* temp = (float*)malloc(sizeof(float) * seq_len * d);
    float* attn_out = (float*)malloc(sizeof(float) * seq_len * d);

    // RETE 1
    // Pre-LN su ogni token
    for (int i = 0; i < seq_len; i++)
        layernorm_1d(&tokens[i * d], &temp[i * d], norm1_w, norm1_b, d);

    // Multi-Head Attention su temp
    multi_head_attention(temp, seq_len, d, cfg->n_heads, in_proj_w, in_proj_b, out_proj_w, out_proj_b, attn_out);

    // Residual 1
    for (int i = 0; i < seq_len * d; i++)
        tokens[i] += attn_out[i];

    // RETE 2
    // Pre-LN su ogni token
    for (int i = 0; i < seq_len; i++)
        layernorm_1d(&tokens[i * d], &temp[i * d], norm2_w, norm2_b, d);

    // FFN su ogni token + residual 2 (ffn opera sempre su d_model=64 fisso)
    for (int i = 0; i < seq_len; i++) {
        float ffn_out[VIT_D_MODEL];
        ffn(&temp[i * d], ffn_out, ff_w1, ff_b1, ff_w2, ff_b2);
        for (int j = 0; j < d; j++)
            tokens[i * d + j] += ffn_out[j];
    }

    free(temp);
    free(attn_out);
}

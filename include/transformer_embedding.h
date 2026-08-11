#ifndef TRANSFORMER_EMBEDDING_H
#define TRANSFORMER_EMBEDDING_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include "../include/tensor.h"
#ifdef _WIN32
    #include "../include/mmap_loader_windows.h"
#else // Posix
    #include "../include/mmap_loader_posix.h"
#endif

#define VIT_IMAGE_SIZE 28   // lato immagine, fisso (MNIST)
#define VIT_D_MODEL 64      // dimensione embedding, fissa: non dipende dalla patch size
#define VIT_N_HEADS 4       // numero teste attention, fisso
#define VIT_D_FF 256        // dimensione hidden della FFN, fissa
#define VIT_N_CLASSES 10    // cifre 0-9, fisso

// ============================================================================
// ViTConfig: tutte le dimensioni derivate dalla patch size scelta.
// d_model/n_heads/d_ff restano fissi (l'architettura del "backbone" transformer
// non cambia); cambiano solo l'embedding delle patch e la lunghezza di sequenza.
// ============================================================================
typedef struct {
    int image_size;        // 28
    int patch_size;        // lato della patch quadrata (es. 2, 4, 7, 14)
    int patches_per_side;  // image_size / patch_size
    int num_patches;        // patches_per_side^2
    int patch_dim;           // patch_size^2 (pixel per patch, immagini grayscale)
    int seq_len;               // num_patches + 1 (token CLS)
    int d_model;
    int n_heads;
    int d_head;                 // d_model / n_heads
    int d_ff;
    int n_classes;
} ViTConfig;

// Calcola e valida la config a partire dalla patch size scelta.
// Ritorna 0 se valida (28 % patch_size == 0), -1 altrimenti.
int vit_config_init(ViTConfig* cfg, int patch_size);

// Pesi di un singolo blocco transformer. Le dimensioni sono fisse
// (dipendono solo da VIT_D_MODEL/VIT_D_FF, non dalla patch size).
typedef struct{
    float* b0_norm1_w; float* b0_norm1_b; //norm1
    float* b0_attn_in_proj_w; float* b0_attn_in_proj_b; //attn in
    float* b0_attn_out_proj_w; float* b0_attn_out_proj_b; //attn out
    float* b0_norm2_w; float* b0_norm2_b; //norm2
    float* b0_ff_w1; float* b0_ff_b1; //ffn 1
    float* b0_ff_w2; float* b0_ff_b2; //ffn 2
}block;

// Modello ViT completo. W_proj/pos_embedding sono allocati dinamicamente
// perché la loro dimensione dipende dalla patch size scelta al caricamento.
typedef struct {
    ViTConfig config;
    float* W_proj;         // [d_model][patch_dim] flattened
    float* b_proj;         // [d_model]
    float* cls_token;      // [d_model]
    float* pos_embedding;  // [seq_len][d_model] flattened
    block b0;
    block b1;
    float final_norm_w[VIT_D_MODEL];
    float final_norm_b[VIT_D_MODEL];
    float classifier_w[VIT_N_CLASSES][VIT_D_MODEL];
    float classifier_b[VIT_N_CLASSES];
} ViTModel;

// image: buffer di image_size*image_size float (una sola immagine, offset già applicato)
// patches: buffer preallocato num_patches*patch_dim (flattened)
void create_patches(const float* image, float* patches, const ViTConfig* cfg);

// patches: [num_patches][patch_dim] flattened, W_proj: [d_model][patch_dim] flattened
// embeddings: [num_patches][d_model] flattened (output)
void compute_embeddings(const float* patches, const float* W_proj, const float* b_proj,
                         float* embeddings, const ViTConfig* cfg);

// embeddings: [num_patches][d_model], tokens: [seq_len][d_model] (output, CLS + patch + pos)
void prepare_tokens(const float* embeddings, const float* cls_token, const float* pos_embedding,
                     float* tokens, const ViTConfig* cfg);

// Legge i pesi dal file .bin secondo le dimensioni indicate in cfg.
// Ritorna 0 se ok, -1 se il file non si apre. Stampa un avviso se la dimensione
// del file non corrisponde a quella attesa per questo patch_size (probabile
// mismatch tra il modello caricato e la patch size dichiarata).
int read_ViT_weights(const char* path, const ViTConfig* cfg,
                      float* W_proj, float* b_proj, float* cls_token, float* pos_embedding,
                      block* b0, block* b1,
                      float final_norm_w[VIT_D_MODEL], float final_norm_b[VIT_D_MODEL],
                      float classifier_w[VIT_N_CLASSES][VIT_D_MODEL], float classifier_b[VIT_N_CLASSES]);

void classifier_into_logits(float classifier_w[VIT_N_CLASSES][VIT_D_MODEL], float classifier_b[VIT_N_CLASSES],
                             float cls_final_token[VIT_D_MODEL], float logits[VIT_N_CLASSES]);

// tokens: buffer flattened [seq_len][d_model], modificato in-place (residual incluso)
void transformer_block(float* tokens, const ViTConfig* cfg,
                       float* norm1_w, float* norm1_b,
                       float* in_proj_w, float* in_proj_b,
                       float* out_proj_w, float* out_proj_b,
                       float* norm2_w, float* norm2_b,
                       float* ff_w1, float* ff_b1,
                       float* ff_w2, float* ff_b2);

// ============================================================================
// API di alto livello: carica i pesi una sola volta per una data patch_size
// e permette di chiamare vit_predict() ripetutamente (utile per benchmark).
// ============================================================================

// Carica il modello da 'path' assumendo che sia stato addestrato con la patch
// size indicata. Ritorna 0 se ok, -1 in caso di errore (patch_size non valida
// per 28x28, o file non apribile).
int vit_load_model(ViTModel* model, const char* path, int patch_size);

int vit_predict(ViTModel* model, const float* image);

void vit_free_model(ViTModel* model);

#endif // TRANSFORMER_EMBEDDING_H

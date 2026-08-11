#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/transformer_embedding.h"
#ifdef _WIN32
    #include "../include/mmap_loader_windows.h"
#else // Posix
    #include "../include/mmap_loader_posix.h"
#endif

// ============================================================================
// Configurazione: calcola tutte le dimensioni derivate dalla patch size.
// d_model/n_heads/d_ff/n_classes sono fissi (architettura del backbone),
// solo l'embedding delle patch e la lunghezza di sequenza dipendono da patch_size.
// ============================================================================
int vit_config_init(ViTConfig* cfg, int patch_size) {
    if (!cfg) return -1;

    if (patch_size <= 0 || VIT_IMAGE_SIZE % patch_size != 0) {
        fprintf(stderr, "Error: patch_size=%d does not evenly divide %d "
                        "(valid values: 1, 2, 4, 7, 14, 28)\n", patch_size, VIT_IMAGE_SIZE);
        return -1;
    }

    cfg->image_size = VIT_IMAGE_SIZE;
    cfg->patch_size = patch_size;
    cfg->patches_per_side = VIT_IMAGE_SIZE / patch_size;
    cfg->num_patches = cfg->patches_per_side * cfg->patches_per_side;
    cfg->patch_dim = patch_size * patch_size;
    cfg->seq_len = cfg->num_patches + 1; // +1 per il token CLS
    cfg->d_model = VIT_D_MODEL;
    cfg->n_heads = VIT_N_HEADS;
    cfg->d_head = VIT_D_MODEL / VIT_N_HEADS;
    cfg->d_ff = VIT_D_FF;
    cfg->n_classes = VIT_N_CLASSES;

    return 0;
}

// image: buffer image_size*image_size (una sola immagine)
// patches: buffer preallocato num_patches*patch_dim (flattened)
void create_patches(const float* image, float* patches, const ViTConfig* cfg){
    int patch_idx = 0;

    for (int py = 0; py < cfg->patches_per_side; py++){
        for (int px = 0; px < cfg->patches_per_side; px++){
            int pixel_in_patch = 0;
            for (int y = 0; y < cfg->patch_size; y++){
                for (int x = 0; x < cfg->patch_size; x++){
                    int global_x = px * cfg->patch_size + x;
                    int global_y = py * cfg->patch_size + y;
                    int global_idx = global_y * cfg->image_size + global_x;

                    patches[patch_idx * cfg->patch_dim + pixel_in_patch] = image[global_idx];
                    pixel_in_patch++;
                }
            }
            patch_idx++;
        }
    }
}

//embedding = patch * W + b
// patches: [num_patches][patch_dim], W_proj: [d_model][patch_dim], embeddings: [num_patches][d_model]
void compute_embeddings(const float* patches, const float* W_proj, const float* b_proj,
                         float* embeddings, const ViTConfig* cfg){
    for (int i = 0; i < cfg->num_patches; i++){
        for (int j = 0; j < cfg->d_model; j++){
            float sum = b_proj[j];
            for (int k = 0; k < cfg->patch_dim; k++){
                sum += patches[i * cfg->patch_dim + k] * W_proj[j * cfg->patch_dim + k];
            }
            embeddings[i * cfg->d_model + j] = sum;
        }
    }
}

// embeddings: [num_patches][d_model], tokens: [seq_len][d_model] (output)
void prepare_tokens(const float* embeddings, const float* cls_token, const float* pos_embedding,
                     float* tokens, const ViTConfig* cfg){
    int d = cfg->d_model;

    // riga 0 = cls_token
    for (int j = 0; j < d; j++){
        tokens[j] = cls_token[j];
    }

    // righe 1..num_patches = embeddings
    for (int i = 0; i < cfg->num_patches; i++){
        for (int j = 0; j < d; j++){
            tokens[(i + 1) * d + j] = embeddings[i * d + j];
        }
    }

    //somma positional encoding a tutti i seq_len token
    for (int i = 0; i < cfg->seq_len; i++){
        for (int j = 0; j < d; j++){
            tokens[i * d + j] += pos_embedding[i * d + j];
        }
    }
}

int read_ViT_weights(const char* path, const ViTConfig* cfg,
                      float* W_proj, float* b_proj, float* cls_token, float* pos_embedding,
                      block* b0, block* b1,
                      float final_norm_w[VIT_D_MODEL], float final_norm_b[VIT_D_MODEL],
                      float classifier_w[VIT_N_CLASSES][VIT_D_MODEL], float classifier_b[VIT_N_CLASSES]){
    FILE* f = fopen(path, "rb");
    if (!f){
        fprintf(stderr, "Error opening %s\n", path);
        return -1;
    }

    // Controllo di coerenza: la dimensione del file deve corrispondere a quella
    // attesa per questa patch_size. Se non corrisponde, il modello è stato
    // probabilmente addestrato con una patch size diversa da quella dichiarata.
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t expected_floats =
        (size_t)cfg->d_model * cfg->patch_dim        // W_proj
        + (size_t)cfg->d_model                        // b_proj
        + (size_t)cfg->d_model                          // cls_token
        + (size_t)cfg->seq_len * cfg->d_model             // pos_embedding
        + 2 * (                                             // due blocchi transformer
              (size_t)cfg->d_model * 2                          // norm1 w+b
            + (size_t)(3 * cfg->d_model) * cfg->d_model          // attn in_proj_w
            + (size_t)(3 * cfg->d_model)                          // attn in_proj_b
            + (size_t)cfg->d_model * cfg->d_model                  // attn out_proj_w
            + (size_t)cfg->d_model                                  // attn out_proj_b
            + (size_t)cfg->d_model * 2                               // norm2 w+b
            + (size_t)cfg->d_ff * cfg->d_model                        // ff_w1
            + (size_t)cfg->d_ff                                        // ff_b1
            + (size_t)cfg->d_model * cfg->d_ff                          // ff_w2
            + (size_t)cfg->d_model                                       // ff_b2
          )
        + (size_t)cfg->d_model * 2                    // final_norm w+b
        + (size_t)cfg->n_classes * cfg->d_model         // classifier_w
        + (size_t)cfg->n_classes;                         // classifier_b

    long expected_bytes = (long)(expected_floats * sizeof(float));
    if (expected_bytes != file_size) {
        fprintf(stderr,
                "Warning: %s has %ld bytes, but %d were expected for patch_size=%ld.\n"
                "  Probable mismatch between loaded model and declared patch size:\n"
                "  verify that this file was trained/exported for the selected patch size.\n",
                path, file_size, cfg->patch_size, expected_bytes);
    }

    size_t got = 0;
    got += fread(W_proj, sizeof(float), (size_t)cfg->d_model * cfg->patch_dim, f);
    got += fread(b_proj, sizeof(float), cfg->d_model, f);
    got += fread(cls_token, sizeof(float), cfg->d_model, f);
    got += fread(pos_embedding, sizeof(float), (size_t)cfg->seq_len * cfg->d_model, f);

    // blocco 0
    got += fread(b0->b0_norm1_w,         sizeof(float), cfg->d_model,                    f);
    got += fread(b0->b0_norm1_b,         sizeof(float), cfg->d_model,                    f);
    got += fread(b0->b0_attn_in_proj_w,  sizeof(float), (size_t)(3*cfg->d_model)*cfg->d_model, f);
    got += fread(b0->b0_attn_in_proj_b,  sizeof(float), 3*cfg->d_model,                  f);
    got += fread(b0->b0_attn_out_proj_w, sizeof(float), (size_t)cfg->d_model*cfg->d_model, f);
    got += fread(b0->b0_attn_out_proj_b, sizeof(float), cfg->d_model,                    f);
    got += fread(b0->b0_norm2_w,         sizeof(float), cfg->d_model,                    f);
    got += fread(b0->b0_norm2_b,         sizeof(float), cfg->d_model,                    f);
    got += fread(b0->b0_ff_w1,           sizeof(float), (size_t)cfg->d_ff*cfg->d_model,   f);
    got += fread(b0->b0_ff_b1,           sizeof(float), cfg->d_ff,                        f);
    got += fread(b0->b0_ff_w2,           sizeof(float), (size_t)cfg->d_model*cfg->d_ff,   f);
    got += fread(b0->b0_ff_b2,           sizeof(float), cfg->d_model,                    f);

    // blocco 1
    got += fread(b1->b0_norm1_w,         sizeof(float), cfg->d_model,                    f);
    got += fread(b1->b0_norm1_b,         sizeof(float), cfg->d_model,                    f);
    got += fread(b1->b0_attn_in_proj_w,  sizeof(float), (size_t)(3*cfg->d_model)*cfg->d_model, f);
    got += fread(b1->b0_attn_in_proj_b,  sizeof(float), 3*cfg->d_model,                  f);
    got += fread(b1->b0_attn_out_proj_w, sizeof(float), (size_t)cfg->d_model*cfg->d_model, f);
    got += fread(b1->b0_attn_out_proj_b, sizeof(float), cfg->d_model,                    f);
    got += fread(b1->b0_norm2_w,         sizeof(float), cfg->d_model,                    f);
    got += fread(b1->b0_norm2_b,         sizeof(float), cfg->d_model,                    f);
    got += fread(b1->b0_ff_w1,           sizeof(float), (size_t)cfg->d_ff*cfg->d_model,   f);
    got += fread(b1->b0_ff_b1,           sizeof(float), cfg->d_ff,                        f);
    got += fread(b1->b0_ff_w2,           sizeof(float), (size_t)cfg->d_model*cfg->d_ff,   f);
    got += fread(b1->b0_ff_b2,           sizeof(float), cfg->d_model,                    f);

    // norma finale e classificatore
    got += fread(final_norm_w, sizeof(float), cfg->d_model,                    f);
    got += fread(final_norm_b, sizeof(float), cfg->d_model,                    f);
    got += fread(classifier_w, sizeof(float), (size_t)cfg->n_classes*cfg->d_model, f);
    got += fread(classifier_b, sizeof(float), cfg->n_classes,                  f);

    fclose(f);

    if (got != expected_floats) {
        fprintf(stderr, "Warning: read %zu floats, expected %zu (truncated file or unexpected format)\n",
                got, expected_floats);
    }

    return 0;
}

void classifier_into_logits(float classifier_w[VIT_N_CLASSES][VIT_D_MODEL], float classifier_b[VIT_N_CLASSES],
                             float cls_final_token[VIT_D_MODEL], float logits[VIT_N_CLASSES]){
    for (int i = 0; i < VIT_N_CLASSES; i++){
        logits[i] = classifier_b[i];
        for (int j = 0; j < VIT_D_MODEL; j++){
            logits[i] += classifier_w[i][j] * cls_final_token[j];
        }
    }
}

// ============================================================================
// API di alto livello: carica i pesi una sola volta (per una data patch size)
// e permette di chiamare vit_predict() ripetutamente senza rileggere il file
// (utile per benchmark). Stesso pattern di MnistModel/mnist_predict.
// ============================================================================

static block alloc_block(void){
    block b;
    b.b0_norm1_w         = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    b.b0_norm1_b         = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    b.b0_attn_in_proj_w  = (float*)malloc(sizeof(float) * (3 * VIT_D_MODEL) * VIT_D_MODEL);
    b.b0_attn_in_proj_b  = (float*)malloc(sizeof(float) * (3 * VIT_D_MODEL));
    b.b0_attn_out_proj_w = (float*)malloc(sizeof(float) * VIT_D_MODEL * VIT_D_MODEL);
    b.b0_attn_out_proj_b = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    b.b0_norm2_w         = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    b.b0_norm2_b         = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    b.b0_ff_w1           = (float*)malloc(sizeof(float) * VIT_D_FF * VIT_D_MODEL);
    b.b0_ff_b1           = (float*)malloc(sizeof(float) * VIT_D_FF);
    b.b0_ff_w2           = (float*)malloc(sizeof(float) * VIT_D_MODEL * VIT_D_FF);
    b.b0_ff_b2           = (float*)malloc(sizeof(float) * VIT_D_MODEL);
    return b;
}

static void free_block(block* b){
    free(b->b0_norm1_w);
    free(b->b0_norm1_b);
    free(b->b0_attn_in_proj_w);
    free(b->b0_attn_in_proj_b);
    free(b->b0_attn_out_proj_w);
    free(b->b0_attn_out_proj_b);
    free(b->b0_norm2_w);
    free(b->b0_norm2_b);
    free(b->b0_ff_w1);
    free(b->b0_ff_b1);
    free(b->b0_ff_w2);
    free(b->b0_ff_b2);
}

int vit_load_model(ViTModel* model, const char* path, int patch_size){
    if (!model) return -1;

    if (vit_config_init(&model->config, patch_size) != 0) {
        return -1;
    }
    ViTConfig* cfg = &model->config;

    model->W_proj = (float*)malloc(sizeof(float) * (size_t)cfg->d_model * cfg->patch_dim);
    model->b_proj = (float*)malloc(sizeof(float) * cfg->d_model);
    model->cls_token = (float*)malloc(sizeof(float) * cfg->d_model);
    model->pos_embedding = (float*)malloc(sizeof(float) * (size_t)cfg->seq_len * cfg->d_model);

    model->b0 = alloc_block();
    model->b1 = alloc_block();

    int rc = read_ViT_weights(path, cfg, model->W_proj, model->b_proj, model->cls_token, model->pos_embedding,
                               &model->b0, &model->b1,
                               model->final_norm_w, model->final_norm_b,
                               model->classifier_w, model->classifier_b);
    if (rc != 0) {
        vit_free_model(model);
        return -1;
    }

    printf("Succesfully loaded ViT model with patch size of %dx%d\n", patch_size, patch_size);

    return 0;
}

int vit_predict(ViTModel* model, const float* image){
    const ViTConfig* cfg = &model->config;

    float* patches = (float*)malloc(sizeof(float) * (size_t)cfg->num_patches * cfg->patch_dim);
    create_patches(image, patches, cfg);

    float* embeddings = (float*)malloc(sizeof(float) * (size_t)cfg->num_patches * cfg->d_model);
    compute_embeddings(patches, model->W_proj, model->b_proj, embeddings, cfg);

    float* tokens = (float*)malloc(sizeof(float) * (size_t)cfg->seq_len * cfg->d_model);
    prepare_tokens(embeddings, model->cls_token, model->pos_embedding, tokens, cfg);

    transformer_block(tokens, cfg,
                       model->b0.b0_norm1_w, model->b0.b0_norm1_b,
                       model->b0.b0_attn_in_proj_w, model->b0.b0_attn_in_proj_b,
                       model->b0.b0_attn_out_proj_w, model->b0.b0_attn_out_proj_b,
                       model->b0.b0_norm2_w, model->b0.b0_norm2_b,
                       model->b0.b0_ff_w1, model->b0.b0_ff_b1,
                       model->b0.b0_ff_w2, model->b0.b0_ff_b2);

    transformer_block(tokens, cfg,
                       model->b1.b0_norm1_w, model->b1.b0_norm1_b,
                       model->b1.b0_attn_in_proj_w, model->b1.b0_attn_in_proj_b,
                       model->b1.b0_attn_out_proj_w, model->b1.b0_attn_out_proj_b,
                       model->b1.b0_norm2_w, model->b1.b0_norm2_b,
                       model->b1.b0_ff_w1, model->b1.b0_ff_b1,
                       model->b1.b0_ff_w2, model->b1.b0_ff_b2);

    // Final norm sul token CLS (riga 0 = tokens[0..d_model-1])
    float cls_final[VIT_D_MODEL];
    layernorm_1d(&tokens[0], cls_final, model->final_norm_w, model->final_norm_b, cfg->d_model);

    float logits[VIT_N_CLASSES];
    classifier_into_logits(model->classifier_w, model->classifier_b, cls_final, logits);

    int predicted = argmax(logits, cfg->n_classes);

    free(patches);
    free(embeddings);
    free(tokens);

    return predicted;
}

void vit_free_model(ViTModel* model){
    if (!model) return;
    free(model->W_proj);
    free(model->b_proj);
    free(model->cls_token);
    free(model->pos_embedding);
    free_block(&model->b0);
    free_block(&model->b1);
    model->W_proj = NULL;
    model->b_proj = NULL;
    model->cls_token = NULL;
    model->pos_embedding = NULL;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/mnist.h"
#include "../include/ops.h"
#include "../include/transformer_embedding.h"
#include "../include/npy_io.h"
#include "../include/benchmark.h"

void predict_mnist() {
    // Carico il modello
    MnistModel model;
    mnist_load_model(&model, "models/CNN_mnist/model.bin");

    // Carico l'immagine
    int num_elements;
    float* images = read_npy_float("models/mnist_test_images_and_labels/test_images.npy", &num_elements);
    if (!images){
        fprintf(stderr, "Error during images loading\n");
        exit(1);
    }
    printf("Number of images: %d\n", num_elements / (28*28));

    int result_list[10] = {7, 2, 1, 0, 5, 1, 4, 9, 5, 9};
    printf("Select an image index (0-%d): ", num_elements / (28*28) - 1);
    int idx;
    scanf("%d", &idx);

    // Do l'immagine al modello e provo la predizione
    int predicted = mnist_predict(&model, images + (size_t)idx * 28 * 28);
    printf("Expected digit: %d | Predicted digit: %d\n", result_list[idx], predicted);

    free(images);
    mnist_free_model(&model);
}

void predict_ViT(int patch_size) {
    char path_string[32];
    snprintf(path_string, sizeof(path_string), "models/ViT_mnist_%dx%d/model.bin", patch_size, patch_size);

    // Carico il modello ViT una sola volta (stesso pattern della CNN)
    ViTModel model;
    if (vit_load_model(&model, path_string, patch_size) != 0) {
        fprintf(stderr, "Error during ViT model loading\n");
        exit(1);
    }

    int num_elements;
    float* images = read_npy_float("models/mnist_test_images_and_labels/test_images.npy", &num_elements);
    if (!images){
        fprintf(stderr, "Error during images loading\n");
        exit(1);
    }

    printf("Number of images: %d\n", num_elements / (28*28));

    printf("Select an image index (0-%d): ", num_elements / (28 * 28) - 1);
    int idx;
    scanf("%d", &idx);

    int result_list[10] = {7, 2, 1, 0, 5, 1, 4, 9, 5, 9};
    int predicted = vit_predict(&model, images + (size_t)idx * 28 * 28);
    printf("Expected digit: %d | Predicted digit: %d\n", result_list[idx], predicted);

    vit_free_model(&model);
    free(images);
}

int main(){
    printf("\n--------------------------------\n\n");
    printf("Choose an option:\n");
    printf("1. CNN - single prediction\n");
    printf("2. ViT Transformer - single prediction (select patch size)\n");
    printf("3. Benchmark - compare CNN vs ViT variants (accuracy, latency, throughput)\n");
    printf("Choice: ");
    int scelta;
    scanf("%d", &scelta);
    printf("\n--------------------------------\n\n");

    if (scelta == 1){
        predict_mnist();
    }
    else if (scelta == 2){
        printf("Select the patch size (2/4/7/14): ");
        int chosen_patch_size;
        scanf("%d", &chosen_patch_size);
        predict_ViT(chosen_patch_size);
    }
    else if (scelta == 3){
        run_benchmark_comparison();
    }
    else{
        fprintf(stderr, "Invalid option\n");
        return 1;
    }
    return 0;
}

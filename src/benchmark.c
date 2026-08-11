#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/benchmark.h"
#include "../include/mnist.h"
#include "../include/ops.h"
#include "../include/transformer_embedding.h"
#include "../include/npy_io.h"

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #define MKDIR(path) _mkdir(path)
    #define RMDIR(path) _rmdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
    #define MKDIR(path) mkdir(path, 0755)
    #define RMDIR(path) rmdir(path)
#endif

// Timer portabile ad alta risoluzione
#ifdef _WIN32
static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

static void compute_latency_stats(double* latencies, int n, double* out_sum, double* out_min, double* out_max) {
    double sum = 0.0, min_l = latencies[0], max_l = latencies[0];
    for (int i = 0; i < n; i++) {
        sum += latencies[i];
        if (latencies[i] < min_l) min_l = latencies[i];
        if (latencies[i] > max_l) max_l = latencies[i];
    }
    *out_sum = sum;
    *out_min = min_l;
    *out_max = max_l;
}

static int ends_with(const char* s, const char* suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    if (m > n) return 0;
    return strcmp(s + (n - m), suffix) == 0;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static float* load_images_for_benchmark(const char* path, int* num_img_elements) {
    if (ends_with(path, ".npy")) return read_npy_float(path, num_img_elements);
    return read_idx_images_float(path, num_img_elements);
}

static int* load_labels_for_benchmark(const char* path, int* num_labels) {
    if (!path) {
        *num_labels = 0;
        return NULL;
    }
    if (ends_with(path, ".npy")) return read_npy_int_labels(path, num_labels);
    return read_idx_labels(path, num_labels);
}

static int run_shell_command(const char* cmd) {
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "Command failed (%d): %s\n", rc, cmd);
        return -1;
    }
    return 0;
}

static int prepare_full_mnist_dataset(char* out_images_path, size_t images_path_size,
                                      char* out_labels_path, size_t labels_path_size) {
    const char* tmp_dir = "results/mnist_tmp";
    const char* idx_images = "results/mnist_tmp/t10k-images-idx3-ubyte";
    const char* idx_labels = "results/mnist_tmp/t10k-labels-idx1-ubyte";

    MKDIR("results");
    MKDIR(tmp_dir);

    printf("\nBenchmark dataset: automatic download of full MNIST test split (10000 images)...\n");
    if (run_shell_command("curl -L \"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz\" -o \"results/mnist_tmp/t10k-images-idx3-ubyte.gz\"") != 0) return -1;
    if (run_shell_command("curl -L \"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz\" -o \"results/mnist_tmp/t10k-labels-idx1-ubyte.gz\"") != 0) return -1;
    if (run_shell_command("gzip -d -f \"results/mnist_tmp/t10k-images-idx3-ubyte.gz\"") != 0) return -1;
    if (run_shell_command("gzip -d -f \"results/mnist_tmp/t10k-labels-idx1-ubyte.gz\"") != 0) return -1;

    if (!file_exists(idx_images) || !file_exists(idx_labels)) {
        fprintf(stderr, "Download completed but IDX files were not found in %s\n", tmp_dir);
        return -1;
    }

    snprintf(out_images_path, images_path_size, "%s", idx_images);
    snprintf(out_labels_path, labels_path_size, "%s", idx_labels);
    return 0;
}

static void cleanup_temporary_dataset(void) {
    remove("results/mnist_tmp/t10k-images-idx3-ubyte.gz");
    remove("results/mnist_tmp/t10k-labels-idx1-ubyte.gz");
    remove("results/mnist_tmp/t10k-images-idx3-ubyte");
    remove("results/mnist_tmp/t10k-labels-idx1-ubyte");
    RMDIR("results/mnist_tmp");
    printf("\nTemporary dataset cleanup completed.\n");
}

void benchmark_cnn(const char* model_path, const char* images_path, const char* labels_path,
                    int max_samples, BenchmarkResult* result) {
    memset(result, 0, sizeof(BenchmarkResult));
    snprintf(result->model_name, sizeof(result->model_name), "CNN");
    result->patch_size = 0;

    MnistModel model;
    if (mnist_load_model(&model, model_path) != 0) {
        fprintf(stderr, "Unable to load CNN model from %s\n", model_path);
        return;
    }

    int num_img_elements = 0;
    float* images = load_images_for_benchmark(images_path, &num_img_elements);
    if (!images) {
        mnist_free_model(&model);
        return;
    }

    int num_labels = 0;
    int* labels = load_labels_for_benchmark(labels_path, &num_labels);

    int total_images = num_img_elements / (28 * 28);
    int n = total_images;
    if (max_samples > 0 && max_samples < n) n = max_samples;
    if (labels && num_labels < n) n = num_labels;
    if (n <= 0) {
        fprintf(stderr, "No images available for CNN benchmark\n");
        free(images);
        free(labels);
        mnist_free_model(&model);
        return;
    }

    double* latencies = (double*)malloc(n * sizeof(double));
    int correct = 0;

    double t_start = get_time_sec();
    for (int i = 0; i < n; i++) {
        double t0 = get_time_sec();
        int pred = mnist_predict(&model, images + (size_t)i * 28 * 28);
        double t1 = get_time_sec();
        latencies[i] = (t1 - t0) * 1000.0; // ms
        if (labels && pred == labels[i]) correct++;
    }
    double t_end = get_time_sec();

    double sum, min_l, max_l;
    compute_latency_stats(latencies, n, &sum, &min_l, &max_l);

    result->num_samples = n;
    result->correct = correct;
    result->accuracy = labels ? (100.0 * correct / n) : -1.0;
    result->total_time_sec = t_end - t_start;
    result->avg_latency_ms = sum / n;
    result->min_latency_ms = min_l;
    result->max_latency_ms = max_l;
    result->throughput_img_per_sec = n / result->total_time_sec;

    free(latencies);
    free(images);
    free(labels);
    mnist_free_model(&model);
}

void benchmark_vit(const char* model_path, int patch_size,
                    const char* images_path, const char* labels_path,
                    int max_samples, BenchmarkResult* result) {
    memset(result, 0, sizeof(BenchmarkResult));
    snprintf(result->model_name, sizeof(result->model_name), "ViT %dx%d", patch_size, patch_size);
    result->patch_size = patch_size;

    ViTModel model;
    if (vit_load_model(&model, model_path, patch_size) != 0) {
        fprintf(stderr, "Unable to load ViT model from %s (patch_size=%d)\n", model_path, patch_size);
        return;
    }

    int num_img_elements = 0;
    float* images = load_images_for_benchmark(images_path, &num_img_elements);
    if (!images) {
        vit_free_model(&model);
        return;
    }

    int num_labels = 0;
    int* labels = load_labels_for_benchmark(labels_path, &num_labels);

    int total_images = num_img_elements / (28 * 28);
    int n = total_images;
    if (max_samples > 0 && max_samples < n) n = max_samples;
    if (labels && num_labels < n) n = num_labels;
    if (n <= 0) {
        fprintf(stderr, "No images available for ViT %dx%d benchmark\n", patch_size, patch_size);
        free(images);
        free(labels);
        vit_free_model(&model);
        return;
    }

    double* latencies = (double*)malloc(n * sizeof(double));
    int correct = 0;

    double t_start = get_time_sec();
    for (int i = 0; i < n; i++) {
        double t0 = get_time_sec();
        int pred = vit_predict(&model, images + (size_t)i * 28 * 28);
        double t1 = get_time_sec();
        latencies[i] = (t1 - t0) * 1000.0; // ms
        if (labels && pred == labels[i]) correct++;
    }
    double t_end = get_time_sec();

    double sum, min_l, max_l;
    compute_latency_stats(latencies, n, &sum, &min_l, &max_l);

    result->num_samples = n;
    result->correct = correct;
    result->accuracy = labels ? (100.0 * correct / n) : -1.0;
    result->total_time_sec = t_end - t_start;
    result->avg_latency_ms = sum / n;
    result->min_latency_ms = min_l;
    result->max_latency_ms = max_l;
    result->throughput_img_per_sec = n / result->total_time_sec;

    free(latencies);
    free(images);
    free(labels);
    vit_free_model(&model);
}

void print_benchmark_result(const BenchmarkResult* r) {
    printf("\n--- %s ---\n", r->model_name);
    printf("  Samples:         %d\n", r->num_samples);
    if (r->accuracy >= 0)
        printf("  Accuracy:        %.2f%% (%d/%d)\n", r->accuracy, r->correct, r->num_samples);
    else
        printf("  Accuracy:        n/a (no labels found)\n");
    printf("  Total time:      %.4f s\n", r->total_time_sec);
    printf("  Avg latency:     %.4f ms/image\n", r->avg_latency_ms);
    printf("  Min/max latency: %.4f / %.4f ms\n", r->min_latency_ms, r->max_latency_ms);
    printf("  Throughput:      %.2f images/s\n", r->throughput_img_per_sec);
}

void save_benchmark_csv(const char* csv_path, BenchmarkResult* results, int num_results) {
    MKDIR("results"); // ignora l'errore se la cartella esiste già

    FILE* f = fopen(csv_path, "w");
    if (!f) {
        fprintf(stderr, "Unable to write CSV file: %s\n", csv_path);
        return;
    }
    fprintf(f, "model,patch_size,num_samples,correct,accuracy_pct,total_time_sec,avg_latency_ms,min_latency_ms,max_latency_ms,throughput_img_per_sec\n");
    for (int i = 0; i < num_results; i++) {
        BenchmarkResult* r = &results[i];
        fprintf(f, "%s,%d,%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                r->model_name, r->patch_size, r->num_samples, r->correct, r->accuracy,
                r->total_time_sec, r->avg_latency_ms, r->min_latency_ms,
                r->max_latency_ms, r->throughput_img_per_sec);
    }
    fclose(f);
    printf("\nResults saved to: %s\n", csv_path);
}

void run_benchmark_comparison(void) {
    printf("\n--------------------------------\n");
    printf("Comparative benchmark: CNN vs N ViT variants\n");
    printf("--------------------------------\n");

    printf("Choose the models to test: \n");
    printf("1. CNN\n");
    printf("2. ViT 2x2\n");
    printf("3. ViT 4x4\n");
    printf("4. ViT 7x7\n");
    printf("5. ViT 14x14\n");
    printf("Please choose in this format (N-N-N): ");
    char choice[12];
    if (scanf("%11s", choice) != 1) {
        printf("Error during benchmark models selection\n");
        return;
    }

    if (strcmp(choice, "") == 0 || strlen(choice) < 2) {
        printf("Error during choice parsing, expected valid format\n");
        return;
    }

    int total = (strlen(choice) + 1) / 2;

    BenchmarkResult* results = (BenchmarkResult*)malloc(sizeof(BenchmarkResult) * total);
    if (!results) {
        fprintf(stderr, "Benchmark results allocation error\n");
        return;
    }

    char images_path[256];  
    char labels_path[256];
    if (prepare_full_mnist_dataset(images_path, sizeof(images_path),
                                   labels_path, sizeof(labels_path)) != 0) {
        fprintf(stderr, "Unable to automatically prepare the full MNIST dataset.\n");
        fprintf(stderr, "Check internet connection and curl/gzip availability.\n");
        free(results);
        return;
    }

    int max_samples;
    printf("\nSelect the max_samples (0 = all the validation set [10.000 images]): ");
    scanf("%d", &max_samples);

    int count = 0;

    char* token = strtok(choice, "-");
    while (token != NULL) {
        if (atoi(token) == 1) {
            printf("\nRunning CNN benchmark...\n");
            benchmark_cnn("models/CNN_mnist/model.bin", images_path, labels_path, max_samples, &results[count]);
            print_benchmark_result(&results[count++]);
            token = strtok(NULL, "-");
        } if (atoi(token) == 2) {
            printf("\nRunning ViT 2x2 benchmark...\n");
            benchmark_vit("models/ViT_mnist_2x2/model.bin", 2, images_path, labels_path, max_samples, &results[count]);
            print_benchmark_result(&results[count++]);
            token = strtok(NULL, "-");
        } if (atoi(token) == 3) {
            printf("\nRunning ViT 4x4 benchmark...\n");
            benchmark_vit("models/ViT_mnist_4x4/model.bin", 4, images_path, labels_path, max_samples, &results[count]);
            print_benchmark_result(&results[count++]);
            token = strtok(NULL, "-");
        } if (atoi(token) == 4) {
            printf("\nRunning ViT 7x7 benchmark...\n");
            benchmark_vit("models/ViT_mnist_7x7/model.bin", 7, images_path, labels_path, max_samples, &results[count]);
            print_benchmark_result(&results[count++]);
            token = strtok(NULL, "-");
        } if (atoi(token) == 5) {
            printf("\nRunning ViT 14x14 benchmark...\n");
            benchmark_vit("models/ViT_mnist_14x14/model.bin", 14, images_path, labels_path, max_samples, &results[count]);
            print_benchmark_result(&results[count++]);
            token = strtok(NULL, "-");
        } if (atoi(token) == 0) {
            break;
        } if (atoi(token) < 1 || atoi(token) > 5) {
            printf("Error during benchmark calculation, invalid choice entered: %d\n", atoi(token));
            return;
        }
    }
    printf("\n================================\n");
    printf("Benchmark calculation ended\n");
    printf("================================\n");

    printf("\n================================\n");
    printf("Comparative summary\n");
    printf("================================\n");
    printf("%-10s %7s %10s %14s %18s %14s\n", "Model", "Patch", "Accuracy", "Avg lat(ms)", "Throughput(img/s)", "Tot time(s)");
    for (int i = 0; i < total; i++) {
        if (results[i].patch_size > 0) {
            printf("%-10s %7d %9.2f%% %14.4f %18.2f %14.4f\n",
               results[i].model_name, results[i].patch_size, results[i].accuracy, results[i].avg_latency_ms,
               results[i].throughput_img_per_sec, results[i].total_time_sec);
        } else {
            printf("%-10s %7s %9.2f%% %14.4f %18.2f %14.4f\n",
               results[i].model_name, "-", results[i].accuracy, results[i].avg_latency_ms,
               results[i].throughput_img_per_sec, results[i].total_time_sec);
        }
    }

    save_benchmark_csv("results/benchmark_results.csv", results, total);
    free(results);
    cleanup_temporary_dataset();
}
#ifndef BENCHMARK_H
#define BENCHMARK_H

typedef struct {
    char model_name[32];   // es. "CNN", "ViT 4x4", "ViT 7x7"
    int patch_size;         // 0 per la CNN (non applicabile)
    int num_samples;
    int correct;
    double accuracy;              // percentuale; -1 se non ci sono etichette
    double total_time_sec;
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    double throughput_img_per_sec;
} BenchmarkResult;

// Esegue l'inferenza su (fino a) max_samples immagini con il modello CNN.
// max_samples <= 0 significa "tutte le immagini disponibili".
void benchmark_cnn(const char* model_path, const char* images_path, const char* labels_path,
                    int max_samples, BenchmarkResult* result);

// Come sopra, ma per un modello ViT caricato con la patch_size indicata.
// Il file in model_path deve essere stato addestrato con quella patch_size.
void benchmark_vit(const char* model_path, int patch_size,
                    const char* images_path, const char* labels_path,
                    int max_samples, BenchmarkResult* result);

void print_benchmark_result(const BenchmarkResult* result);
void save_benchmark_csv(const char* csv_path, BenchmarkResult* results, int num_results);

// Flusso interattivo completo: chiede quanti campioni testare, quante varianti
// ViT confrontare (percorso + patch size per ciascuna), esegue tutti i benchmark,
// stampa la tabella comparativa e salva il CSV in results/.
void run_benchmark_comparison(void);

#endif // BENCHMARK_H

# Neural Inference Engine for MNIST Research

A C-based inference engine for **MNIST digit classification** used to compare two architectures under the same runtime pipeline:
- **CNN** (convolutional baseline)
- **ViT** (Vision Transformer baseline)

The project is designed as a small, reproducible research setup for latency/throughput/accuracy comparison on identical inputs.

## Research Scope

- Unified inference runtime in C for both models
- Cross-platform execution (Windows/Linux/macOS)
- Reproducible benchmark flow with CSV export
- Side-by-side comparison:
  - Accuracy
  - Average latency
  - Throughput
  - Total execution time

## Requirements

Minimum required tools:
- **GCC** (or compatible C compiler)
- **Make**
- **curl** (for model download)

Optional:
- Python 3.11+ (only for utility scripts in `src/tools/`)

## Project Structure

```
neural-inference-engine/
├── include/                # Header files
├── models/                 # Model weights (downloaded separately)
├── src/
│   ├── core/               # Memory-mapped loaders (POSIX & Windows) and tensor structure
│   ├── mnist_model/        # CNN model: load, predict, free
│   ├── ops/                # Math operations (matmul, softmax, conv, gelu, etc.)
│   ├── tools/              # Python utilities (image viewer)
│   ├── utils/              # NumPy I/O helpers
│   ├── visual_transformer/ # ViT: patch embedding, transformer blocks, attention
│   └── main.c              # Entry point and menu
├── tests/
├── Makefile
└── README.md
```

## Models

| Model | Description | Patch size | Repository |
|-------|-------------|------------|------------|
| MNIST CNN | Convolutional digit recognition |-----| [sbolba/mnist-model](https://huggingface.co/sbolba/CNN_mnist) |
| MNIST ViT 4x4 | Vision Transformer digit recognition | 4x4 | [sbolba/ViT_mnist](https://huggingface.co/sbolba/ViT_mnist_4x4) |
| MNIST ViT 2x2 | Vision Transformer digit recognition | 2x2 | [sbolba/ViT_mnist](https://huggingface.co/sbolba/ViT_mnist_2x2) |
| MNIST ViT 7x7 | Vision Transformer digit recognition | 7x7 | [sbolba/ViT_mnist](https://huggingface.co/sbolba/ViT_mnist_7x7) |
| MNIST ViT 14x14 | Vision Transformer digit recognition | 14x14 | [sbolba/ViT_mnist](https://huggingface.co/sbolba/ViT_mnist_14x14) |

### Downloading model weights (`model.bin`)

From the project root (the original images size is 28x28):

```bash
curl -L "https://huggingface.co/sbolba/CNN_mnist/resolve/main/model.bin" -o models/CNN_mnist/model.bin
curl -L "https://huggingface.co/sbolba/ViT_mnist_4x4/resolve/main/model.bin" -o models/ViT_mnist_4x4/model.bin

curl -L "https://huggingface.co/sbolba/ViT_mnist_2x2/resolve/main/model.bin" -o models/ViT_mnist_2x2/model.bin
curl -L "https://huggingface.co/sbolba/ViT_mnist_7x7/resolve/main/model.bin" -o models/ViT_mnist_7x7/model.bin
curl -L "https://huggingface.co/sbolba/ViT_mnist_14x14/resolve/main/model.bin" -o models/ViT_mnist_14x14/model.bin
```

You also need:
- `models/mnist/test_images.npy`
- `models/mnist/test_labels.npy`

### Full MNIST benchmark is automatic

Benchmark option `3`:
- downloads the full MNIST test split (`t10k`, 10000 images) into a temporary folder,
- runs CNN and one or more ViT variants on that dataset,
- deletes temporary dataset files at the end of the benchmark.

Requirements for this flow:
- internet connection
- `curl`
- `gzip` (or `gunzip`)

## Build & Run

Default behavior:

```bash
make clean && make
```

This now:
1. cleans previous artifacts,
2. builds the executable,
3. starts the interactive menu automatically.

Other useful targets:

```bash
make              # Build + run (default)
make all          # Same as above
make release      # Build only
make debug        # Debug build with symbols
make fast         # Aggressive optimizations + LTO
make run          # Build + run
make clean        # Remove build artifacts
make info         # Show build configuration
make help         # Show available targets
```

## Runtime Menu

At launch the program offers:
1. CNN single prediction
2. ViT single prediction (you provide model path and patch size)
3. Comparative benchmark: CNN vs N ViT variants

Why `model.bin` is requested for ViT:
- each ViT variant (different patch size and/or training run) can have a different `model.bin`,
- the benchmark can compare multiple variants in one run, so the path is requested per variant.

## Benchmark Output (Interpretation)

For each model/variant, the benchmark reports:
- samples
- accuracy
- total time
- average/min/max latency
- throughput

CSV output is saved to:
- `results/benchmark_results.csv`

The CSV includes:
- `model`
- `patch_size`
- `num_samples`
- `correct`
- `accuracy_pct`
- `total_time_sec`
- `avg_latency_ms`
- `min_latency_ms`
- `max_latency_ms`
- `throughput_img_per_sec`

`max_samples = 0` means "use all available samples" from the full MNIST test split (10000 images).

## Input Format

The engine expects **28×28 grayscale images**:
- `.npy` for local sample inference paths
- MNIST IDX (`t10k-*.ubyte`) for full automatic benchmark

## Cross-Platform Memory Mapping

The CNN path uses memory mapping for fast weight loading:
- POSIX: `mmap`
- Windows: `CreateFileMapping` + `MapViewOfFile`

The proper implementation is selected automatically in `Makefile`.

## MNIST Inference Benchmark

| Model | Patch Size | Accuracy | Avg Latency (ms) | Throughput (img/s) | Total Execution Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN** | N/A | **98.90%** | 13.3901 | 74.68 | 133.9031 |
| **ViT 2x2** | 2 | 97.67% | 7.3981 | 135.17 | 73.9820 |
| **ViT 4x4** | 4 | 94.77% | 1.1880 | 841.71 | 11.8806 |
| **ViT 7x7** | 7 | 97.79% | 0.4528 | 2208.50 | 4.5280 |
| **ViT 14x14** | 14 | 97.92% | **0.1305** | **7660.49** | **1.3054** |

## License

This project is provided as-is for educational and research purposes.
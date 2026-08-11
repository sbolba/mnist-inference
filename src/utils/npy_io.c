#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../include/npy_io.h"

static uint32_t read_be_u32(FILE* f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

float* read_npy_float(const char* path, int* num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file: %s\n", path);
        *num_elements = 0;
        return NULL;
    }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); *num_elements = 0; return NULL; }

    unsigned short header_len;
    fread(&header_len, 2, 1, f);

    char* header = (char*)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    long data_offset = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, data_offset, SEEK_SET);

    long data_size = file_size - data_offset;
    *num_elements = (int)(data_size / sizeof(float));

    float* data = (float*)malloc(data_size);
    size_t read_count = fread(data, sizeof(float), *num_elements, f);
    if ((int)read_count != *num_elements) {
        fprintf(stderr, "Warning: read fewer elements than expected from %s\n", path);
    }

    free(header);
    fclose(f);
    return data;
}

// Estrae il codice dtype (es. "<i8", "<f4", "|u1") dall'header .npy,
// cercando il campo 'descr': '<...>'
static void parse_dtype(const char* header, char* dtype_out, size_t out_size) {
    dtype_out[0] = '\0';

    const char* key = strstr(header, "descr");
    if (!key) return;

    const char* colon = strchr(key, ':');
    if (!colon) return;

    const char* q1 = strchr(colon, '\'');
    if (!q1) return;

    const char* q2 = strchr(q1 + 1, '\'');
    if (!q2) return;

    size_t len = (size_t)(q2 - (q1 + 1));
    if (len >= out_size) len = out_size - 1;

    memcpy(dtype_out, q1 + 1, len);
    dtype_out[len] = '\0';
}

int* read_npy_int_labels(const char* path, int* num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file: %s\n", path);
        *num_elements = 0;
        return NULL;
    }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); *num_elements = 0; return NULL; }

    unsigned char version_major = magic[6];
    unsigned int header_len;
    if (version_major >= 2) {
        uint32_t hl = 0;
        fread(&hl, 4, 1, f);
        header_len = hl;
    } else {
        uint16_t hl = 0;
        fread(&hl, 2, 1, f);
        header_len = hl;
    }

    char* header = (char*)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    char dtype[16];
    parse_dtype(header, dtype, sizeof(dtype));

    long data_offset = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, data_offset, SEEK_SET);
    long data_size = file_size - data_offset;

    int* labels = NULL;
    int n = 0;

    if (strstr(dtype, "i8")) {
        n = (int)(data_size / sizeof(int64_t));
        int64_t* raw = (int64_t*)malloc(data_size);
        fread(raw, sizeof(int64_t), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    } else if (strstr(dtype, "i4")) {
        n = (int)(data_size / sizeof(int32_t));
        int32_t* raw = (int32_t*)malloc(data_size);
        fread(raw, sizeof(int32_t), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    } else if (strstr(dtype, "u1") || strstr(dtype, "i1") || strstr(dtype, "b1")) {
        n = (int)(data_size / sizeof(uint8_t));
        uint8_t* raw = (uint8_t*)malloc(data_size);
        fread(raw, sizeof(uint8_t), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    } else if (strstr(dtype, "f4")) {
        n = (int)(data_size / sizeof(float));
        float* raw = (float*)malloc(data_size);
        fread(raw, sizeof(float), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    } else if (strstr(dtype, "f8")) {
        n = (int)(data_size / sizeof(double));
        double* raw = (double*)malloc(data_size);
        fread(raw, sizeof(double), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    } else {
        // Fallback: la maggior parte dei tool salva le label MNIST come int64
        fprintf(stderr, "Warning: unrecognized dtype '%s' in %s, assuming int64\n", dtype, path);
        n = (int)(data_size / sizeof(int64_t));
        int64_t* raw = (int64_t*)malloc(data_size);
        fread(raw, sizeof(int64_t), n, f);
        labels = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) labels[i] = (int)raw[i];
        free(raw);
    }

    free(header);
    fclose(f);
    *num_elements = n;
    return labels;
}

float* read_idx_images_float(const char* path, int* num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error opening IDX images file: %s\n", path);
        *num_elements = 0;
        return NULL;
    }

    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);
    uint32_t rows = read_be_u32(f);
    uint32_t cols = read_be_u32(f);

    if (magic != 2051U) {
        fprintf(stderr, "Invalid IDX images magic in %s (got %u, expected 2051)\n", path, magic);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }
    if (count == 0 || rows == 0 || cols == 0) {
        fprintf(stderr, "Invalid IDX images shape in %s (%u, %u, %u)\n", path, count, rows, cols);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    size_t total = (size_t)count * (size_t)rows * (size_t)cols;
    if (total > (size_t)INT32_MAX) {
        fprintf(stderr, "IDX images too large for int element count: %s\n", path);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    uint8_t* raw = (uint8_t*)malloc(total);
    if (!raw) {
        fprintf(stderr, "Out of memory while reading %s\n", path);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }
    if (fread(raw, 1, total, f) != total) {
        fprintf(stderr, "Unexpected EOF while reading %s\n", path);
        free(raw);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    float* data = (float*)malloc(total * sizeof(float));
    if (!data) {
        fprintf(stderr, "Out of memory while converting %s\n", path);
        free(raw);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    for (size_t i = 0; i < total; i++) {
        data[i] = (((float)raw[i] / 255.0f) - 0.1307f) / 0.3081f;
    }

    free(raw);
    fclose(f);
    *num_elements = (int)total;
    return data;
}

int* read_idx_labels(const char* path, int* num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error opening IDX labels file: %s\n", path);
        *num_elements = 0;
        return NULL;
    }

    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);
    if (magic != 2049U) {
        fprintf(stderr, "Invalid IDX labels magic in %s (got %u, expected 2049)\n", path, magic);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }
    if (count == 0 || count > (uint32_t)INT32_MAX) {
        fprintf(stderr, "Invalid IDX labels count in %s (%u)\n", path, count);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    uint8_t* raw = (uint8_t*)malloc((size_t)count);
    if (!raw) {
        fprintf(stderr, "Out of memory while reading %s\n", path);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }
    if (fread(raw, 1, count, f) != count) {
        fprintf(stderr, "Unexpected EOF while reading %s\n", path);
        free(raw);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }

    int* labels = (int*)malloc((size_t)count * sizeof(int));
    if (!labels) {
        fprintf(stderr, "Out of memory while converting %s\n", path);
        free(raw);
        fclose(f);
        *num_elements = 0;
        return NULL;
    }
    for (uint32_t i = 0; i < count; i++) {
        labels[i] = (int)raw[i];
    }

    free(raw);
    fclose(f);
    *num_elements = (int)count;
    return labels;
}

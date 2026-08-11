#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/mmap_loader_windows.h"
//windows headers
#include <windows.h>

MmapFile* mmap_open(const char* filepath) {
    MmapFile* mf = (MmapFile*)malloc(sizeof(MmapFile));
    if (!mf) return NULL;

    // --- IMPLEMENTAZIONE WINDOWS ---

    // 1. Apri il file
    mf->h_file = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (mf->h_file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        free(mf);
        return NULL;
    }

    // 2. Ottieni dimensione
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(mf->h_file, &fileSize)) {
        CloseHandle(mf->h_file);
        free(mf);
        return NULL;
    }
    mf->size = (size_t)fileSize.QuadPart;

    // 3. Crea oggetto mapping
    mf->h_map = CreateFileMappingA(mf->h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mf->h_map) {
        CloseHandle(mf->h_file);
        free(mf);
        return NULL;
    }

    // 4. Mappa la vista in memoria
    mf->data = MapViewOfFile(mf->h_map, FILE_MAP_READ, 0, 0, 0);
    if (!mf->data) {
        CloseHandle(mf->h_map);
        CloseHandle(mf->h_file);
        free(mf);
        return NULL;
    }

    return mf;
}

void mmap_close(MmapFile* mf) {
    if (!mf) return;

    if (mf->data) UnmapViewOfFile(mf->data);
    if (mf->h_map) CloseHandle(mf->h_map);
    if (mf->h_file) CloseHandle(mf->h_file);

    free(mf);
}

Tensor* mmap_read_tensor(MmapFile* mf, size_t offset, int32_t* shape, int32_t ndim) {
    if (!mf || !mf->data) return NULL;
    
    // Calcola dimensione richiesta in byte
    size_t num_elements = tensor_compute_size(shape, ndim);
    size_t byte_size = num_elements * sizeof(float);
    
    // Controlla bounds (importante per evitare crash!)
    if (offset + byte_size > mf->size) {
        fprintf(stderr, "Error: mmap read out of bounds. File size: %zu, Request end: %zu\n", mf->size, offset + byte_size);
        return NULL;
    }

    // Calcola puntatore: base + offset
    // NOTA: Assumiamo che i dati nel file siano float32 raw (Little Endian)
    float* data_ptr = (float*)((char*)mf->data + offset);

    // Crea tensore che NON possiede i dati (owns_data=0)
    // Non allochiamo nuova memoria per i dati, usiamo quella mappata.
    return tensor_from_data(data_ptr, shape, ndim, 0);
}

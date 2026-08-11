#ifndef _WIN32

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/mmap_loader_posix.h"
//posix headers
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MmapFile* mmap_open(const char* filepath) {
    MmapFile* mf = (MmapFile*)malloc(sizeof(MmapFile));
    if (!mf) return NULL;

    // --- IMPLEMENTAZIONE POSIX (Linux/Mac) ---
    
    mf->fd = open(filepath, O_RDONLY);
    if (mf->fd == -1) {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        free(mf);
        return NULL;
    }

    struct stat sb;
    if (fstat(mf->fd, &sb) == -1) {
        close(mf->fd);
        free(mf);
        return NULL;
    }
    mf->size = sb.st_size;

    mf->data = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->data == MAP_FAILED) {
        close(mf->fd);
        free(mf);
        return NULL;
    }

    return mf;
}

void mmap_close(MmapFile* mf) {
    if (!mf) return;

    if (mf->data) munmap(mf->data, mf->size);
    if (mf->fd != -1) close(mf->fd);

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

#endif // !_WIN32
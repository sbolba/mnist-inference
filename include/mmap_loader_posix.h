#ifndef MMAP_LOADER_POSIX_H
#define MMAP_LOADER_POSIX_H

#include <stddef.h>
#include <stdint.h>
#include "tensor.h"

// Struttura posix
typedef struct {
    void* data;         // Puntatore all'inizio del file in memoria
    size_t size;        // Dimensione totale del file
    int fd;             // File descriptor (Posix)
} MmapFile;

// Apre un file e lo mappa in memoria (sola lettura)
MmapFile* mmap_open(const char* filepath);

// Chiude il file e libera la mappatura
void mmap_close(MmapFile* mf);

// Crea un tensore che punta direttamente ai dati mappati (zero-copy)
// offset: byte offset nel file dove iniziano i dati del tensore
Tensor* mmap_read_tensor(MmapFile* mf, size_t offset, int32_t* shape, int32_t ndim);

#endif // MMAP_LOADER_POSIX_H

#ifndef NPY_IO_H
#define NPY_IO_H

// Legge un file .npy assumendo dati raw float32 little-endian (come le immagini MNIST).
// Ritorna un buffer allocato con malloc (il chiamante deve fare free()).
// num_elements viene impostato al numero di elementi letti.
float* read_npy_float(const char* path, int* num_elements);

// Legge un file .npy contenente etichette intere, rilevando automaticamente il dtype
// dall'header (int64/int32/uint8/float32/float64) e convertendo tutto in int.
// Ritorna un buffer allocato con malloc (il chiamante deve fare free()), o NULL in caso di errore.
int* read_npy_int_labels(const char* path, int* num_elements);

// Legge immagini MNIST in formato IDX (idx3-ubyte) e le converte in float32
// usando la normalizzazione: (pixel / 255 - 0.1307) / 0.3081.
float* read_idx_images_float(const char* path, int* num_elements);

// Legge etichette MNIST in formato IDX (idx1-ubyte) e le converte in int.
int* read_idx_labels(const char* path, int* num_elements);

#endif // NPY_IO_H

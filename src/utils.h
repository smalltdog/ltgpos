#pragma once

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "geodistance.h"


// Get the datetime string secs ago of the given one.
void getPrevDatetime(char* datetime, unsigned int secs);

// Comparation function for sorting itdfs.
int cmpItdfs(const void* a, const void* b);


inline int malloc_s(void** p, int fsize, const char* var_name)
{
    for (int i = 0; i < 3; i++) {
        *p = (F*)malloc(fsize * sizeof(F));
        if (*p) return 0;
    }
    fprintf(stderr, "%s (line %d): Failed to malloc memory for %s.\n", __FILE__, __LINE__, var_name);
    return 1;
}


inline int cudaMalloc_s(void** p, int fsize, const char* var_name)
{
    for (int i = 0; i < 3; i++) {
        cudaError_t status = cudaMalloc(p, fsize * sizeof(F));
        if (status == cudaSuccess) return 0;
    }
    fprintf(stderr, "%s (line %d): Failed to malloc CUDA memory for %s.\n", __FILE__, __LINE__, var_name);
    return 1;
}

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>


typedef double F;


// Get the datetime string secs ago of the given one.
void getPrevDatetime(char* datetime, unsigned int secs);

// Comparation function for sorting itdfs.
int cmpItdf(const void* a, const void* b);


inline int malloc_s(F** p, int fsize, const char* var_name)
{
    for (int i = 0; i < 3; i++) {
        *p = (F*)malloc(fsize * sizeof(F));
        if (*p) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc memory for %s.\n",
            __FILE__, __LINE__, var_name);
    return 1;
}


inline int cudaMalloc_s(F** p, int fsize, const char* var_name)
{
    for (int i = 0; i < 3; i++) {
        cudaError_t status = cudaMalloc((void**)p, fsize * sizeof(F));
        if (status == cudaSuccess) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc CUDA memory for %s.\n",
            __FILE__, __LINE__, var_name);
    return 1;
}


inline F calItdf(F dist, F u)
{
    return pow(dist / 100, 1.13) * exp((dist - 100) / 1e5) / 3.576 * u;
}

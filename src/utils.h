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

int log2(long n);


inline int malloc_s(F** p, int fsize)
{
    for (int i = 0; i < 3; i++) {
        *p = (F*)malloc(fsize * sizeof(F));
        if (*p) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc memory.\n",
            __FILE__, __LINE__);
    return 1;
}


inline int cudaMalloc_s(F** p, int fsize)
{
    for (int i = 0; i < 3; i++) {
        cudaError_t status = cudaMalloc((void**)p, fsize * sizeof(F));
        if (status == cudaSuccess) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc CUDA memory.\n",
            __FILE__, __LINE__);
    return 1;
}


inline F calItdf(F dist, F u)
{
    return pow(dist / 100, 1.13) * exp((dist - 100) / 1e5) / 3.576 * u;
}


inline F avg(F* a, int len, int dim, int d)
{
    F sum = 0;
    for (int i = 0; i < len; i++) sum += a[i * dim + d];
    return sum / len;
}


inline F var(F* a, int len, int dim, int d, F e)
{
    F dev = 0;
    for (int i = 0; i < len; i++) dev += (a[i * dim + d] - e) * (a[i * dim + d] - e);
    return dev / len;
}

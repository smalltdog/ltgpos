#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "configs.h"

#define mask 0x1l


// Get the datetime string secs ago of the given one.
void getPrevDatetime(char* datetime, unsigned int secs);

// Generate seach domain.
void gen_sch_dom(double* srr_locs, int num_ssrs, long involved, double* sch_dom);

// Filter out outlier sensors.
long filter_outliers(double* ssr_locs, int num_ssrs);

// Grubbs's test.
std::vector<double> grubbs_test(std::vector<double> data);


inline int get_num_involved(long n) {
    int num_involved = 0;
    for (int i = 0; i < sizeof(n) * 8; i++) if (n & mask << i) ++num_involved;
    return num_involved;
}


inline int get_first_involved(long n) {
    for (int i = 0; i < sizeof(n) * 8; i++) if (n & mask << i) return i;
    return -1;
}


inline double itdf(double dist, double u) {
    return pow(dist / 100, 1.13) * exp((dist - 100) / 1e5) / 3.576 * u;
}


// Malloc with auto retry and error message.
inline int malloc_s(double** p, int fsize) {
    for (int i = 0; i < 3; i++) {
        *p = (double*)malloc(fsize * sizeof(double));
        if (*p) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc memory.\n",
            __FILE__, __LINE__);
    return 1;
}


// CudaMalloc with auto retry and error message.
inline int cudaMalloc_s(double** p, int fsize) {
    for (int i = 0; i < 3; i++) {
        cudaError_t status = cudaMalloc((void**)p, fsize * sizeof(double));
        if (status == cudaSuccess) return 0;
    }
    fprintf(stderr, "%s(%d): failed to malloc CUDA memory.\n",
            __FILE__, __LINE__);
    return 1;
}

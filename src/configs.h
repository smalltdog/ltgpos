#pragma once

// Max number of input sensors.
static const int kMaxNumSsrs = 64;

// Number of search times.
static const int kNumSchs = 2;
// Number of extended intervals for next search.
static const int kNxtSchDomInvs = 2;

// Max number of threads on GPU.
static const int kMaxNumThreads = 512 * 65535;
// // Max number of concurrent threads on GTX 1080 Ti.
// extern const int kMaxNumCncrThreads;

// Number of CUDA grids set for a single dimension.
extern int gMaxGridNum;
// Number of total CUDA grids set for grid search.
extern int gMaxGridSize;

// Expansion ratio for search domain.
extern double gSchDomRatio;
// Threshold for goodness of output results.
extern double gGoodThres;
// // Threshold for time delta of output results.
// extern double gDtimeThres;

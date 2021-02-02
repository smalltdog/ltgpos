// Max number of input sensors.
// const int kMaxNumSsrs = 64;

// Number of search times.
// const int kNumSchs = 2;
// Number of extended intervals for next search.
// const int kNxtSchDomInvs = 2;

// Max number of threads on GPU.
// const int kMaxNumThreads = 512 * 65535;
// // Max number of concurrent threads on GTX 1080 Ti.
// const int kMaxNumCncrThreads = 28 * 2048;

// Number of CUDA grids set for a single dimension.
int gMaxGridNum = 625;
// Number of total CUDA grids set for grid search.
int gMaxGridSize = gMaxGridNum * gMaxGridNum;

// Expansion ratio for search domain.
double gSchDomRatio = 1.2;
// // Threshold for goodness of output results.
double gGoodThres = 40;
// // Threshold for time delta of output results.
// double gDtimeThres = 1 / C;

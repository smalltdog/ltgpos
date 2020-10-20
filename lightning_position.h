#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nested_grid_search.h"
#include "cJSON.h"


// 为计算结果分配内存空间
F* mallocResBytes(void);

// 为计算结果分配CUDA内存空间
F* cudamallocResBytes(void);

// 雷电定位算法
char* ltgPosition(char* json_str, F* dChiOutFst, F* dChiOutSec, F* hChiOutFst, F* hChiOutSec);

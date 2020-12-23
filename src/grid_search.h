#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utils.h"
#include "geodistance.h"


typedef struct info {
    bool is3d;
    int grid_sizes[3];
    F grid_inv;
    F sch_dom[6];
    F* outs_h;
    F* outs_d;
} info_t;

typedef struct sysinfo {
    F* sensor_locs_d;
    F* sensor_times_d;
    info_t nodes[2];
} sysinfo_t;


extern const int kNxtSchDomInvs;

// ...
void grid_search(sysinfo_t* sysinfo, int num_sensors, F* sensor_locs, F* sensor_times, F results[5]);

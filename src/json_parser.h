#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "cJSON.h"
#include "utils.h"
#include "geodistance.h"


typedef struct data {
    int num_sensors;
    bool is3d;
    F* sensor_locs;
    F* sensor_times;
    F* sch_dom;
    F* out_ans;
    F* us;
    F base_ms;
    char* base_datetime;
    char** node_str;
    int* is_involved;
} data_t;


// Returns pointer to cJSON_Item if string parsed successfully.
// cJSON_Item returned should be freed after use.
cJSON* parseJsonStr(const char* jstr, data_t* data, F gSchDomRatio, int gMaxNumSensors);

// Returns string formatted from cJSON_Object created according to result.
// String returned should be deallocated after use.
char* formatRetJsonStr(data_t* result, cJSON* jarr, int gMaxNumSensors);

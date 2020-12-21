#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "cJSON.h"
#include "geodistance.h"


typedef struct data {
    int num_sensors;
    bool is3d;
    F* sensor_locs;
    F* sensor_times;
    F* sch_dom;
    F base_ms;
    char* base_datetime;
} data_t;


// Returns pointer to cJSON_Item if string parsed successfully.
// cJSON_Item returned should be freed after use.
cJSON* parseJsonStr(const char* jstr, data_t* data, F gSchDomRatio);

// Returns string formatted from cJSON_Object created according to result.
// String returned should be deallocated after use.
char* formatRetJsonStr(result_t* result, cJSON* jarr);

#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "cJSON.h"
#include "utils.h"
#include "configs.h"
#include "geodistance.h"


typedef struct schdata {
    int num_ssrs;
    long involved;
    double ssr_locs[kMaxNumSsrs * 2];
    double ssr_times[kMaxNumSsrs];
    double sch_dom[4];
    double out_ans[5];
} schdata_t;


// Returns pointer to cJSON_Item if string parsed successfully.
// cJSON_Item returned should be freed after use.
cJSON* parseJsonStr(const char* jstr, schdata_t* schdata);

// Returns string formatted from cJSON_Object created according to result.
// String returned should be deallocated after use.
char* formatRetJsonStr(schdata_t* schdata, cJSON* jarr);

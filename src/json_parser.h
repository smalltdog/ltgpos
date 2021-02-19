#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "cJSON.h"
#include "utils.h"
#include "configs.h"
#include "geodistance.h"


typedef struct schdata {
    int num_ssrs;
    long involved;
    F ssr_locs[64 * 2];
    F ssr_times[64];
    F sch_dom[4];
    F out_ans[5];
} schdata_t;


// Returns pointer to cJSON_Item if string parsed successfully.
// cJSON_Item returned should be freed after use.
cJSON* parseJsonStr(const char* jstr, schdata_t* schdata);

// Returns string formatted from cJSON_Object created according to result.
// String returned should be deallocated after use.
char* formatRetJsonStr(schdata_t* schdata, cJSON* jarr);

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "geodistance.h"
#include "grid_search.h"
#include "json_parser.h"
#include "utils.h"


// ...
// Returns 0 if re-init sysinfo successfully.
int set_cfg(int num_sensors, int grid_size);

// Returns string formatted from result cJSON_Object.
// String returned should be deallocated after use.
char* ltgpos(char* str);

int initSysInfo(void);

void freeSysInfo(void);

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "configs.h"
#include "geodistance.h"
#include "json_parser.h"
#include "grid_search.h"
#include "comb_mapper.h"


// Returns JSON-formatted string of output results.
// String returned should be deallocated after use.
char* ltgpos(char* str);

// Free calculation informaiton for Ltgpos service,
// including CPU & GPU memory allocated for parallel computing.
// Should be called once the service is about to abort.
void freeSysInfo(void);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nested_grid_search.h"
#include "cJSON.h"


// 配置信息初始化
CfgInfo* initCfgInfo()
// 雷电定位算法
char* ltgPosition(char* json_str, CfgInfo cfgInfo);

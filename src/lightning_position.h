#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nested_grid_search.h"
#include "cJSON.h"


/**
 * @brief 为系统计算设定配置参数
 * @param  maxNumSensors    最大检测站点数，默认 64
 * @param  maxGridSize      最大搜索网格数，默认 80 * 80 * 80
 * @param  schDomRatio      搜索区域扩大比例，默认 1.2
 * @param  dtimeThreshold   反演时选取阈值，默认 1 km / C km/ms
 * @param  isInvCal         是否进行初筛以及反演计算，默认 true
 * @return Info_t* CUDA内存管理和网格搜索信息结构的指针
 */
void setCfg(int maxNumSensors, int maxGridSize, double schDomRatio, double dtimeThreshold, bool isInvCal);
// 从文件中读取系统计算配置参数
void setCfgFromFile(char* filename);

// 为网格搜索计算结果分配 Host 和 Device 内存空间
int mallocResBytes(void);
// 释放为网格搜索计算结果分配的 Host 和 Device 内存空间
void freeResBytes(void);

// 并行雷电定位计算
char* ltgPosition(char* json_str);

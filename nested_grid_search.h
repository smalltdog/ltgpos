#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


/// 是否默认二级搜索
#define DEFAULT_SECOND_SEARCH 1

/// 对一级搜索结果进行二级搜索的范围 [-SEC_GRID_SIZE,+SEC_GRID_SIZE]
#define SEC_GRID_SIZE (2)

/// 整个系统计算的数据类型
typedef double F;

/// 最大GridSize，若超过则infoInit退出
const int kMaxGridSize = 80 * 80 * 80;

/// CUDA 内存管理和网格搜索信息结构
typedef struct {
    F schDom[6];

    /// 两次搜索的网格数量
    int gridXSizeFst;
    int gridYSizeFst;
    int gridZSizeFst;
    int gridXSizeSec;
    int gridYSizeSec;
    int gridZSizeSec;

    /// host端两次搜索输出的位置
    F* hChiOutFst;
    F* hChiOutSec;
    /// device端的两次搜索的输出
    F* dChiOutFst;
    F* dChiOutSec;
    /// device端两次搜索的范围
    F* dSchDomFst;
    F* dSchDomSec;
    /// 两次搜索的间隔
    F gridInvFst;
    F gridInvSec;
} Info_t;


void nested_grid_search_sph(unsigned int nOfSensor, F* sensorLocs, F* sensorTimes, Info_t* info_p, F outAns[5], bool is_3D);

Info_t* infoInit(F gridInvFst, F gridInvSec, F schDom[6], bool is_3D, F* dChiOutFst, F* dChiOutSec, F* hChiOutFst, F* hChiOutSec);

int infoFree(Info_t* p);

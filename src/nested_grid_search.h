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


/// C : 光速，km/ms
#define C (299792458 / 1e6)
/// 圆周率值
#define PI ((F) 3.14159265358979323846)
/// 计算角度到弧度
#define RAD(d) ((F)((d) * PI / 180.0))
/// 地球半径相关计算
#define RA  ((F)6378.137)
#define RB  ((F)(1 - F_T) * RA)
#define F_T ((F)1 / 298.257223563)
#define FLATTEN ((F)(RA - RB) / RA)
#define Rad_to_deg ((F)(45.0 / atan(1.0)))
#define DEF_R ((F)6370693.5)


/// 是否默认二级搜索
#define DEFAULT_SECOND_SEARCH 1

/// 对一级搜索结果进行二级搜索的范围 [-SEC_GRID_SIZE,+SEC_GRID_SIZE]
#define SEC_GRID_SIZE (2)

#define DEBUG 1

/// 整个系统计算的数据类型
typedef double F;


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


void nested_grid_search_sph(unsigned int nOfSensor, F* sensorLocs, F* sensorTimes, Info_t* info_p, F outAns[5], bool is3d);

Info_t* infoInit(F gridInvFst, F gridInvSec, F schDom[6], bool is3d);

int infoFree(Info_t* p);

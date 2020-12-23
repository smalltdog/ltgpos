#include "ltgpos.h"


// The max number of threads on GPU.
const int kMaxNumThreads = 512 * 65535;
// The max number of concurrent threads on GTX 1080 Ti
const int kMaxNumCncrThreads = 28 * 2048;

// Configs
int gMaxNumSensors = 64;        // 最大检测站点数
int gMaxGridSize = 768 * 768;   // 最大搜索网格数
double gSchDomRatio = 1.2;      // 搜索区域扩大比例
double gGoodThres = 4;          // 优度阈值
double gDtimeThres = 1 / C;     // 时差阈值

// Global system info
sysinfo_t gSysInfo;
bool isInit = false;


int initSysInfo()
{
    int status;
    status = malloc_s(&gSysInfo.nodes[0].outs_h, gMaxGridSize, "outs_h0");
    if (status) return 1;
    status = malloc_s(&gSysInfo.nodes[1].outs_h, gMaxGridSize, "outs_h1");
    if (status) return 1;

    status = cudaMalloc_s(&gSysInfo.nodes[0].outs_d, gMaxGridSize, "outs_d0");
    if (status) return 1;
    status = cudaMalloc_s(&gSysInfo.nodes[1].outs_d, gMaxGridSize, "outs_d1");
    if (status) return 1;

    status = cudaMalloc_s(&gSysInfo.sensor_locs_d, gMaxNumSensors * 3, "sensor_locs");
    if (status) return 1;
    status = cudaMalloc_s(&gSysInfo.sensor_times_d, gMaxNumSensors, "sensor_times");
    if (status) return 1;
    isInit = true;
    return 0;
}


void freeSysInfo()
{
    free(gSysInfo.nodes[0].outs_h);
    free(gSysInfo.nodes[1].outs_h);
    cudaFree(gSysInfo.nodes[0].outs_d);
    cudaFree(gSysInfo.nodes[1].outs_d);
    cudaFree(gSysInfo.sensor_locs_d);
    cudaFree(gSysInfo.sensor_times_d);
    isInit = false;
}


void initCalInfo(F* sch_dom, bool is3d)
{
    F area = (sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]);
    gSysInfo.nodes[0].is3d = false;
    gSysInfo.nodes[1].is3d = is3d;
    F inv = sqrt(area / gMaxGridSize);
    gSysInfo.nodes[0].grid_inv = inv > 0.003 ? inv : 0.003;
    inv = inv * 2 * kNxtSchDomInvs / sqrt(gMaxGridSize);
    gSysInfo.nodes[1].grid_inv = inv > 0.0001 ? inv : 0.0001;
    memcpy(gSysInfo.nodes[0].sch_dom, sch_dom, 6 * sizeof(F));
    // printf("%lf  %lf  ", gSysInfo.nodes[0].grid_inv, gSysInfo.nodes[1].grid_inv);
}


int set_cfg(int num_sensors, int grid_size)
{
    gMaxNumSensors = num_sensors;
    gMaxGridSize <= kMaxNumThreads ? (gMaxGridSize = grid_size) :
    fprintf(stderr, "%s(%d): grid size > the upper limit of concurrent threads.\n",
            __FILE__, __LINE__);
    freeSysInfo();
    return initSysInfo();
}


char* ltgpos(char* str)
{
    if (!isInit) {
        fprintf(stderr, "%s(%d): sysInfo had not been initialized yet.\n", __FILE__, __LINE__);
        return NULL;
    }

    F sensor_locs[gMaxNumSensors * 3];
    F sensor_times[gMaxNumSensors];
    F sch_dom[6];
    F out_ans[5];
    F us[gMaxNumSensors];
    char* node_str[gMaxNumSensors];
    int is_involved[gMaxNumSensors];

    data_t data;
    data.sensor_locs = sensor_locs;
    data.sensor_times = sensor_times;
    data.sch_dom = sch_dom;
    data.out_ans = out_ans;
    data.us = us;
    data.node_str = node_str;
    data.is_involved = is_involved;

    // Get input data by parsing json string.
    // Ensure jarr is deleted before return.
    cJSON* jarr = parseJsonStr(str, &data, gSchDomRatio);
    if (!jarr) return NULL;

    initCalInfo(data.sch_dom, data.is3d);
    grid_search(&gSysInfo, data.num_sensors, sensor_locs, sensor_times, out_ans);

    #ifdef TEST
    printf("%7.4lf  %8.4lf  %.4lf\n", out_ans[1], out_ans[2], out_ans[4]);
    #endif

    char* ret_str = formatRetJsonStr(&data, jarr, gMaxNumSensors);
    // Ensure ret_str is deallocated after use.
    return ret_str;
}

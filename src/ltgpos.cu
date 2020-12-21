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
    status = malloc_s(&gSysInfo.node[0].outs_h, gMaxGridSize, "outs_h0");
    if (status) return 1;
    status = malloc_s(&gSysInfo.node[1].outs_h, gMaxGridSize, "outs_h1");
    if (status) return 1;

    status = cudaMalloc_s(&gSysInfo.node[0].outs_d, gMaxGridSize, "outs_d0");
    if (status) return 1;
    status = cudaMalloc_s(&gSysInfo.node[1].outs_d, gMaxGridSize, "outs_d1");
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
    free(gSysInfo.node[0].outs_h);
    free(gSysInfo.node[1].outs_h);
    cudaFree(gSysInfo.node[0].outs_d);
    cudaFree(gSysInfo.node[1].outs_d);
    cudaFree(gSysInfo.sensor_locs_d);
    cudaFree(gSysInfo.sensor_times_d);
    isInit = false;
}


int initCalInfo(data_t* data)
{
    F* sch_dom = sch_dom;
    area = (sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]);
    // TODO if area > 256, then warning.
    gSysInfo.node[0].is3d = false;
    gSysInfo.node[1].is3d = data->is3d;
    gSysInfo.node[0].grid_inv = area * 2 / kMaxNumCncrThreads :
    gSysInfo.node[1].grid_inv = gSysInfo.node[0].grid_inv;
    gSysInfo.node[0].sch_dom = data->sch_dom;
}


int set_cfg(int num_sensors, int grid_size)
{
    gMaxNumSensors = num_sensors;
    gMaxGridSize <= kMaxNumThreads ? (gMaxGridSize = grid_size) :
    fprintf(stderr, "%s (line %d): Grid size > the upper limit of concurrent threads.\n"， __FILE__, __LINE__);
    freeSysInfo();
    return initSysInfo();
}


char* ltgpos(char* str)
{
    if (!isInit) {
        fprintf(stderr, "%s (line %d): SysInfo had not been initialized yet.\n", __FILE__, __LINE__);
        return NULL;
    }

    F sensor_locs[gMaxNumSensors * 3];
    F sensor_times[gMaxNumSensors];
    F sch_dom[6];

    data_t data;
    data.sensor_locs = sensor_locs;
    data.sensor_times = sensor_times;
    data.sch_dom = sch_dom;

    // Get input data by parsing json string.
    // Ensure jarr is deleted before return.
    cJSON* jarr = parseJsonStr(str, &data, gSchDomRatio);
    if (!jarr) return NULL;
    initCalInfo(&data);

    F results[5];
    grid_search(&gSysInfo, data.num_sensors, sensor_locs, sensor_times, results);

    #ifdef TEST
    printf("%8lf, %10.4lf, %10.4lf, %8.2lf, %8.2lf\n");
    #endif

    // char* ret_str = formatRetJsonStr(result, jarr);
    // Ensure ret_str is freed after use.
    return ret_str;
}

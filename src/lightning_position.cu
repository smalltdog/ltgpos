#include "lightning_position.h"


const char* kJsonKey[5] = { "time", "latitude", "longitude", "altitude", "goodness" };

int gMaxNumSensors = 64;            // 最大检测站点数，默认 64
int gMaxGridSize = 80 * 80 * 80;    // 最大搜索网格数，默认 80 * 80 * 80
double gSchDomRatio = 1.2;          // 搜索区域扩大比例，默认 1.2
double gDtimeThreshold = 1 / C;     // 反演时选取阈值，默认 1 km / C km/ms
bool gIsInvCal = true;              // 是否进行初筛以及反演计算，默认 true

// 为搜索结果分配的 Host 和 Device 内存空间
F* ghChiOutFst, * ghChiOutSec, * gdChiOutFst, * gdChiOutSec;


// 为网格搜索计算结果分配 Host 内存空间，分配成功返回1，否则返回0
int H_mallocResBytes(F** hChiOut)
{
    for (int i = 10; i;--i) {        
        *hChiOut = (F*)malloc(gMaxGridSize * sizeof(F));
        if (*hChiOut) return 1;
    }
    fprintf(stderr, "lightning_position(line %d): malloc hChiOut failed!\n", __LINE__);
    return 0;
}


// 为网格搜索计算结果分配 Device 内存空间，分配成功返回1，否则返回0
int D_mallocResBytes(F** dChiOut)
{    
    for (int i = 10; i;--i) {        
        cudaError_t status = cudaMalloc((void**)dChiOut, gMaxGridSize * sizeof(F));
        if (status == cudaSuccess) return 1;
    }
    fprintf(stderr, "lightning_position(line %d): malloc hChiOut failed!\n", __LINE__);
    return 0;
}


int mallocResBytes(void)
{
    if (!H_mallocResBytes(&ghChiOutFst)) return 0;
    if (!H_mallocResBytes(&ghChiOutSec)) return 0;
    if (!D_mallocResBytes(&gdChiOutFst)) return 0;
    if (!D_mallocResBytes(&gdChiOutSec)) return 0;
    return 1;
}


void freeResBytes(void)
{
    free(ghChiOutFst);
    free(ghChiOutSec);
    cudaFree(gdChiOutFst);
    cudaFree(gdChiOutSec);
}


void setCfg(int maxNumSensors, int maxGridSize, double schDomRatio, double dtimeThreshold, bool isInvCal)
{
    gMaxNumSensors = maxNumSensors;
    gMaxGridSize = maxGridSize;
    gSchDomRatio = schDomRatio;
    gDtimeThreshold = dtimeThreshold;
    gIsInvCal = isInvCal;
}


void setCfgFromFile(char* filename)
{
    return;
}


// 二维 WGS_84 坐标距离计算
F getDistance2d(F lat1, F lng1, F lat2, F lng2)
{
    F rad_lat_A = RAD(lat1);
    F rad_lng_A = RAD(lng1);
    F rad_lat_B = RAD(lat2);
    F rad_lng_B = RAD(lng2);

    F pA = atan(RB / RA * tan(rad_lat_A));
    F pB = atan(RB / RA * tan(rad_lat_B));

    F xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B));
    F c1 = (sin(xx) - xx) * pow(sin(pA) + sin(pB), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pA) - sin(pB), 2) / pow(sin(xx / 2), 2);
    F dr = FLATTEN / 8 * (c1 - c2);
    F distance = RA * (xx + dr);

    return distance;
}


// 三维 WGS_84 坐标距离计算
F getDistance3d(F lat1, F lng1, F asl1, F lat2, F lng2, F asl2)
{
    F distance = getDistance2d(lat1, lng1, lat2, lng2);
    return sqrt(distance * distance + (asl1 - asl2) * (asl1 - asl2));
}


// 获取时间戳减一秒后的标准时间格式字符串
char* timeminus1(char* ss)
{
    char *p;
    struct tm info;
    struct tm *info1;
    time_t timer;
    static char ans[20];

    p = strtok(ss, "-");
    info.tm_year = atoi(p) - 1900;
    p = strtok(NULL,"-");
    info.tm_mon = atoi(p)-1;
    p=strtok(NULL," ");
    info.tm_mday = atoi(p);
    p = strtok(NULL, ":");
    info.tm_hour = atoi(p);
    p = strtok(NULL, ":");
    info.tm_min = atoi(p);
    p = strtok(NULL," ");
    info.tm_sec = atoi(p);
    timer = mktime(&info);                  // 将tm结构转为time_t结构
    timer = timer - 1;                      // 对1970年到现在的秒数-1
    info1 = gmtime(&timer);                 // 将time_t转回tm结构
    info1->tm_hour = info1->tm_hour + 8;    // 中国时区+8
    strftime(ans, sizeof(ans), "%Y-%m-%d %H:%M:%S", info1);
    return ans;
}


int cmpfunc(const void* a, const void* b)
{
    return *(F*)a - *(F*)b;
}


/*
inline cJSON* cJSON_GetObjectItem_s(cJSON* object, const char* string)
{
    cJSON* item = cJSON_GetObjectItem(object, string);
    if (!item) {
        fprintf(stderr, "lightning_position(line %d): Missing JSON key \"%s\"\n", __LINE__, string);
        goto L1;
    }
    return item;
}
*/


#define cJSON_GetObjectItem_s(json_item, obj, string) { \
    (json_item) = cJSON_GetObjectItem((obj), (string)); \
    if (!(json_item)) { \
        fprintf(stderr, "lightning_position(line %d): Missing JSON key \"%s\"\n", __LINE__, (string)); \
        return NULL; \
    } \
}


char* ltgPosition(char* json_str)
{
    #ifdef DEBUG
    printf("[Configs]: \n");
    printf("\t   MaxNumSensors:  %d\n", gMaxNumSensors);
    printf("\t   MaxGridSize:    %d\n", gMaxGridSize);
    printf("\t   SchDomRatio:    %lf\n", gSchDomRatio);
    printf("\t   DtimeThreshold: %lf\n", gDtimeThreshold);
    printf("\t   IsInvCal:       %d\n\n", gIsInvCal);
    #endif

    cJSON* json_arr = cJSON_Parse(json_str);
    cJSON* json_obj = NULL;
    cJSON* json_item = NULL;

    int num_dims = 2, num_sensors = cJSON_GetArraySize(json_arr);
    if (num_sensors < 3) {
        fprintf(stderr, "lightning_position(line %d): Lightning position expect num of sensors > 2, but get %d\n",
                __LINE__, num_sensors);
        return NULL;
    }

    F sensor_locs[gMaxNumSensors * 3], sensor_times[gMaxNumSensors], sch_dom[6], out_ans[5];
    F base_ms;
    char* base_datetime = NULL;
    bool is3d = false;
    F coord_dom[6];               // lat_min, lat_max, lon_min, lon_max, alt_min, alt_max;

    // get item from json object
    for (int i = 0; i < num_sensors; ++i) {
        json_obj = cJSON_GetArrayItem(json_arr, i);

        // j for 3 dimensions
        for (int j = 0; j < 3; ++j) {
            if (!is3d && i && j == 2) continue;                                 // not have altitude
            json_item = cJSON_GetObjectItem(json_obj, kJsonKey[j + 1]);
            if (!i && j == 2) {
                if (!json_item) continue;                                       // not have altitude
                else { is3d = true; num_dims = 3; }
            }
            if (!json_item) {                                                   // miss JSON key
                fprintf(stderr, "lightning_position(line %d): Missing JSON key \"%s\"\n", __LINE__, kJsonKey[j + 1]);
                return NULL;
            }

            sensor_locs[i * num_dims + j] = json_item->valuedouble;
            if (!i)                                                             // i=0, initialize coordinate domain
                coord_dom[2 * j + 1] = coord_dom[2 * j] = sensor_locs[j];
            else if (sensor_locs[i * num_dims + j] > coord_dom[2 * j + 1])      // loc > max value of that dimension
                coord_dom[2 * j + 1] = sensor_locs[i * num_dims + j];
            else if (sensor_locs[i * num_dims + j] < coord_dom[2 * j])          // loc < min value of that dimension
                coord_dom[2 * j] = sensor_locs[i * num_dims + j];
        }

        // datetime & microseconds
        if (i) {
            cJSON_GetObjectItem_s(json_item, json_obj, "microsecond");
            sensor_times[i] = (F)json_item->valueint / 1e4 - base_ms;
            if (sensor_times[i] <= 0) sensor_times[i] += 1e3;
        }
        else {
            cJSON_GetObjectItem_s(json_item, json_obj, "datetime");
            base_datetime = json_item->valuestring;
            cJSON_GetObjectItem_s(json_item, json_obj, "microsecond");
            base_ms = (F)json_item->valueint / 1e4;
            sensor_times[i] = 0;
        }
    }

    // generate search domain, expand_ratio = 2
    for (int i = 0; i < 6; ++i) {
        sch_dom[i] = (gSchDomRatio / 2 + 0.5) * coord_dom[i] -
                     (gSchDomRatio / 2 - 0.5) * coord_dom[i % 2 ? i - 1 : i + 1];
    }

    Info_t* info_p;
    info_p = infoInit(sqrt(sqrt((sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]) / 1e6)), 0.001,
                      sch_dom, is3d, gMaxGridSize, ghChiOutFst, ghChiOutSec, gdChiOutFst, gdChiOutSec);
    if (!info_p) {
        fprintf(stderr, "lightning_position(line %d): SystemInfo init failed!\n", __LINE__);
        return NULL;
    }

    #ifdef DEBUG
    printf("[Infoinit] GridInv: %lf\n", sqrt(sqrt((sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]) / 1e6)));
    printf("[Infoinit] SchDom: ");
    for (int i = 0; i < 6; ++i) printf("%lf ", sch_dom[i]);
    printf("\n\n");
    #endif

    // preliminary positioning using 3 nodes
    F sensor_lf[gMaxNumSensors * 3], sensor_tf[gMaxNumSensors];     // sensors locs & times for final calculation
    F best_goodness;
    int best_ijk[3];

    // ijk for node index in sensor_locs & sensor_times
    for (int ijk[3] = {0}; ijk[0] < num_sensors; ++ijk[0]) {
        for (ijk[1] = ijk[0] + 1; ijk[1] < num_sensors; ++ijk[1]) {
            for (ijk[2] = ijk[1] + 1; ijk[2] < num_sensors; ++ijk[2]) {
                // i for 3 referrence node idx
                for (int i = 0; i < 3; ++i) {
                    memcpy(sensor_lf + num_dims * i, sensor_locs + num_dims * ijk[i], num_dims * sizeof(F));
                    sensor_tf[i] = sensor_times[ijk[i]];
                }

                nested_grid_search_sph(3, sensor_lf, sensor_tf, info_p, out_ans, is3d);
                if ((ijk[0] != 0 || ijk[1] != 1 || ijk[2] != 2) && out_ans[4] >= best_goodness) continue;
                best_goodness = out_ans[4];
                memcpy(best_ijk, ijk, 3 * sizeof(int));
            }
        }
    }

    int is_involved[gMaxNumSensors] = {0};
    char* node_str[gMaxNumSensors];
    F us[gMaxNumSensors], itdfs[gMaxNumSensors];

    // copy best goodness nodes to sensor_lf & sensor_tf
    // i for 3 nodes of best goodness
    for (int i = 0; i < 3; ++i) {
        memcpy(sensor_lf + num_dims * i, sensor_locs + num_dims * best_ijk[i], num_dims * sizeof(F));
        sensor_tf[i] = sensor_times[best_ijk[i]];
        is_involved[best_ijk[i]] = 1;
        cJSON_GetObjectItem_s(json_item, cJSON_GetArrayItem(json_arr, best_ijk[i]), "node");
        node_str[i] = json_item->valuestring;
        cJSON_GetObjectItem_s(json_item, cJSON_GetArrayItem(json_arr, best_ijk[i]), "signal_strength");
        us[i] = json_item->valuedouble;
    }

    // inverse calculation
    F dtime;
    int num_involved = 3;

    for (int i = 0; i < num_sensors; ++i) {
        if (i == best_ijk[0] || i == best_ijk[1] || i == best_ijk[2]) continue;

        // memcpy(sensor_l + num_dims * 3, sensor_locs + num_dims * i, num_dims * sizeof(F));
        // sensor_t[3] = sensor_times[i];
        // nested_grid_search_sph(4, sensor_l, sensor_t, info_p, out_ans, is3d);

        dtime = is3d ?
                getDistance3d(sensor_locs[3 * i], sensor_locs[3 * i + 1], sensor_locs[3 * i + 2],
                              out_ans[1], out_ans[2], out_ans[3]) / C :
                getDistance2d(sensor_locs[2 * i], sensor_locs[2 * i + 1],
                              out_ans[1], out_ans[2]) / C;
        if (gIsInvCal && abs(dtime - sensor_times[i] + out_ans[0]) >= gDtimeThreshold) continue;

        memcpy(sensor_lf + num_dims * num_involved, sensor_locs + num_dims * i, num_dims * sizeof(F));
        sensor_tf[num_involved] = sensor_times[i];
        is_involved[i] = 1;
        cJSON_GetObjectItem_s(json_item, cJSON_GetArrayItem(json_arr, i), "node");
        node_str[num_involved] = json_item->valuestring;
        cJSON_GetObjectItem_s(json_item, cJSON_GetArrayItem(json_arr, i), "signal_strength");
        us[num_involved++] = json_item->valuedouble;
    }

    // final calculation
    nested_grid_search_sph(num_involved, sensor_lf, sensor_tf, info_p, out_ans, is3d);
    infoFree(info_p);

    if ((out_ans[0] += base_ms) < 0) {
        out_ans[0] += 1e3;
        base_datetime = timeminus1(base_datetime);
    }

    F all_dist[gMaxNumSensors], all_dtime[gMaxNumSensors];
    for (int i = 0; i < num_sensors; ++i) {
        all_dist[i] = is3d ?
                      getDistance3d(sensor_locs[3 * i], sensor_locs[3 * i + 1], sensor_locs[3 * i + 2],
                                    out_ans[1], out_ans[2], out_ans[3]) / C :
                      getDistance2d(sensor_locs[2 * i], sensor_locs[2 * i + 1],
                                    out_ans[1], out_ans[2]) / C;
        all_dtime[i] = (sensor_times[i] > sensor_times[0] ?
                        sensor_times[i] / 1e4 : sensor_times[i] / 1e4 + 1e3) + out_ans[0];
    }

    for (int i = 0, j = 0; i < num_sensors; ++i) {
        if (!is_involved[i]) continue;
        itdfs[j++] = pow(all_dist[i] / 100, 1.13) * exp((all_dist[i] - 100) / 100000) / 3.576 * us[j];
    }

    F itdfs_sort[gMaxNumSensors];
    memcpy(itdfs_sort, itdfs, num_involved * sizeof(F));
    qsort(itdfs_sort, num_involved, sizeof(F), cmpfunc);

    // create json obj for return
    json_obj = cJSON_CreateObject();
    cJSON_AddItemToObject(json_obj, "datetime", cJSON_CreateString(base_datetime));

    // create item from out_ans & add to json obj
    for (int i = 0; i < 5; ++i) {
        if (i == 3 && !is3d) continue;
        json_item = cJSON_CreateNumber(out_ans[i]);
        cJSON_AddItemToObject(json_obj, kJsonKey[i], json_item);
    }

    cJSON_AddItemToObject(json_obj, "current",
                          cJSON_CreateNumber(num_sensors % 2 ? itdfs_sort[num_sensors / 2] :
                          (itdfs_sort[num_sensors / 2] + itdfs_sort[num_sensors / 2 - 1]) / 2));
    cJSON_AddItemToObject(json_obj, "raw", json_arr);
    cJSON_AddItemToObject(json_obj, "allDistance", cJSON_CreateDoubleArray(all_dist, num_sensors));
    cJSON_AddItemToObject(json_obj, "timeDiff", cJSON_CreateDoubleArray(all_dtime, num_sensors));
    cJSON_AddItemToObject(json_obj, "involvedNode", cJSON_CreateStringArray((const char**)node_str, num_involved));
    cJSON_AddItemToObject(json_obj, "referenceNodeNode", cJSON_CreateString(node_str[0]));
    cJSON_AddItemToObject(json_obj, "basicThreeNode", cJSON_CreateStringArray((const char**)node_str, 3));
    cJSON_AddItemToObject(json_obj, "isInvolved", cJSON_CreateIntArray(is_involved, num_sensors));
    cJSON_AddItemToObject(json_obj, "involvedSignalStrength", cJSON_CreateDoubleArray(us, num_involved));
    cJSON_AddItemToObject(json_obj, "involvedEstimatedCurrent", cJSON_CreateDoubleArray(itdfs, num_involved));

    char* ret_str = cJSON_PrintUnformatted(json_obj);
    cJSON_Delete(json_obj);
    return ret_str;
}

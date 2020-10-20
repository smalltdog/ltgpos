#include "lightning_position.h"

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


const int kMaxNumSensors = 64;          // TODO
const double kSchDomRatio = 1.2;        // TODO
const double kDtimeThreshold = 1 / C;   // TODO
const char kJsonKey[5][16] = { "time", "longitude", "latitude", "altitude", "goodness" };


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


F getDistance3d(F lat1, F lng1, F asl1, F lat2, F lng2, F asl2)
{
    F distance = getDistance2d(lat1, lng1, lat2, lng2);
    return sqrt(distance * distance + (asl1 - asl2) * (asl1 - asl2));
}


int cmpfunc(const void* a, const void* b)
{
    return *(F*)a - *(F*)b;
}


cJSON* cJSON_GetObjectItem_s(cJSON* object, const char* string)
{
    cJSON* item = cJSON_GetObjectItem(object, string);
    if (!item) {
        fprintf(stderr, "lightning_position(line %d): Missing JSON key \"%s\"", __LINE__, string);
        exit(1);
    }
    return item;
}


char* ltgPosition(char* json_str, F* dChiOutFst, F* dChiOutSec, F* hChiOutFst, F* hChiOutSec)
{
    cJSON* json_arr = cJSON_Parse(json_str);
    cJSON* json_obj = NULL;
    cJSON* json_item = NULL;

    int num_dims = 2, num_sensors = cJSON_GetArraySize(json_arr);
    if (num_sensors < 3) {
        fprintf(stderr, "lightning_position(line %d): Lightning position expect num of sensors > 2, but get %d",
                __LINE__, num_sensors);
        exit(1);
    }

    F sensor_locs[kMaxNumSensors * 3], sensor_times[kMaxNumSensors], sch_dom[6], out_ans[5];
    F base_ms;
    char* base_datetime = NULL;
    bool is3d = false;
    F coord_dom[6];               // lon_min, lon_max, lat_min, lat_max, alt_min, alt_max;

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
                fprintf(stderr, "lightning_position(line %d): Missing JSON key \"%s\"", __LINE__, kJsonKey[j + 1]);
                exit(1);
            }

            sensor_locs[i * num_dims + j] = json_item->valuedouble;
            if (!i)                                                             // i=0, initialize coordinate domain
                coord_dom[2 * j + 1] = coord_dom[2 * j] = sensor_locs[i * num_dims + j];
            else if (sensor_locs[i * num_dims + j] > coord_dom[2 * j + 1])      // loc > max value of that dimension
                coord_dom[2 * j + 1] = sensor_locs[i * num_dims + j];
            else if (sensor_locs[i * num_dims + 1] < coord_dom[2 * j])          // loc < min value of that dimension
                coord_dom[2 * j] = sensor_locs[i * num_dims + j];
        }

        // datetime & microseconds
        if (i) {
            sensor_times[i] = (F)cJSON_GetObjectItem_s(json_obj, "microsecond")->valueint / 1e4 - base_ms;
            if (sensor_times[i] <= 0) sensor_times[i] += 1e3;
        }
        else {
            base_datetime = cJSON_GetObjectItem_s(json_obj, "datetime")->valuestring;
            base_ms = (F)cJSON_GetObjectItem_s(json_obj, "microsecond")->valueint / 1e4;
            sensor_times[i] = 0;
        }
        // printf("%lf ", sensor_times[i]);
    }

    // generate search domain, expand_ratio = 2
    for (int i = 0; i < 6; ++i) {
        sch_dom[i] = (kSchDomRatio / 2 + 0.5) * coord_dom[i] -
                     (kSchDomRatio / 2 - 0.5) * coord_dom[i % 2 ? i - 1 : i + 1];
    }

    Info_t* info_p;
    info_p = infoInit(sqrt(sqrt((sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]) / 1e6)),
                      0.001, sch_dom, is3d, dChiOutFst, dChiOutSec, hChiOutFst, hChiOutSec);
    if (!info_p) {
        fprintf(stderr, "%s(%d): SystemInfo init failed!", __FILE__, __LINE__);
        exit(1);
    }


    // preliminary positioning using 3 nodes
    F sensor_lf[kMaxNumSensors * 3], sensor_tf[kMaxNumSensors];     // sensors locs & times for final calculation
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

    int is_involved[kMaxNumSensors] = {0};
    char* node_str[kMaxNumSensors];
    F us[kMaxNumSensors], itdfs[kMaxNumSensors];

    // copy best goodness nodes to sensor_lf & sensor_tf
    // i for 3 nodes of best goodness
    for (int i = 0; i < 3; ++i) {
        memcpy(sensor_lf + num_dims * i, sensor_locs + num_dims * best_ijk[i], num_dims * sizeof(F));
        sensor_tf[i] = sensor_times[best_ijk[i]];
        is_involved[best_ijk[i]] = 1;
        node_str[i] = cJSON_GetObjectItem_s(cJSON_GetArrayItem(json_arr, best_ijk[i]), "node")->valuestring;
        us[i] = cJSON_GetObjectItem_s(cJSON_GetArrayItem(json_arr, best_ijk[i]), "signal_strength")->valuedouble;
    }


    // inverse calculation
    F dtime;
    int num_involved = 3;

    for (int i = 0; i < num_sensors; ++i) {
        if (i == best_ijk[0] && i == best_ijk[1] || i == best_ijk[2]) continue;

        // memcpy(sensor_l + num_dims * 3, sensor_locs + num_dims * i, num_dims * sizeof(F));
        // sensor_t[3] = sensor_times[i];
        // nested_grid_search_sph(4, sensor_l, sensor_t, info_p, out_ans, is3d);

        dtime = is3d ?
                getDistance3d(sensor_locs[3 * i], sensor_locs[3 * i + 1], sensor_locs[3 * i + 2],
                              out_ans[1], out_ans[2], out_ans[3]) / C :
                getDistance2d(sensor_locs[2 * i], sensor_locs[2 * i + 1],
                              out_ans[1], out_ans[2]) / C;
        if (abs(dtime - sensor_times[i] + out_ans[0]) < kDtimeThreshold) {
        // if (1) {
            memcpy(sensor_lf + num_dims * num_involved, sensor_locs + num_dims * i, num_dims * sizeof(F));
            sensor_tf[num_involved] = sensor_times[i];
            is_involved[i] = 1;
            node_str[num_involved] = cJSON_GetObjectItem_s(cJSON_GetArrayItem(json_arr, i), "node")->valuestring;
            us[num_involved++] = cJSON_GetObjectItem_s(cJSON_GetArrayItem(json_arr, i), "signal_strength")->valuedouble;
        }
    }


    // final calculation
    nested_grid_search_sph(--num_involved, sensor_lf, sensor_tf, info_p, out_ans, is3d);
    infoFree(info_p);

    if ((out_ans[0] += base_ms) < 0) {
        out_ans[0] += 1e3;
        base_datetime = timeminus1(base_datetime);
    }

    F all_dist[kMaxNumSensors], all_dtime[kMaxNumSensors];
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

    F itdfs_sort[kMaxNumSensors];
    memcpy(itdfs_sort, itdfs, num_involved * sizeof(F));
    qsort(itdfs_sort, num_involved, sizeof(F), cmpfunc);


    // create json obj for return
    json_obj = cJSON_CreateObject();
    cJSON_AddItemToObject(json_obj, "datetime", cJSON_CreateString(base_datetime));
    // create item from out_ans & add to json obj
    for (int i = 0; i < 5; ++i) {
        if (i == 3 && !is3d) continue;
        cJSON* json_item = cJSON_CreateNumber(out_ans[i]);
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
    printf("\n\n");
    return ret_str;
}


F* mallocResBytes(void)
{
    F* hChiOut = NULL;
    while (1) {
        hChiOut = (F*)malloc(kMaxGridSize * sizeof(F));
        if (hChiOut) break;
        fprintf(stderr, "lightning_position(line %d): malloc hChiOut failed!\n", __LINE__);
    }
    return hChiOut;
}


F* cudamallocResBytes(void)
{
    F* dChiOut = NULL;
    cudaError_t cudaStatus;
    do {
        cudaStatus = cudaMalloc((void**)&dChiOut, kMaxGridSize * sizeof(F));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "lightning_position(line %d): cudamalloc dChiOut failed!\n", __LINE__);
    } while (cudaStatus != cudaSuccess);

    return dChiOut;
}

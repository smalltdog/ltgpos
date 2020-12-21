#include "json_parser.h"


const char* kJsonKeys[5] = { "time", "latitude", "longitude", "altitude", "goodness" };


#define cJSON_GetObjectItem_s(jitem, jobj, key) { \
    (jitem) = cJSON_GetObjectItem((jobj), (key)); \
    if (!(jitem)) { \
        fprintf(stderr, "%s (line %d): Missing json key \"%s\".\n", __FILE__, __LINE__, key); \
        return NULL; \
    } \
}


cJSON* parseJsonStr(const char* jstr, data_t* data, F gSchDomRatio)
{
    cJSON* jarr = cJOSN_Parse(jstr);
    cJSON* jobj = NULL;
    cJSON* jitem = NULL;

    F* sensor_locs = data->sensor_locs;
    F* sensor_times = data->sensor_times;
    F* sch_dom = data->sch_dom;
    F coord_dom[6], base_ms;
    char* base_datetime = data->base_datetime;

    int num_sensors = cJSON_GetArraySize(jarr), num_dims = 2;
    bool is3d = false;

    if (num_sensors < 3) {
        fprintf(stderr, "%s (line %d): Lightning positioning expects get num of sensors >= 3, but get %d.\n",
                __FILE__, __LINE__, num_sensors);
        return NULL;
    }

    for (int i = 0; i < num_sensors; i++) {
        jobj = cJSON_GetArrayItem(jarr, i);

        // j for 3 dimensions of sensor location.
        for (int j = 0; j < 3; j++) {
            // Determine whether input is of 3 dimensions.
            if (i == 0 && j == 2) {
                jitem = cJSON_GetObjectItem(jobj, kJsonKeys[j + 1]);
                if (!jitem) continue;
                num_dims = 3;
                is3d = true;
            }
            if (!is3d && j == 2) continue;

            cJSON_GetObjectItem_s(jitem, jobj, kJsonKeys[j + 1]);
            sensor_locs[i * num_dims + j] = jitem->valuedouble;

            // Update coordinate domain with min or max values.
            if (!i)
                coord_dom[2 * j + 1] = coord_dom[2 * j] = sensor_locs[j];
            else if (sensor_locs[i * num_dims + j] > coord_dom[2 * j + 1])
                coord_dom[2 * j + 1] = sensor_locs[i * num_dims + j];
            else if (sensor_locs[i * num_dims + j] < coord_dom[2 * j])
                coord_dom[2 * j] = sensor_locs[i * num_dims + j];
        }

        // Get datetime & milliseconds.
        if (i == 0) {
            cJSON_GetObjectItem_s(jitem, jobj, "datetime");
            base_datetime = jitem->valuestring;
            cJSON_GetObjectItem_s(jitem, jobj, "microsecond");
            base_ms = (F)jitem->valueint / 1e4;
            sensor_times[i] = 0;
        }
        else {
            cJSON_GetObjectItem_s(jitem, jobj, "microsecond");
            sensor_times[i] = (F)jitem->valueint / 1e4 - base_ms;
            // Assert the diff of seconds < 1 s.
            if (sensor_times[i] < 0) sensor_times[i] += 1e3;
        }
    }

    // Generate search domain with expand ratio.
    for (int i = 0; i < 6; i++) {
        sch_dom[i] = (gSchDomRatio / 2 + 0.5) * coord_dom[i] -
                     (gSchDomRatio / 2 - 0.5) * coord_dom[i%2 ? i-1 : i+1];
    }

    data->num_sensors = num_sensors;
    data->is3d = is3d;
    data->base_ms = base_ms;
    return jarr;
}


char* formatRetJsonStr(result_t* result, cJSON* jarr)
{
    jobj = cJSON_CreateObject();
    cJSON_AddItemToObject(jobj, "datetime", cJSON_CreateString(result->base_datetime));

    F* out_ans = reult->out_ans;
    // Create cJSON_Item from out_ans.
    for (int i = 0; i < 5; ++i) {
        if (i == 3 && !is3d) continue;
        jitem = cJSON_CreateNumber(out_ans[i]);
        cJSON_AddItemToObject(jobj, kJsonKey[i], jitem);
    }

    cJSON_AddItemToObject(jobj, "current",
                          cJSON_CreateNumber(num_sensors % 2 ? itdfs_sort[num_sensors / 2] :
                          (itdfs_sort[num_sensors / 2] + itdfs_sort[num_sensors / 2 - 1]) / 2));
    cJSON_AddItemToObject(jobj, "raw", json_arr);
    cJSON_AddItemToObject(jobj, "allDistance", cJSON_CreateDoubleArray(all_dist, num_sensors));
    cJSON_AddItemToObject(jobj, "timeDiff", cJSON_CreateDoubleArray(all_dtime, num_sensors));
    cJSON_AddItemToObject(jobj, "involvedNode", cJSON_CreateStringArray((const char**)node_str, num_involved));
    cJSON_AddItemToObject(jobj, "referenceNodeNode", cJSON_CreateString(node_str[0]));
    cJSON_AddItemToObject(jobj, "basicThreeNode", cJSON_CreateStringArray((const char**)node_str, 3));
    cJSON_AddItemToObject(jobj, "isInvolved", cJSON_CreateIntArray(is_involved, num_sensors));
    cJSON_AddItemToObject(jobj, "involvedSignalStrength", cJSON_CreateDoubleArray(us, num_involved));
    cJSON_AddItemToObject(jobj, "involvedEstimatedCurrent", cJSON_CreateDoubleArray(itdfs, num_involved));

    char* ret_str = cJSON_PrintUnformatted(jobj);
    cJSON_Delete(jobj);
    return ret_str;
}

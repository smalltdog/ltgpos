#include "json_parser.h"


const char* kJsonKeys[5] = { "time", "latitude", "longitude", "altitude", "goodness" };


#define cJSON_GetObjectItem_s(jitem, jobj, key) { \
    (jitem) = cJSON_GetObjectItem((jobj), (key)); \
    if (!(jitem)) { \
        fprintf(stderr, "%s(%d): missing json key \"%s\".\n", \
                __FILE__, __LINE__, key); \
        return NULL; \
    } \
}


cJSON* parseJsonStr(const char* jstr, data_t* data, F gSchDomRatio, int gMaxNumSensors)
{
    cJSON* jarr = cJSON_Parse(jstr);
    cJSON* jobj = NULL;
    cJSON* jitem = NULL;

    F* sensor_locs = data->sensor_locs;
    F* sensor_times = data->sensor_times;
    F* sch_dom = data->sch_dom;
    F* us = data->us;
    F coord_dom[6], base_ms;
    char** base_datetime = &data->base_datetime;
    char** node_str = data->node_str;
    int* is_involved = data->is_involved;

    int num_sensors = cJSON_GetArraySize(jarr), num_dims = 2;
    bool is3d = false;

    if (num_sensors < 3) {
        fprintf(stderr, "%s(%d): lightning positioning expects to get num of sensors >= 3, but got %d.\n",
                __FILE__, __LINE__, num_sensors);
        return NULL;
    }
    if (num_sensors > gMaxNumSensors) {
        fprintf(stderr, "%s(%d): lightning positioning expects to get num of sensors <= %d, but got %d.\n",
                __FILE__, __LINE__, gMaxNumSensors, num_sensors);
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
            *base_datetime = jitem->valuestring;
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

        // Get node string * signal strength.
        cJSON_GetObjectItem_s(jitem, jobj, "node");
        node_str[i] = jitem->valuestring;
        cJSON_GetObjectItem_s(jitem, jobj, "signal_strength");
        us[i] = jitem->valuedouble;
        is_involved[i] = 1;
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


char* formatRetJsonStr(data_t* data, cJSON* jarr, int gMaxNumSensors)
{
    int num_sensors = data->num_sensors;
    F* sensor_locs = data->sensor_locs;
    F* sensor_times = data->sensor_times;
    F* out_ans = data->out_ans;
    F* us = data->us;
    char** node_str = data->node_str;
    int* is_involved = data->is_involved;

    if ((out_ans[0] += data->base_ms) < 0) {
        out_ans[0] += 1e3;
        getPrevDatetime(data->base_datetime, 1);
    }

    cJSON* jobj = cJSON_CreateObject();
    cJSON_AddItemToObject(jobj, "datetime", cJSON_CreateString(data->base_datetime));

    // Create cJSON_Item from out_ans.
    for (int i = 0; i < 5; ++i) {
        if (i == 3 && !data->is3d) continue;
        cJSON_AddItemToObject(jobj, kJsonKeys[i],
                              cJSON_CreateNumber(out_ans[i]));
    }

    int num_involved = 0;
    F all_dist[gMaxNumSensors], all_dtime[gMaxNumSensors];
    F inv_us[gMaxNumSensors], inv_itdfs[gMaxNumSensors];
    char* inv_nodes[gMaxNumSensors];

    for (int i = 0; i < num_sensors; i++) {
        all_dist[i] = data->is3d ?
                      getGeoDistance3d_H(sensor_locs[i * 3], sensor_locs[i * 3 + 1], sensor_locs[i * 3 + 2],
                                       out_ans[1], out_ans[2], out_ans[3]) :
                      getGeoDistance2d_H(sensor_locs[i * 3], sensor_locs[i * 3 + 1],
                                       out_ans[1], out_ans[2]);
        all_dtime[i] = sensor_times[i] > sensor_times[0] ?
                       sensor_times[i] / 1e4 : sensor_times[i] / 1e4 + 1e3;
        all_dtime[i] += out_ans[0];

        if (!is_involved[i]) continue;
        inv_us[num_involved] = us[i];
        inv_itdfs[num_involved] = calItdf(all_dist[i], us[i]);
        inv_nodes[num_involved++] = node_str[i];
    }
    qsort(inv_itdfs, num_involved, sizeof(F), cmpItdf);

    cJSON_AddItemToObject(jobj, "current",
                          cJSON_CreateNumber(num_involved % 2 ? inv_itdfs[num_involved / 2] :
                          (inv_itdfs[num_involved / 2] + inv_itdfs[num_involved / 2 - 1]) / 2));
    cJSON_AddItemToObject(jobj, "raw", jarr);

    cJSON_AddItemToObject(jobj, "allDist",
                          cJSON_CreateDoubleArray(all_dist, num_sensors));
    cJSON_AddItemToObject(jobj, "allDtime",
                          cJSON_CreateDoubleArray(all_dtime, num_sensors));

    cJSON_AddItemToObject(jobj, "isInvolved",
                          cJSON_CreateIntArray(is_involved, num_sensors));
    cJSON_AddItemToObject(jobj, "involvedNodes",
                          cJSON_CreateStringArray((const char**)inv_nodes, num_involved));
    cJSON_AddItemToObject(jobj, "referNode",
                          cJSON_CreateString(inv_nodes[0]));
    cJSON_AddItemToObject(jobj, "basicNodes",
                          cJSON_CreateStringArray((const char**)inv_nodes, 3));

    cJSON_AddItemToObject(jobj, "involvedSignalStrength",
                          cJSON_CreateDoubleArray(inv_us, num_involved));
    cJSON_AddItemToObject(jobj, "involvedResultCurrent",
                          cJSON_CreateDoubleArray(inv_itdfs, num_involved));

    char* ret_str = cJSON_PrintUnformatted(jobj);
    cJSON_Delete(jobj);
    return ret_str;
}

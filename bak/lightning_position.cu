// 为搜索结果分配的 Host 和 Device 内存空间
F* ghChiOutFst, * ghChiOutSec, * gdChiOutFst, * gdChiOutSec;


char* ltgPosition(char* json_str)
{
    Info_t* info_p;
    info_p = infoInit(sqrt(sqrt((sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]) / 1e6)), 0.001,
                      sch_dom, is3d, gMaxGridSize, ghChiOutFst, ghChiOutSec, gdChiOutFst, gdChiOutSec);
    if (!info_p) {
        fprintf(stderr, "lightning_position(line %d): SystemInfo init failed!\n", __LINE__);
        return NULL;
    }


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
}

#include "json_parser.h"


const char* kJsonKeys[5] = { "time", "latitude", "longitude", "altitude", "goodness" };
const long mask = 0x1;


#define cJSON_GetObjectItem_s(jitem, jobj, key) { \
    (jitem) = cJSON_GetObjectItem((jobj), (key)); \
    if (!(jitem)) { \
        fprintf(stderr, "%s(%d): missing json key \"%s\".\n", \
                __FILE__, __LINE__, key); \
        return NULL; \
    } \
}


cJSON* parseJsonStr(const char* jstr, schdata_t* schdata)
{
    cJSON* jarr = cJSON_Parse(jstr);
    cJSON* jobj = NULL;
    cJSON* jitem = NULL;

    F* ssr_locs = schdata->ssr_locs;
    F* ssr_times = schdata->ssr_times;
    F* sch_dom = schdata->sch_dom;
    F ssr_dom[4], ms0;

    int num_ssrs = cJSON_GetArraySize(jarr);
    long involved = 0;

    if (num_ssrs < 3) {
        fprintf(stderr, "%s(%d): ltgpos expects number of input sensors >= 3, but got %d.\n",
                __FILE__, __LINE__, num_ssrs);
        return NULL;
    }
    if (num_ssrs > kMaxNumSsrs) {
        fprintf(stderr, "warning: ltgpos expects number of input sensors <= %d, but got %d. ignoring exceeding part.\n",
                kMaxNumSsrs, num_ssrs);
        num_ssrs = 64;
    }

    for (int i = 0; i < num_ssrs; i++) {
        jobj = cJSON_GetArrayItem(jarr, i);

        // j for 2 dimensions of sensor location.
        for (int j = 0; j < 2; j++) {
            cJSON_GetObjectItem_s(jitem, jobj, kJsonKeys[j + 1]);
            ssr_locs[i * 2 + j] = jitem->valuedouble;

            // Update coordinate domain with min or max values.
            if (!i)
                ssr_dom[2 * j + 1] = ssr_dom[2 * j] = ssr_locs[j];
            else if (ssr_locs[i * 2 + j] > ssr_dom[2 * j + 1])
                ssr_dom[2 * j + 1] = ssr_locs[i * 2 + j];
            else if (ssr_locs[i * 2 + j] < ssr_dom[2 * j])
                ssr_dom[2 * j] = ssr_locs[i * 2 + j];
        }

        // Get datetime & milliseconds.
        cJSON_GetObjectItem_s(jitem, jobj, "microsecond");
        if (i == 0) ms0 = (F)jitem->valueint / 1e4;
        ssr_times[i] = (F)jitem->valueint / 1e4 - ms0;
        // Assert the diff of seconds < 1 s.
        if (ssr_times[i] < 0) ssr_times[i] += 1e3;
    }
    involved = (mask << num_ssrs) - 1;

    // Filter out outlier sensors.
    F s = (ssr_dom[3] - ssr_dom[2]) * (ssr_dom[1] - ssr_dom[0]);
    F e0 = avg(ssr_locs, num_ssrs, 2, 0);
    F e1 = avg(ssr_locs, num_ssrs, 2, 1);
    F d0 = var(ssr_locs, num_ssrs, 2, 0, e0);
    F d1 = var(ssr_locs, num_ssrs, 2, 1, e1);

    if (num_ssrs > 3 && (s > 150 + 2.5 * num_ssrs || d0 + d1 > 20 + 0.5 * num_ssrs)) {
        for (int i = 0; i < num_ssrs; i++) {
            F d = abs(ssr_locs[i * 2] - e0) / d0 + abs(ssr_locs[i * 2 + 1] - e1) / d1;
            if (d > 0.5 + 0.05 * num_ssrs) {
                involved ^= mask << i;
            }
        }

        for (int i = 0; i < num_ssrs; i++) {
            if (!(involved & mask << i)) continue;
            for (int j = 0; j < 2; j++) {
                if (i == log2(involved)) ssr_dom[2 * j + 1] = ssr_dom[2 * j] = ssr_locs[j];
                else if (ssr_locs[i * 2 + j] > ssr_dom[2 * j + 1])
                    ssr_dom[2 * j + 1] = ssr_locs[i * 2 + j];
                else if (ssr_locs[i * 2 + j] < ssr_dom[2 * j])
                    ssr_dom[2 * j] = ssr_locs[i * 2 + j];
            }
        }
    }

    // Generate search domain with expand ratio.
    F area = (ssr_dom[1] - ssr_dom[0]) * (ssr_dom[3] - ssr_dom[2]);
    F exp_ratio = gSchDomRatio * (area > 8 ? 1 : sqrt(8 / area));
    for (int i = 0; i < 4; i++) {
        sch_dom[i] = (exp_ratio / 2 + 0.5) * ssr_dom[i] -
                     (exp_ratio / 2 - 0.5) * ssr_dom[i % 2 ? i - 1 : i + 1];
    }

    schdata->num_ssrs = num_ssrs;
    schdata->involved = involved;
    return jarr;
}


char* formatRetJsonStr(schdata_t* schdata, cJSON* jarr)
{
    int num_ssrs = schdata->num_ssrs;
    long involved = schdata->involved;
    F* ssr_locs = schdata->ssr_locs;
    F* ssr_times = schdata->ssr_times;
    F* out_ans = schdata->out_ans;

    cJSON* jobj = cJSON_GetArrayItem(jarr, log2(involved));
    cJSON* jitem;
    cJSON_GetObjectItem_s(jitem, jobj, "microsecond");
    F ms0 = (F)jitem->valueint / 1e4;
    cJSON_GetObjectItem_s(jitem, jobj, "datetime");
    char* datetime = jitem->valuestring;
    if ((out_ans[0] += ms0) < 0) {
        out_ans[0] += 1e3;
        getPrevDatetime(datetime, 1);
    }

    F all_dist[kMaxNumSsrs], all_dtime[kMaxNumSsrs];
    for (int i = 0; i < num_ssrs; i++) {
        // ===========================
        // TODO : get distance in 2d or 3d ?
        // ===========================
        all_dist[i] = getGeoDistance2d_H(ssr_locs[i * 2], ssr_locs[i * 2 + 1], out_ans[1], out_ans[2]);
                    // getGeoDistance3d_H(ssr_locs[i * 3], ssr_locs[i * 3 + 1], ssr_locs[i * 3 + 2], out_ans[1], out_ans[2], out_ans[3]);

        all_dtime[i] = ssr_times[i] - out_ans[0];
        all_dtime[i] += all_dtime[i] < 0 ? 1e3 : 0;
    }

    int num_involved = 0, is_involved[kMaxNumSsrs] = { 0 }, num_cur = 0;
    F us[kMaxNumSsrs], itdfs[kMaxNumSsrs], current;
    char* nodes[kMaxNumSsrs];
    for (int i = 0; i < num_ssrs; i++) {
        if (!(involved & mask << i)) continue;
        jobj = cJSON_GetArrayItem(jarr, i);
        is_involved[i] = 1;
        cJSON_GetObjectItem_s(jitem, jobj, "signal_strength");
        us[num_involved] = jitem->valuedouble;
        cJSON_GetObjectItem_s(jitem, jobj, "node");
        nodes[num_involved] = jitem->valuestring;
        itdfs[num_involved] = calItdf(all_dist[i], us[num_involved]);
        if (all_dist[i] < 150) {
            current += itdfs[num_involved];
            ++num_cur;
        }
        ++num_involved;
    }
    // qsort(itdfs, num_involved, sizeof(F), cmpItdf);
    // F current = num_involved % 2 ? itdfs[num_involved / 2] : (itdfs[num_involved / 2] + itdfs[num_involved / 2 - 1]) / 2;

    // ===========================
    // TODO : altitude calculation
    // ===========================
    out_ans[3] = 0;

    jobj = cJSON_CreateObject();
    cJSON_AddItemToObject(jobj, "datetime", cJSON_CreateString(datetime));
    // Create cJSON_Item from out_ans.
    for (int i = 0; i < 5; ++i) {
        cJSON_AddItemToObject(jobj, kJsonKeys[i], cJSON_CreateNumber(out_ans[i]));
    }
    cJSON_AddItemToObject(jobj, "current", cJSON_CreateNumber(current / num_cur));
    cJSON_AddItemToObject(jobj, "raw", jarr);
    cJSON_AddItemToObject(jobj, "allDist", cJSON_CreateDoubleArray(all_dist, num_ssrs));
    cJSON_AddItemToObject(jobj, "allDtime", cJSON_CreateDoubleArray(all_dtime, num_ssrs));
    cJSON_AddItemToObject(jobj, "isInvolved", cJSON_CreateIntArray(is_involved, num_ssrs));
    cJSON_AddItemToObject(jobj, "involvedNodes", cJSON_CreateStringArray((const char**)nodes, num_involved));
    cJSON_AddItemToObject(jobj, "referNode", cJSON_CreateString(nodes[0]));
    cJSON_AddItemToObject(jobj, "basicNodes", cJSON_CreateStringArray((const char**)nodes, 3));
    cJSON_AddItemToObject(jobj, "involvedSignalStrength", cJSON_CreateDoubleArray(us, num_involved));
    cJSON_AddItemToObject(jobj, "involvedResultCurrent", cJSON_CreateDoubleArray(itdfs, num_involved));

    char* ret_str = cJSON_PrintUnformatted(jobj);
    cJSON_Delete(jobj);
    return ret_str;
}

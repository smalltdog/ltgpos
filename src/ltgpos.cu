#include "ltgpos.h"


ssrinfo_t gSsrInfo;
grdinfo_t gGrdInfo;
bool isInit = false;


int initSysInfo()
{
    if (malloc_s(&gGrdInfo.houts, kMaxGrdNum)) return 1;
    if (cudaMalloc_s(&gGrdInfo.douts, kMaxGrdNum)) return 1;
    if (cudaMalloc_s(&gSsrInfo.ssr_locs, kMaxNumSsrs * 2)) return 1;
    if (cudaMalloc_s(&gSsrInfo.ssr_times, kMaxNumSsrs)) return 1;
    isInit = true;
    return 0;
}


void freeSysInfo()
{
    free(gGrdInfo.houts);
    cudaFree(gGrdInfo.douts);
    cudaFree(gSsrInfo.ssr_locs);
    cudaFree(gSsrInfo.ssr_times);
    isInit = false;
}


// Divide & conquer.
long dac_search(schdata_t& d)
{
    long involved = 0, prev_involved;
    // printf("%lx\n", d.involved);
    std::vector<long> combs = comb_mapper(d.involved);
    for (int i = 0; i != combs.size(); i++) {
        // printf("sub: %lx\n", combs[i]);
        d.involved = combs[i];
        grid_search(&gSsrInfo, &gGrdInfo, &d);
        // double* out_ans = d.out_ans;
        // printf("%7.4lf  %8.4lf  %.4lf\n", out_ans[1], out_ans[2], out_ans[4]);
        prev_involved = involved;
        involved |= d.out_ans[4] > kGoodThres ? dac_search(d) : combs[i];
        d.involved = involved;
        grid_search(&gSsrInfo, &gGrdInfo, &d);
        if (d.out_ans[4] > kGoodThres) involved = prev_involved;
    }
    // printf("fin: %lx\n", involved);
    return involved;
}


char* ltgpos(char* str)
{
    if (!isInit && initSysInfo()) {
        fprintf(stderr, "%s(%d): failed to initialize sysinfo.\n", __FILE__, __LINE__);
        return NULL;
    }

    schdata_t schdata;
    // Ensure jarr is deleted before return.
    cJSON* jarr = parseJsonStr(str, &schdata);
    if (!jarr) return NULL;

    grid_search(&gSsrInfo, &gGrdInfo, &schdata);
    if (schdata.out_ans[4] >= kGoodThres) {
        long prev_involved = schdata.involved;
        long involved = dac_search(schdata);
        schdata.involved = involved ? involved : prev_involved;
        grid_search(&gSsrInfo, &gGrdInfo, &schdata);
    }

    #ifdef TEST
    double* out_ans = schdata.out_ans;
    // double* sch_dom = schdata.sch_dom;
    // double* ssr_locs = schdata.ssr_locs;
    printf("%7.4lf  %8.4lf  %.4lf\n", out_ans[1], out_ans[2], out_ans[4]);
    // printf("%7.4lf  %7.4lf  %8.4lf %8.4lf\n", sch_dom[0], sch_dom[1], sch_dom[2], sch_dom[3]);
    // printf("%.2lf\n", (sch_dom[1] - sch_dom[0]) * (sch_dom[3] - sch_dom[2]));
    // printf("%d\n", schdata.num_ssrs);
    // for (int i = 0; i < schdata.num_ssrs; i++) printf("%.4lf, %.4lf\n", ssr_locs[i * 2], ssr_locs[i * 2 + 1]);
    #endif

    // Ensure the string returned is deallocated after use.
    return formatRetJsonStr(&schdata, jarr);
}

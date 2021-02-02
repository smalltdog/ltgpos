#include "ltgpos.h"


ssrinfo_t gSsrInfos[kNumSchs];
grdinfo_t gGrdInfos[kNumSchs];
bool isInit = false;


int initSysInfo()
{
    if (malloc_s(&gGrdInfos[0].houts, gMaxGridSize)) return 1;
    gGrdInfos[1].houts = gGrdInfos[0].houts;
    if (cudaMalloc_s(&gGrdInfos[0].douts, gMaxGridSize)) return 1;
    gGrdInfos[1].douts = gGrdInfos[0].douts;

    if (cudaMalloc_s(&gSsrInfos[0].ssr_locs, kMaxNumSsrs * 3)) return 1;
    gSsrInfos[1].ssr_locs = gSsrInfos[0].ssr_locs;
    if (cudaMalloc_s(&gSsrInfos[0].ssr_times, kMaxNumSsrs)) return 1;
    gSsrInfos[1].ssr_times = gSsrInfos[0].ssr_times;

    isInit = true;
    return 0;
}


void freeSysInfo()
{
    free(gGrdInfos[0].houts);
    cudaFree(gGrdInfos[0].douts);
    cudaFree(gSsrInfos[0].ssr_locs);
    cudaFree(gSsrInfos[0].ssr_times);
    isInit = false;
}


// int set_cfg(int num_sensors, int grid_size)
// {
//     gMaxGridSize <= kMaxNumThreads ? (gMaxGridSize = grid_size) :
//     fprintf(stderr, "%s(%d): grid size > the upper limit of concurrent threads.\n",
//             __FILE__, __LINE__);
//     freeSysInfo();
//     return initSysInfo();
// }


// Divide & conquer.
long dac_search(schdata_t& d)
{
    long involved = 0;
    // printf("%lx\n", d.involved);
    std::vector<long> combs = comb_mapper(d.involved);
    for (int i = 0; i != combs.size(); i++) {
        // printf("sub: %lx\n", combs[i]);
        d.involved = combs[i];
        grid_search(gSsrInfos, gGrdInfos, &d);
        // F* out_ans = d.out_ans;
        // printf("%7.4lf  %8.4lf  %.4lf\n", out_ans[1], out_ans[2], out_ans[4]);
        involved |= d.out_ans[4] > gGoodThres ? dac_search(d) : combs[i];
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
    F* out_ans = schdata.out_ans;

    // Ensure jarr is deleted before return.
    cJSON* jarr = parseJsonStr(str, &schdata);
    if (!jarr) return NULL;

    grid_search(gSsrInfos, gGrdInfos, &schdata);
    // printf("[scd] %lx %d\n", schdata.involved, schdata.num_ssrs);
    for (int i = 0; i < 5; i++) {
        if (out_ans[4] < gGoodThres) break;
        long prev_involved = schdata.involved;
        long new_involved = dac_search(schdata);
        // printf("[prev] %lx\n", prev_involved);
        // printf("[new] %lx\n", new_involved);
        if (!new_involved) {
            schdata.involved = prev_involved;
            break;
        }
        schdata.involved = new_involved;
        grid_search(gSsrInfos, gGrdInfos, &schdata);
        if (schdata.involved == prev_involved) break;
    }
    // printf("[scd] %lx\n", schdata.involved);
    grid_search(gSsrInfos, gGrdInfos, &schdata);

    #ifdef TEST
    // F* sch_dom = schdata.sch_dom;
    printf("%7.4lf  %8.4lf  %.4lf\n", out_ans[1], out_ans[2], out_ans[4]);
    // printf("%7.4lf  %7.4lf  %8.4lf %8.4lf\n", sch_dom[0], sch_dom[1], sch_dom[2], sch_dom[3]);
    // printf("%d\n", schdata.num_ssrs);

    // F* ssr_locs = schdata.ssr_locs;
    // for (int i = 0; i < schdata.num_ssrs; i++) printf("%.4lf, %.4lf\n", ssr_locs[i * 2], ssr_locs[i * 2 + 1]);
    // printf("\n");
    #endif

    // Ensure the string returned is deallocated after use.
    return formatRetJsonStr(&schdata, jarr);
}

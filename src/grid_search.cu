#include "grid_search.h"


__constant__ __device__ long dmask = 0x1;


__global__ void calGirdGoodness2d_G(ssrinfo_t sinfo, grdinfo_t ginfo)
{
    int num_ssrs = sinfo.num_ssrs;
    long involved = sinfo.involved;
    F* ssr_locs = sinfo.ssr_locs;
    F* ssr_times = sinfo.ssr_times;

    F x = ginfo.sch_dom[0] + ginfo.grd_inv[0] * threadIdx.x;
    F y = ginfo.sch_dom[2] + ginfo.grd_inv[1] * blockIdx.x;
    F t0, dt, err = 0;

    for (int i = 0; i < num_ssrs; i++) {
        if (!(involved & dmask << i)) continue;
        dt = getGeoDistance2d_D(ssr_locs[i * 2], ssr_locs[i * 2 + 1], x, y) / C;

        if (involved & -involved & dmask << i) { t0 = dt; continue; }  // Is referrence sensor
        dt -= t0 + ssr_times[i];
        err += dt * dt * 1e6;
    }
    err /= num_ssrs - 2;
    ginfo.douts[blockIdx.x * blockDim.x + threadIdx.x] = err;
}


// int dump_to_file(F* outs, int grid_sizes[3], const char* filename)
// {
//     FILE* fp = NULL;
//     fp = fopen(filename, "w");
//     if (!fp) {
//         fprintf(stderr, "%s(%d): failed to open file %s.\n", __FILE__, __LINE__, filename);
//         return 1;
//     }

//     for (int i = 0; i < grid_sizes[2]; i++) {
//         for (int j = 0; j < grid_sizes[1]; j++) {
//             for (int k = 0; k < grid_sizes[0]; k++) {
//                 fprintf(fp, "%8.2f ", outs[i * grid_sizes[1] * grid_sizes[0] + j * grid_sizes[1] + k]);
//             }
//             fprintf(fp, "\n");
//         }
//         fprintf(fp, "\n");
//     }
//     fclose(fp);
//     printf("[Dump] outputs dumped to %s\n", filename);
//     return 0;
// }


void grid_search(ssrinfo_t* sinfos, grdinfo_t* ginfos, schdata_t* schdata)
{
    ssrinfo_t* ssrinfo;
    grdinfo_t* grdinfo;

    int num_ssrs = schdata->num_ssrs;
    F* ssr_locs  = schdata->ssr_locs;
    F* ssr_times = schdata->ssr_times;
    F* out_ans   = schdata->out_ans;

    int grd_size, grd_sizes[2];
    F* sch_dom, * grd_inv, * houts, min_err;

    for (int i = 0; i < kNumSchs; i++) {
        ssrinfo = &sinfos[i];
        grdinfo = &ginfos[i];

        sch_dom = grdinfo->sch_dom;
        grd_inv = grdinfo->grd_inv;
        houts = grdinfo->houts;

        // Initialize search domain.
        if (!i) memcpy(sch_dom, schdata->sch_dom, 4 * sizeof(F));
        // Generate search domain based on result of previous search.
        else {
            for (int j = 0; j < 4; j++) {
                sch_dom[j] = out_ans[j/2+1] + grd_inv[j/2] * kNxtSchDomInvs * ((j % 2) ? 1 : -1);
            }
        }
        // Do 3D search in height of 0 ~ 20 km.

        for (int j = 0; j < 2; j++) {
            grd_inv[j] = (j == 2) ? 1 : max((sch_dom[j*2+1] - sch_dom[j*2]) / (gMaxGridNum - 1) * (1.02), 2e-5);
            // ===========
            // TODO : 1.02
            // ===========
            grd_sizes[j] = (sch_dom[j*2+1] - sch_dom[j*2]) / grd_inv[j] + 1;
        }
        grd_size = grd_sizes[0] * grd_sizes[1];

        ssrinfo->num_ssrs = schdata->num_ssrs;
        ssrinfo->involved = schdata->involved;
        cudaMemcpy(ssrinfo->ssr_locs, ssr_locs, num_ssrs * 2 * sizeof(F), cudaMemcpyHostToDevice);
        cudaMemcpy(ssrinfo->ssr_times, ssr_times, num_ssrs * sizeof(F), cudaMemcpyHostToDevice);

        dim3 grid(grd_sizes[1]), block(grd_sizes[0]);
        calGirdGoodness2d_G <<<grid, block>>> (*ssrinfo, *grdinfo);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s(%d): %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));
            out_ans[4] = -1;
            return;
        }
        cudaMemcpy(houts, grdinfo->douts, grd_size * sizeof(F), cudaMemcpyDeviceToHost);

        min_err = houts[0];
        int min_idx = 0;
        for (int j = 1; j < grd_size; j++) {
            if (houts[j] >= min_err) continue;
            min_err = houts[j];
            min_idx = j;
        }
        // dump_to_file(houts, grd_sizes, "test/gridres.csv");

        out_ans[1] = sch_dom[0] + min_idx % grd_sizes[0] * grd_inv[0];
        out_ans[2] = sch_dom[2] + min_idx / grd_sizes[0] % grd_sizes[1] * grd_inv[1];
    }

    int ref_idx = log2(schdata->involved);
    out_ans[0] = ssr_times[ref_idx] - getGeoDistance2d_H(ssr_locs[ref_idx * 2], ssr_locs[ref_idx * 2 + 1],out_ans[1], out_ans[2]) / C;
    out_ans[4] = min_err;
    return;
}

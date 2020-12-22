#include "grid_search.h"


const int kNumSearches = 1;
const int kNxtSchDomInvs = 2;


__global__ void calGirdGoodness2d_G(info_t info, int num_sensors, F* sensor_locs, F* sensor_times)
{
    F basetime, dtime, err = 0;
    F x = info.sch_dom[0] + info.grid_inv * threadIdx.x;
    F y = info.sch_dom[2] + info.grid_inv * blockIdx.x;
    int num_dims = info.is3d ? 3 : 2;

    basetime = getGeoDistance2d_D(sensor_locs[0], sensor_locs[1], x, y);
    for (int i = 1; i < num_sensors; i++) {
        dtime = getGeoDistance2d_D(sensor_locs[i * num_dims], sensor_locs[i * num_dims + 1], x, y) / C -
                basetime - sensor_times[i];
        err += dtime * dtime * 1e6;
    }
    err /= num_sensors - 2;
    info.outs_d[blockIdx.x * blockDim.x + threadIdx.x] = err;
}


__global__ void calGirdGoodness3d_G(info_t info, int num_sensors, F* sensor_locs, F* sensor_times)
{
    F basetime, dtime, err = 0;
    F x = info.sch_dom[0] + info.grid_inv * threadIdx.x;
    F y = info.sch_dom[2] + info.grid_inv * blockIdx.x;
    F z = info.sch_dom[3] + info.grid_inv * blockIdx.y;

    basetime = getGeoDistance3d_D(sensor_locs[0], sensor_locs[1], sensor_locs[2], x, y, z);
    for (int i = 1; i < num_sensors; i++) {
        dtime = getGeoDistance3d_D(sensor_locs[i * 3], sensor_locs[i * 3 + 1], sensor_locs[i * 3 + 2],
                                   x, y, z) / C - basetime - sensor_times[i];
        err += dtime * dtime * 1e6;
    }
    err /= num_sensors - 2;
    info.outs_d[blockIdx.y * blockDim.x * gridDim.x +
                blockIdx.x * blockDim.x + threadIdx.x] = err;
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


void grid_search(sysinfo_t* sysinfo, int num_sensors, F* sensor_locs, F* sensor_times, F results[5])
{
    info_t* info;
    int grid_size, res_bytes;
    int grid_sizes[3];
    int num_dims = sysinfo->nodes[1].is3d ? 3 : 2;
    F* sch_dom, * outs_h, min_err, grid_inv;
    F* sensor_locs_d = sysinfo->sensor_locs_d;
    F* sensor_times_d = sysinfo->sensor_times_d;

    for (int i = 0; i < kNumSearches; i++) {
        info = &sysinfo->nodes[i];
        sch_dom = info->sch_dom;
        outs_h = info->outs_h;
        grid_inv = info->grid_inv;

        // Generate search domain based on result of previous search.
        if (i != 0) {
            for (int j = 0; j < 4; j++) {
                info->sch_dom[j] = results[j % 2 + 1] + info->grid_inv *
                                   kNxtSchDomInvs * ((j % 2) ? 1 : -1);
            }
            // Do 3D search in height of 0 ~ 20 km.
            sch_dom[4] = 0;
            sch_dom[5] = info->is3d ? 20.0 / 100 : 0;
        }

        for (int j = 0; j < 3; j++) {
            grid_sizes[j] = (sch_dom[j * 2 + 1] - sch_dom[j * 2]) / grid_inv + 1;
        }

        grid_size = grid_sizes[0] * grid_sizes[1] * grid_sizes[2];
        res_bytes = grid_size * sizeof(F);

        cudaMemcpy(sensor_locs_d, sensor_locs, num_sensors * num_dims * sizeof(F),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(sensor_times_d, sensor_times, num_sensors * sizeof(F),
                   cudaMemcpyHostToDevice);

        // 3D grid search
        if (info->is3d && i != 0) {
            dim3 grid(grid_sizes[1], grid_sizes[2]);
            dim3 block(grid_sizes[0]);
            calGirdGoodness3d_G <<<grid, block>>> (*info, num_sensors, sensor_locs_d, sensor_times_d);
        }
        // 2D grid search
        else {
            dim3 grid(grid_sizes[1]), block(grid_sizes[0]);
            calGirdGoodness2d_G <<<grid, block>>> (*info, num_sensors, sensor_locs_d, sensor_times_d);
        }
        cudaMemcpy(outs_h, info->outs_d, res_bytes, cudaMemcpyDeviceToHost);

        min_err = outs_h[0];
        int min_idx = 0;
        for (int i = 1; i < grid_size; i++) {
            if (outs_h[i] >= min_err) continue;
            min_err = outs_h[i];
            min_idx = i;
        }
        // dump_to_file(outs_h, grid_sizes, "test/gridres.csv");

        results[1] = sch_dom[0] + min_idx % grid_sizes[0] * grid_inv;
        results[2] = sch_dom[2] + min_idx / grid_sizes[0] % grid_sizes[1] * grid_inv;
        results[3] = sch_dom[4] + min_idx / grid_sizes[0] / grid_sizes[1] * grid_inv;
    }
    // Assert results[3] == 0 and grid_sizes[2] == 1 when in 2D search.
    results[0] = sensor_times[0] - getGeoDistance3d_H(sensor_locs[0], sensor_locs[1], sensor_locs[2],
                                                      results[1], results[2], results[3]) / C;
    results[3] *= 100;
    results[4] = min_err;
    return;
}

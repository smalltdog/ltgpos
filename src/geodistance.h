#pragma once

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


#define C   299.792458
#define PI  3.14159265358979323846

#define RA  6378.137
#define RB  6356.7523142
#define OBLATE  (1 / 298.257223563)


F getGeoDistance2d_H(F lat1, F lon1, F lat2, F lon2);

__device__ F getGeoDistance2d_D(F lat1, F lon1, F lat2, F lon2);


// The unit of asl is 100 km to match the magnitude of lat and lon.
inline F getGeoDistance3d_H(F lat1, F lon1, F lat2, F lon2, F asl1, F asl2) {
    F dist = getGeoDistance2d_H(lat1, lon1, lat2, lon2);
    F dasl = (asl1 - asl2) * 100;
    return sqrt(dist * dist + dasl * dasl);
}

// The unit of asl is 100 km to match the magnitude of lat and lon.
__device__ F getGeoDistance3d_D(F lat1, F lon1, F lat2, F lon2, F asl1, F asl2);


// inline F getCartesianDistance2d_H(F x1, F y1, F x2, F y2) {
//     F dx = x1 - x2;
//     F dy = y1 - y2;
//     return sqrt(dx * dx + dy * dy);
// }

// __device__ F getCartesianDistance2d_D(F x1, F y1, F x2, F y2);

// inline F getCartesianDistance3d_H(F x1, F y1, F x2, F y2, F z1, F z2) {
//     F dist = getCartesianDistance2d_H(x1, y1, x2, y2);
//     return sqrt(dist * dist + (z1 - z2) * (z1 - z2));
// }

// __device__ F getCartesianDistance3d_D(F x1, F y1, F x2, F y2, F z1, F z2);

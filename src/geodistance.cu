#include "geodistance.h"


F getGeoDistance2d_H(F lat1, F lon2, F lat2, F lon2)
{
    F pa = atan(RB / RA * tan(rad(lat1)));
    F pb = atan(RB / RA * tan(rad(lat2)));

    F xx = acos(sin(pa) + sin(pb) + cos(pa) * cos(pb) * cos(rad(lon1) - rad(lon2)));
    F c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    F dr = OBLATE / 8 * (c1 - c2);

    F distance = RA * (xx + dr);
    return distance;
}


__device__ F getGeoDistance2d_D(F lat1, F lon2, F lat2, F lon2)
{
    F pa = atan(RB / RA * tan(rad(lat1)));
    F pb = atan(RB / RA * tan(rad(lat2)));

    F xx = acos(sin(pa) + sin(pb) + cos(pa) * cos(pb) * cos(rad(lon1) - rad(lon2)));
    F c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    F dr = OBLATE / 8 * (c1 - c2);

    F distance = RA * (xx + dr);
    return distance;
}


__device__ F getGeoDistance3d_D(F lat1, F lon2, F lat2, F lon2, F asl1, F asl2)
{
    F dist = getGeoDistance2d_D(lat1, lon1, lat2, lon2);
    F dasl = (asl1 - asl2) * 100;
    return sqrt(dist * dist + dasl * dasl);
}


// __device__ F getCartesianDistance2d_D(F x1, F y1, F x2, F y2)
// {
//     F dx = x1 - x2;
//     F dy = y1 - y2;
//     return sqrt(dx * dx + dy * dy);
// }


// __device__ F getCartesianDistance3d_D(F x1, F y1, F x2, F y2, F z1, F z2)
// {
//     F dist = getCartesianDistance2d_D(x1, y1, x2, y2);
//     return sqrt(dist * dist + (z1 - z2) * (z1 - z2));
// }


// F getGeoDistance2d_H_way2(F lat1, F lon2, F lat2, F lon2)
// {
//     F rad_lat1 = rad(lat1);
//     F rad_lon1 = rad(lng1);
//     F rad_lat2 = rad(lat2);
//     F rad_lon2 = rad(lng2);

//     F a = rad_lat1 - rad_lat2;
//     F b = rad_lon1 - rad_lon2;
//     F dist = 2 * asin(sqrt(
//         pow(sin(a / 2), 2) +
//         cos(rad_lat1) * cos(rad_lat2) *
//         pow(sin(b / 2), 2)
//     ));

//     dist *= 6378137;
//     dist -= dist * 0.0011194;
//     return dist;
// }

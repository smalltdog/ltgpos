#include "geodistance.h"


inline F rad(F deg) {
    return deg * PI / 180.0;
}


__device__ F rad_d(F deg) {
    return deg * PI / 180.0;
}


F getGeoDistance2d_H(F lat1, F lon1, F lat2, F lon2)
{
    F pa = atan(RB / RA * tan(rad(lat1)));
    F pb = atan(RB / RA * tan(rad(lat2)));

    F xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad(lon1) - rad(lon2)));
    F c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    F dr = OBLATE / 8 * (c1 - c2);

    F distance = RA * (xx + dr);
    return distance;
}


__device__ F getGeoDistance2d_D(F lat1, F lon1, F lat2, F lon2)
{
    F pa = atan(RB / RA * tan(rad_d(lat1)));
    F pb = atan(RB / RA * tan(rad_d(lat2)));

    F xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad_d(lon1) - rad_d(lon2)));
    F c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    F dr = OBLATE / 8 * (c1 - c2);

    F distance = RA * (xx + dr);
    return distance;
}

#include "geodistance.h"


double getGeoDistance2d_H(double lat1, double lon1, double lat2, double lon2)
{
    double pa = atan(RB / RA * tan(rad(lat1)));
    double pb = atan(RB / RA * tan(rad(lat2)));

    double xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad(lon1) - rad(lon2)));
    double c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    double c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    double dr = OBLATE / 8 * (c1 - c2);

    double distance = RA * (xx + dr);
    return distance;
}


__device__
double getGeoDistance2d_D(double lat1, double lon1, double lat2, double lon2)
{
    double pa = atan(RB / RA * tan(rad(lat1)));
    double pb = atan(RB / RA * tan(rad(lat2)));

    double xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad(lon1) - rad(lon2)));
    double c1 = (sin(xx) - xx) * pow(sin(pa) + sin(pb), 2) / pow(cos(xx / 2), 2);
    double c2 = (sin(xx) + xx) * pow(sin(pa) - sin(pb), 2) / pow(cos(xx / 2), 2);
    double dr = OBLATE / 8 * (c1 - c2);

    double distance = RA * (xx + dr);
    return distance;
}

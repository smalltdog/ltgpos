from math import *


def get_geodistance(lat1, lon1, lat2, lon2):
    '''Calculates distance of two GCS coordinates (measured in meters).'''

    ra = 6378.137
    rb = 6356.7523142
    oblate = 1 / 298.257223563

    rad_lon1, rad_lat1, rad_lon2, rad_lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    pa = atan(rb / ra * tan(rad_lat1))
    pb = atan(rb / ra * tan(rad_lat2))

    xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad_lon1 - rad_lon2))
    c1 = (sin(xx) - xx) * (sin(pa) + sin(pb)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pa) - sin(pb)) ** 2 / cos(xx / 2) ** 2
    dr = oblate / 8 * (c1 - c2)
    return ra * (xx + dr)

# Amap distance calculation
# https://lbs.amap.com/demo/javascript-api/example/calcutation/calculate-distance-between-two-markers

from math import *


RA = ra = 6378.137
RB = rb = 6356.7523142
F  = 1 / 298.257223563


def get_geodistance(lat1, lon1, lat2, lon2):
    '''Calculates distance of two GCS coordinates (measured in kilometers).'''

    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    pa = atan((1 - F) * tan(lat1))
    pb = atan((1 - F) * tan(lat2))

    xx = sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(lon1 - lon2)
    xx = acos(xx) if xx <= 1 else 0
    c1 = (sin(xx) - xx) * (sin(pa) + sin(pb)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pa) - sin(pb)) ** 2 / cos(xx / 2) ** 2
    dr = F / 8 * (c1 - c2)
    return ra * (xx + dr)


def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + sin(dlon / 2) ** 2 * cos(lat1) * cos(lat2)
    return ra * 2 * asin(sqrt(a))


def vincenty(lat1, lon1, lat2, lon2, tol=1e-8):
    # tol is set to 1e-8 instead of 1e-12,
    # which usually converges in 3 iter and produces error in 1m.

    u1  = atan((1 - F) * tan(radians(lat1)));
    u2  = atan((1 - F) * tan(radians(lat2)));
    lon = radians(lon2) - radians(lon1);
    lambd = lon;
    dlambda = 1;
    it = 100;
    while (dlambda > tol and it != 0):
        sin_sigma = sqrt(pow(cos(u2) * sin(lambd), 2) + pow(cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lambd), 2))
        cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lambd)
        sigma = atan(sin_sigma / cos_sigma)

        sin_alpha = cos(u1) * cos(u2) * sin(lambd) / sin_sigma
        cos2_alpha = 1 - pow(sin_alpha, 2)
        cos_2sigma = cos_sigma - 2 * sin(u1) * sin(u2) / cos2_alpha
        c = F / 16 * cos2_alpha * (4 + F * (4 - 3 * cos2_alpha))
        lambda_p = lambd
        lambd = lon + (1 - c) * F * sin_alpha * (sigma + c * sin_sigma * (cos_2sigma + c * cos_sigma * (2 * pow(cos_2sigma, 2) - 1)))
        dlambda = abs(lambda_p - lambd)
        it -= 1
    print(100 - it)
    u_sq = cos2_alpha * 0.0067394967423
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    dsigma = B * sin_sigma * (cos_2sigma + B / 4 * (cos_sigma * (-1 + 2 * pow(cos_2sigma, 2)) - B / 6 * cos_2sigma * (-3 + 4 * pow(sin_sigma, 2)) * (-3 + 4 * pow(cos_2sigma, 2))))
    distance = RB * A * (sigma - dsigma)
    return distance


if __name__ == '__main__':
    from vincenty import vincenty as vincentty
    import pdb; pdb.set_trace()

#include "geodistance.h"


double getGeoDistance2d_H(double lat1, double lon1, double lat2, double lon2)
{
    double u1  = atan((1 - F) * tan(rad(lat1)));
    double u2  = atan((1 - F) * tan(rad(lat2)));
    double lon = rad(lon2) - rad(lon1);
    double lambda = lon;

    double sin_sigma, cos_sigma, sigma;
    double sin_alpha, cos2_alpha, cos_2sigma;
    double lambda_p, dlambda = 1;
    double c;
    int it = 10;
    while (dlambda > 1e-8 && --it) {
        sin_sigma = sqrt(pow(cos(u2) * sin(lambda), 2) + pow(cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lambda), 2));
        cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lambda);
        sigma = atan(sin_sigma / cos_sigma);

        sin_alpha = cos(u1) * cos(u2) * sin(lambda) / sin_sigma;
        cos2_alpha = 1 - pow(sin_alpha, 2);
        cos_2sigma = cos_sigma - 2 * sin(u1) * sin(u2) / cos2_alpha;
        c = F / 16 * cos2_alpha * (4 + F * (4 - 3 * cos2_alpha));
        lambda_p = lambda;
        lambda = lon + (1 - c) * F * sin_alpha * (sigma + c * sin_sigma * (cos_2sigma + c * cos_sigma * (2 * pow(cos_2sigma, 2) - 1)));
        dlambda = abs(lambda_p - lambda);
    }

    double u_sq = cos2_alpha * 0.0067394967423;
    double A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)));
    double B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)));
    double dsigma = B * sin_sigma * (cos_2sigma + B / 4 * (cos_sigma * (-1 + 2 * pow(cos_2sigma, 2)) - B / 6 * cos_2sigma * (-3 + 4 * pow(sin_sigma, 2)) * (-3 + 4 * pow(cos_2sigma, 2))));
    double distance = RB * A * (sigma - dsigma);
    return distance;
}


__device__
double getGeoDistance2d_D(double lat1, double lon1, double lat2, double lon2)
{
    double u1  = atan((1 - F) * tan(rad(lat1)));
    double u2  = atan((1 - F) * tan(rad(lat2)));
    double lon = rad(lon2) - rad(lon1);
    double lambda = lon;

    double sin_sigma, cos_sigma, sigma;
    double sin_alpha, cos2_alpha, cos_2sigma;
    double lambda_p, dlambda = 1;
    double c;
    int it = 10;
    while (dlambda > 1e-8 && --it) {
        sin_sigma = sqrt(pow(cos(u2) * sin(lambda), 2) + pow(cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lambda), 2));
        cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lambda);
        sigma = atan(sin_sigma / cos_sigma);

        sin_alpha = cos(u1) * cos(u2) * sin(lambda) / sin_sigma;
        cos2_alpha = 1 - pow(sin_alpha, 2);
        cos_2sigma = cos_sigma - 2 * sin(u1) * sin(u2) / cos2_alpha;
        c = F / 16 * cos2_alpha * (4 + F * (4 - 3 * cos2_alpha));
        lambda_p = lambda;
        lambda = lon + (1 - c) * F * sin_alpha * (sigma + c * sin_sigma * (cos_2sigma + c * cos_sigma * (2 * pow(cos_2sigma, 2) - 1)));
        dlambda = abs(lambda_p - lambda);
    }

    double u_sq = cos2_alpha * 0.0067394967423;
    double A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)));
    double B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)));
    double dsigma = B * sin_sigma * (cos_2sigma + B / 4 * (cos_sigma * (-1 + 2 * pow(cos_2sigma, 2)) - B / 6 * cos_2sigma * (-3 + 4 * pow(sin_sigma, 2)) * (-3 + 4 * pow(cos_2sigma, 2))));
    double distance = RB * A * (sigma - dsigma);
    return distance;
}


// ########################
//  Unknown Distance Formula
// ########################
// double u1 = atan((1 - F) * tan(rad(lat1)));
// double u2 = atan((1 - F) * tan(rad(lat2)));

// double xx = acos(sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(rad(lon1) - rad(lon2)));
// double c1 = (sin(xx) - xx) * pow(sin(u1) + sin(u2), 2) / pow(cos(xx / 2), 2);
// double c2 = (sin(xx) + xx) * pow(sin(u1) - sin(u2), 2) / pow(cos(xx / 2), 2);
// double dr = F / 8 * (c1 - c2);

// double distance = RA * (xx + dr);
// ########################


// ########################
//  Haversine Formula
// ########################
// double dlat = rad(lat2) - rad(lat1);
// double dlon = rad(lon2) - rad(lon1);
// double haversine = pow(sin(dlat / 2), 2) + pow(sin(dlon / 2), 2) * cos(rad(lat1)) * cos(rad(lat2));
// double distance = RA * 2 * asin(sqrt(haversine));
// ########################


// ########################
//  Vincenty Formula
// ########################
// double u1  = atan((1 - F) * tan(rad(lat1)));
// double u2  = atan((1 - F) * tan(rad(lat2)));
// double lon = rad(lon2) - rad(lon1);
// double lambda = lon;

// double sin_sigma, cos_sigma, sigma;
// double sin_alpha, cos2_alpha, cos_2sigma;
// double lambda_p, dlambda = 1;
// double c;
// int it = 10;
// while (dlambda > 1e-8 && --it) {
//     sin_sigma = sqrt(pow(cos(u2) * sin(lambda), 2) + pow(cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lambda), 2));
//     cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lambda);
//     sigma = atan(sin_sigma / cos_sigma);

//     sin_alpha = cos(u1) * cos(u2) * sin(lambda) / sin_sigma;
//     cos2_alpha = 1 - pow(sin_alpha, 2);
//     cos_2sigma = cos_sigma - 2 * sin(u1) * sin(u2) / cos2_alpha;
//     c = F / 16 * cos2_alpha * (4 + F * (4 - 3 * cos2_alpha));
//     lambda_p = lambda;
//     lambda = lon + (1 - c) * F * sin_alpha * (sigma + c * sin_sigma * (cos_2sigma + c * cos_sigma * (2 * pow(cos_2sigma, 2) - 1)));
//     dlambda = abs(lambda_p - lambda);
// }

// double u_sq = cos2_alpha * 0.0067394967423;
// double A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)));
// double B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)));
// double dsigma = B * sin_sigma * (cos_2sigma + B / 4 * (cos_sigma * (-1 + 2 * pow(cos_2sigma, 2)) - B / 6 * cos_2sigma * (-3 + 4 * pow(sin_sigma, 2)) * (-3 + 4 * pow(cos_2sigma, 2))));
// double distance = RB * A * (sigma - dsigma);
// ########################

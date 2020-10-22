import numpy as np
import os
import socket
import datetime
from math import *
from itertools import product

from sock import cuda_init, cuda_cal, cuda_end, get_msg


def get_distance(lng_a, lat_a, lng_b, lat_b):
    '''Calculates distance of two GCS coordinates (measured in meters).'''

    ra = 6378.137  # r of equator (km)
    f = 1 / 298.257223563
    rb = (1 - f) * ra
    flatten = (ra - rb) / ra  # oblateness of earth

    rad_lng_a = radians(lng_a)
    rad_lat_a = radians(lat_a)
    rad_lng_b = radians(lng_b)
    rad_lat_b = radians(lat_b)

    pa = atan(rb / ra * tan(rad_lat_a))
    pb = atan(rb / ra * tan(rad_lat_b))

    xx = acos(sin(pa) * sin(pb) + cos(pa) * cos(pb) * cos(rad_lng_a - rad_lng_b))
    c1 = (sin(xx) - xx) * (sin(pa) + sin(pb)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pa) - sin(pb)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = 1000 * ra * (xx + dr)
    return distance


def measure_i(sensorLocs, slng, slat, u):
    '''Caculates the median itdf for a set of lightning data.

    Args:
        sensorLocs: 2d-array of GCS coordinates of sensors.
        slng: longitude of lightning detection result.
        slat: latitude of lightning detection result.
        u: intensity data of lightning detection.

    Return:
        median of itdfs calculated.
    '''

    itdfs = []
    for i, l in enumerate(sensorLocs):
        if u[i] != -1:
            r = get_distance(l[0], l[1], slng, slat)
            itdfs.append((r / 100) ** 1.13 * exp((r - 100) / 1000000) / 3.576 * u[i])
    return np.median(itdfs)


def is_st_cond(sensorLocs, t):
    x = np.array(t)
    if x[x != -1].shape[0] < 3:
        return False
    for i, sa in enumerate(sensorLocs):
        for j, sb in enumerate(sensorLocs):
            if i == j:
                continue
            t_delta = abs(t[i] - t[j])
            d = get_distance(sa[0], sa[1], sb[0], sb[1])
            if t_delta >= d / 299.792458:
                return False
    return True


class Comb(object):

    def __init__(self, data):
        x = data[:, 1] + data[:, 2] * 1j  # convert to complex num and get unique locs
        idx = np.unique(x, return_index=True)[1]
        self.sensorLocs = data[idx][:, 1:-1]        

        ks, v = self.sensorLocs[:, 0] + self.sensorLocs[:, 1] * 1j, [[-1, -1]]
        self.sensorTimes = { k: v[:] for k in ks }
        self.data_c = np.concatenate((x[:, np.newaxis], data[:, [0, 3]]), axis=1)

        for c in self.data_c:
            self.sensorTimes[c[0]].append([c.real[1], c.real[2]])

    def get_comb(self):
        list_ = list(self.sensorTimes.values())
        return product(*list_)


def cal_comb(data, base_time, exe_dir, port=6666, host="127.0.0.1"):
    '''calculate results for combinations of one set of data.'''

    local_sock = socket.socket()

    # 命令行方式传入端口号
    command = "start " + exe_dir + " " + str(port)
    os.system(command)  # 启动cuda 进程

    comb = Comb(data)

    # 初始化
    init_msg, init_msgToString = get_msg(comb.sensorLocs[:, ::-1], type='init')
    cuda_init(host, port, local_sock, init_msg, init_msgToString)

    result = {}
    for tus in comb.get_comb():
        t = tuple([tu[0] for tu in tus])
        u = [tu[1] for tu in tus]

        if not is_st_cond(comb.sensorLocs, t):
            continue
        cal_msg, cal_msgToString = get_msg(t, type='cal')
        r = cuda_cal(local_sock, cal_msg, cal_msgToString)

        i = 0
        for tt in t:
            if t != -1:
                i += 1
        print(r.results_xyz[1], r.results_xyz[0], r.error)
        if not result or r.error  < result['err']:
            result['ms'] = r.time
            result['lng'] = r.results_xyz[1]
            result['lat'] = r.results_xyz[0]
            result['asl'] = r.results_xyz[2]
            result['err'] = r.error
            result['t'] = t
            result['u'] = u

    if result:
        result['time'] = datetime.datetime.strptime(base_time[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
        result['ms'] += base_time[1] / 10
        result['time'] += datetime.timedelta(seconds=result['ms'] // 1e6)
        result['ms'] %= 1e6
        result['itdf'] = measure_i(comb.sensorLocs, result['lng'], result['lat'], result['u'])

    end_msg, end_msgToString = get_msg(type='end')
    cuda_end(local_sock, end_msg, end_msgToString)

    if result:
        return [result['time'], result['ms'], result['lng'], result['lat'],
                result['asl'], result['err'], result['itdf']]
    else:
        return -1

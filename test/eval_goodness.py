import pandas as pd
import argparse
import json
from vincenty import vincenty

# from geodistance import *


parser = argparse.ArgumentParser()
parser.add_argument('--no', type=str)
args = parser.parse_args()

label = 'test/data/label_' + args.no + '.csv'
output = 'test/data/json_' + args.no + '.csv'


def goodness(lat, lon, locs, times, involved=None):
    err = 0
    t0 = 0
    n_involved = 0
    dt = [0 for _ in times]
    for i, t in enumerate(times):
        if involved is not None and involved[i] != 1:
            continue
        dt[i] = vincenty((locs[i * 2], locs[i * 2 + 1]), (lat, lon))
        t0 += dt[i] / 299.792458 - times[i]
        n_involved += 1
    t0 /= n_involved

    for i, t in enumerate(times):
        if involved is not None and involved[i] != 1:
            continue
        dt[i] -= t0 + times[i]
        err += dt[i] ** 2 * 1e6
    return err / (n_involved)


def main():
    df = pd.read_csv(label, sep=',', header=None)

    with open(output, encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            # if len(data['isInvolved']) != sum(data['isInvolved']):
            #     print()
            input = data['raw']
            ssr_locs = []
            ssr_times = []
            # t_base = input[0]['microsecond']
            t_base = 0
            for node in input:
                ssr_locs.append(node['latitude'])
                ssr_locs.append(node['longitude'])
                t = node['microsecond']
                dt = (t-t_base) / 1e4
                if dt < 0 :
                    dt += 1e3
                ssr_times.append(dt)
            g0 = goodness(df.iloc[i, 0], df.iloc[i, 1], ssr_locs, ssr_times, data['isInvolved'])
            g1 = goodness(data['latitude'], data['longitude'], ssr_locs, ssr_times, data['isInvolved'])
            print(f'{g0 - g1:.6f}')


if __name__ == '__main__':
    main()

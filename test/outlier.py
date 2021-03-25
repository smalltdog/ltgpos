import numpy as np
import matplotlib.pyplot as plt

from math import sqrt


def area(s):
    return (np.max(s[:, 0]) - np.min(s[:, 0])) * (np.max(s[:, 1]) - np.min(s[:, 1]))


for i, data in enumerate(datas):
    data = np.array(data)
    # data = np.array(data)
    # avg0 = np.mean(data[:, 0])
    # avg1 = np.mean(data[:, 1])
    # var0 = np.var(data[:, 0])
    # var1 = np.var(data[:, 1])

    # if area(data) <= 150 + 2.5 * len(data) and var0+var1 < 20 + 0.5 * len(data):
    #     continue

    # print(len(data), '%.2f' % area(data), '%.2f' % (var0 + var1), '%.2f' % np.max([(abs(d[0] - avg0) / var0 + abs(d[1] - avg1) / var1) for d in data]))

    if not (area(data) >= 200):
        continue

    fig, axs = plt.subplots(1, 2)
    mask = np.array([True for _ in range(len(data))])

    for i, d in enumerate(data):
        print("  ", d[0], d[1])
    #     if abs(d[0] - avg0) / var0 + abs(d[1] - avg1) / var1 > 0.5 + 0.05 * len(data):
    #         mask[i] = False
    # else:
    #     if (np.max(data[:, 0]) - np.min(data[:, 0])) * (np.max(data[:, 1]) - np.min(data[:, 1])) < 50:
    #         continue
    #     err = data[:, 0] - avg0 + data[:, 1] - avg1
    #     mask[np.where(err==np.max(err))] = False

    axs[0].scatter(data[:, 0], data[:, 1])
    axs[1].scatter(data[mask][:, 0], data[mask][:, 1])
    plt.show()

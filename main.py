#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
# from sklearn.cluster import AffinityPropagation

from comb import cal_comb


data_csv = './data_mod.csv'
coord_csv = './coordinate.csv'
result_csv = './result.csv'
time_delta = 0.0015 * 1e7  # unit: 0.1ms


def get_time_delta(t1, t2):
    '''return time delta in unit of 2nd element'''
    d1 = datetime.datetime.strptime(t1[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
    d2 = datetime.datetime.strptime(t2[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
    return (d2 - d1).seconds * 1e7 + t2[1] - t1[1]


def get_coordinate(df_coord, station_no):
    '''return lng and lat coordinate of station no'''
    df = df_coord[df_coord['mno'] == station_no[0]]
    return np.array(df[df['no'] == station_no[1]].loc[:, ['lng', 'lat']]).tolist()


def plot_data(data):
    '''plot the scatter of coordinate data'''
    print(data)
    plt.scatter(data[:, 1], data[:, 2])
    plt.show()


def plot_data2(data1, data3, data2):
    '''plot the scatter of coordinate data'''
    plt.scatter(data1[:, 0], data1[:, 1], c='g')
    plt.scatter(data3[:, 0], data3[:, 1], c='r')
    plt.scatter(data2[:, 0], data2[:, 1], c='b')
    plt.show()


def main():
    df = pd.read_csv(data_csv).iloc[:, [0, 1, 2, 3, 5]]
    df_coord = pd.read_csv(coord_csv)
    prev_time = 0

    with open(result_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'ms', 'lng', 'lat', 'asl', 'err', 'itdf'])

        n_iter = 0
        nr = 0
        r1 = []
        r2 = []

        for _, row in df.iterrows():
            # if ... TODO exception handler
            # 时间差小于阈值，扩展数据
            if prev_time and get_time_delta(prev_time, row[:2]) < time_delta:
                sub_data = [[get_time_delta(prev_time, row[:2])]]
                coord = get_coordinate(df_coord, row[2:4])
                if len(coord):
                    sub_data[0].extend(coord[0])
                    sub_data[0].append(row[4])
                    data = np.append(data, sub_data, axis=0)

            # 初始化新一组数据
            else:
                # 对前一组数据进行计算
                if prev_time and 10 >= data.shape[0] >= 3:
                    # plot_data(data)
                    # clustering = AffinityPropagation().fit(data[:, 1:])
                    # plot_data(data[clustering.labels_ == 0])

                    data[:, 0] /= 1e4  # convert time to ms
                    print(data)
                    r = cal_comb(data, prev_time, "./lighting.exe")
                    print('\n')
                    if r != -1:
                    #     writer.writerow(r)
                        rr = [r[2], r[3]]
                        r1.append(rr)
                        nr += 1

                data = np.zeros((1, 4))
                prev_time = 0
                coord = get_coordinate(df_coord, row[2:4])
                if len(coord) > 1:
                    coord = [coord[0]]
                if len(coord):
                    data[0, 1:3] = np.asanyarray(coord)
                    data[0, 3] = row[4]
                    prev_time = row[:2].to_list()
            
        #     n_iter += 1
        #     if n_iter > 500:
        #         break


        # n_iter = 0
        # for _, row in df.iterrows():
        #     # if ... TODO exception handler
        #     # 时间差小于阈值，扩展数据
        #     if prev_time and get_time_delta(prev_time, row[:2]) < time_delta:
        #         sub_data = [[get_time_delta(prev_time, row[:2])]]
        #         coord = get_coordinate(df_coord, row[2:4])
        #         if len(coord):
        #             sub_data[0].extend(coord[0])
        #             sub_data[0].append(row[4])
        #             data = np.append(data, sub_data, axis=0)

        #     # 初始化新一组数据
        #     else:
        #         # 对前一组数据进行计算
        #         if prev_time and 10 >= data.shape[0] >= 3:
        #             # plot_data(data)
        #             # clustering = AffinityPropagation().fit(data[:, 1:])
        #             # plot_data(data[clustering.labels_ == 0])

        #             data[:, 0] /= 1e4  # convert time to ms
        #             print(data)
        #             r = cal_comb2(data, prev_time, "./lighting.exe")
        #             if r != -1:
        #             #     writer.writerow(r)
        #                 rr = [r[2], r[3]]
        #                 r2.append(rr)

        #         data = np.zeros((1, 4))
        #         prev_time = 0
        #         coord = get_coordinate(df_coord, row[2:4])
        #         if len(coord) > 1:
        #             coord = [coord[0]]
        #         if len(coord):
        #             data[0, 1:3] = np.asanyarray(coord)
        #             data[0, 3] = row[4]
        #             prev_time = row[:2].to_list()
            
        #     n_iter += 1
        #     if n_iter > 500:
        #         break

        # dfr = pd.read_csv('test_data_0615.csv').iloc[:nr * 2, [3, 2]]
        # plot_data2(np.array(r1), np.array(r2), np.array(dfr))

        f.close()


if __name__ == '__main__':
    main()

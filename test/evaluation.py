import pandas as pd
import argparse
from vincenty import vincenty

# from geodistance import get_geodistance


parser = argparse.ArgumentParser()
parser.add_argument('--no', type=str)
args = parser.parse_args()

label = 'test/data/label_' + args.no + '.csv'
output = 'test/data/output_' + args.no + '.csv'
resanal = 'test/data/resanal_' + args.no + '.csv'


def hist(df: pd.DataFrame, attr: str, box: float, overflow: float):
    print(attr)
    cur = 0
    while cur < overflow:
        print(f'{cur + box:.1f}\t{sum((cur <= df).multiply(df < cur + box)) / len(df) * 100:.2f}')
        cur += box
    print(f'>{overflow}\t{sum(cur <= df) / len(df) * 100:.2f}\n')


df_l = pd.read_csv(label, sep=',', header=None)
df_o = pd.read_csv(output, sep=',', header=None)
df = pd.concat([df_l, df_o], axis=1)
df[df.shape[1]] = [vincenty((row[1], row[2]), (row[4], row[5])) if df.shape[1] != 7 else
                   vincenty((row[1], row[2]), (row[3], row[4])) for row in df.itertuples()]      # deprecated
df.to_csv(resanal, sep =',', index=False, header=False, float_format='%.4f')

hist(df.iloc[:, -1], 'Dist', box=0.2, overflow=5)
if df.shape[1] != 8:
    df = df[df.iloc[:, -1] < 1.6]
    hist(abs((df.iloc[:, 7] - df.iloc[:, 2]) / df.iloc[:, 2]), 'DCurrent', box=0.2, overflow=1)

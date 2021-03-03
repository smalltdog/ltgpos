import pandas as pd
import argparse
from geodistance import get_geodistance


parser = argparse.ArgumentParser()
parser.add_argument('--no', type=str)
args = parser.parse_args()

label = 'test/data/label_' + args.no + '.csv'
output = 'test/data/output_' + args.no + '.csv'
resanal = 'test/data/resanal_' + args.no + '.csv'


df_l = pd.read_csv(label, sep='\t', header=None)
df_o = pd.read_csv(output, sep='  ', header=None)
df = pd.concat([df_l, df_o], axis=1)
df.columns = [0, 1, 2, 3, 4]
df[5] = [get_geodistance(row[1], row[2], row[3], row[4]) for row in df.itertuples()]
df.to_csv(resanal, sep =',', index=False, header=False, float_format='%.4f')

import pandas as pd
import argparse


def cond(goodness, distance):
    if goodness > 40 and distance > 20:
        return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--no', type=str)
parser.add_argument('--no_out', type=str)
args = parser.parse_args()

input = 'test/data/input_' + args.no + '.csv'
label = 'test/data/label_' + args.no + '.csv'
output = 'test/data/output_' + args.no + '.csv'
resanal = 'test/data/resanal_' + args.no + '.csv'


df_i = pd.read_csv(input, sep='\t', header=None)
df_l = pd.read_csv(label, sep='\t', header=None)
df_r = pd.read_csv(resanal, sep=',', header=None)

input = 'test/data/input_' + args.no_out + '.csv'
label = 'test/data/label_' + args.no_out + '.csv'

rowidxs = [row[0] for row in df_r.itertuples() if cond(row[5], row[6])]

df_i.iloc[rowidxs].to_csv(input, index=False, header=False)
df_l.iloc[rowidxs].to_csv(label, sep ='\t', index=False, header=False, float_format='%.6f')

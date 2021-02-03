#%%
import pandas as pd
import argparse


input = 'test/input.csv'
label = 'test/label.csv'

output = 'test/output.csv'
resanal = 'test/resanal.csv'


parser = argparse.ArgumentParser()
parser.add_argument("--idx", help="display a square of a given number")
args = parser.parse_args()


def cond(goodness, distance):
    if goodness > 40 and distance > 20:
        return True
    return False


df_i = pd.read_csv(input, sep='\t', header=None)
df_l = pd.read_csv(label, sep='\t', header=None)
df_r = pd.read_csv(resanal, sep=',', header=None)

rowidxs = [row[0] for row in df_r.itertuples() if cond(row[5], row[6])]

input2 = input.split('.')[0] + '_' + args.idx + '.csv'
label2 = label.split('.')[0] + '_' + args.idx + '.csv'

df_i.iloc[rowidxs].to_csv(input2, index=False, header=False)
df_l.iloc[rowidxs].to_csv(label2, sep ='\t', index=False, header=False, float_format='%.6f')

import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--no', type=str)
args = parser.parse_args()

output = 'test/data/output_' + args.no + '.csv'
json = 'test/data/json_' + args.no + '.csv'

df = pd.read_csv('test/data/output.csv', sep=';', header=None)
df.iloc[:, :-1].to_csv(output, index=False, header=False)
df.iloc[:, -1].to_csv(json, index=False, header=False)

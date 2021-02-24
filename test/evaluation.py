import pandas as pd
from geodistance import get_geodistance


label = 'test/data/label_1036.csv'
output = 'test/output.csv'
resanal = 'test/data/resanal_1036.csv'


df_l = pd.read_csv(label, sep='\t', header=None)
df_o = pd.read_csv(output, sep='  ', header=None)
df = pd.concat([df_l, df_o], axis=1)
df.columns = [0, 1, 2, 3, 4]
df[5] = [get_geodistance(row[1], row[2], row[3], row[4]) for row in df.itertuples()]
df.to_csv(resanal, sep =',', index=False, header=False, float_format='%.4f')

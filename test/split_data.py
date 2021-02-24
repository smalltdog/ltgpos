import pandas as pd
import json


data = './data.csv'
input = './input.csv'
label = './label.csv'


def round6(value):
    return round(value, 6)

# Convert seperator from "," to ";" by replacing:
# (.*?),(.*?),(.*)
# $1;$2;$3
df = pd.read_csv(data, sep=';')

df["longitude"] = list(map(round6, df["longitude"]))
df["latitude"] = list(map(round6, df["latitude"]))

df["json"].to_csv(input, index=False, header=False)
df.drop("json", axis=1).to_csv(label, index=False, header=False, sep='\t')

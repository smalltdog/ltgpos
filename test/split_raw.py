import pandas as pd
import json


src = './src.csv'
dst = './dst.csv'


def round6(value):
    return(round(value, 6))


df = pd.read_csv(src, sep=',')
df["json"] = [json.loads(j)["raw"] for j in df["retjson"]]

df["longitude"] = list(map(round6, df["longitude"]))
df["latitude"] = list(map(round6, df["latitude"]))

df = df.drop("retjson", axis=1)
df.to_csv(dst, index=False)

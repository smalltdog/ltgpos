import pandas as pd
import json


df = pd.read_csv('test/data/json.csv', sep=';', header=None)
df["json"] = [json.loads(row[1].item())["raw"] for row in df.iterrows()]
df["json"].to_csv('test/data/input_.csv', index=False, header=False)

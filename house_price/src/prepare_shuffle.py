import os
import json

import numpy as np
import pandas as pd


df = pd.read_csv("../data/train.csv")
df = df.set_index("Id")

output_dir = "../data/interim/shuffled"

shuffle_candidates = df.select_dtypes(include=["float", "int"]).apply(lambda x: x.nunique()).sort_values(ascending=False).iloc[:10].index
cols_to_shuffle = np.random.choice(shuffle_candidates, 3, replace=False)
split_df = np.array_split(df, 3)

for i, i_df in enumerate(split_df):
    to_shuffle = cols_to_shuffle[i]
    col_values = i_df[to_shuffle].copy().values
    np.random.shuffle(col_values)
    i_df[to_shuffle] = col_values
    i_df.to_csv(os.path.join(output_dir, f"shuffle_{i}.csv"))

json.dump({i: c for i, c in enumerate(cols_to_shuffle)}, open(os.path.join(output_dir, "shuffled.json"), "wt"))
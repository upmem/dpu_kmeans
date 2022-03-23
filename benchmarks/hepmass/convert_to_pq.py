# -*- coding: utf-8 -*-
"""
Converts the Higgs dataset CSV to a parquet file for faster access.
"""

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv("data/all_test.csv")
df.mass = StandardScaler().fit_transform(df.mass.values.reshape(-1, 1))
df.to_parquet("./data/hepmass_test.pq", index=False, compression=None)

# %%
df = pd.read_csv("data/all_train.csv")
df.mass = StandardScaler().fit_transform(df.mass.values.reshape(-1, 1))
df.to_parquet("./data/hepmass_train.pq", index=False, compression=None)

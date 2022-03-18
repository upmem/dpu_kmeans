# -*- coding: utf-8 -*-
"""
Converts the Higgs dataset CSV to a parquet file for faster access.
"""

# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("data/Phones_accelerometer.csv", header=None, sep=",")
df.columns = df.columns.astype(str)
df.to_parquet("./data/higgs.pq", index=False, compression=None)

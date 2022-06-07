# -*- coding: utf-8 -*-
"""
Converts the Higgs dataset CSV to a parquet file for faster access.
"""

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar

df = dd.read_csv("HIGGS.csv", dtype=np.float32, header=None, sep=",")
df.columns = df.columns.astype(str)
with ProgressBar():
    df.to_parquet("higgs.pq", write_index=False, compression=None)

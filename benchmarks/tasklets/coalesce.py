#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

import pandas as pd

all_files = glob.glob("tasklets_*.pkl")

li = []

for filename in all_files:
    df = pd.read_pickle(filename)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=False)
frame.index.name = "tasklets"
frame.sort_index(inplace=True)

frame.to_csv("tasklets.csv")
frame.to_pickle("tasklets.pkl")

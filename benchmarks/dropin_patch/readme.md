# KMeans drop-in patch

Quick and dirty drop in patch for sklearn.cluster._kmeans to add perf counters to the KMeans class. Required to run most benchmarks in this project.

## Automatic install

Run `install_patch.sh` from this directory.

## Manual install

Copy and past to `[python env path]/lib/python3.8/site-packages/sklearn/cluster/_kmeans.py`

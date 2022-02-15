#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
import numpy as np
import sys
import time
import pandas as pd
from hurry.filesize import size
from tqdm import tqdm
from dpu_kmeans import DIMM_data, KMeans, _dimm

nfeatures = 8
min_nclusters = 15
max_nclusters = 15
loops = 10
verbose = False
tol = 1e-4

npoints_str = "1e5"
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
inner_times = []
iter = []

_dimm.load_kernel("kmeans", verbose)

data, tags, centers = make_blobs(
    int(float(npoints_str)), 8, centers=15, random_state=42, return_centers=True
)

data = data.astype(np.float32)
print("data size for {} points : {}".format(npoints_str, size(sys.getsizeof(data))))

dimm_data = DIMM_data(data)

_dimm.load_data(dimm_data, tol, verbose)

n_clusters_time = []
n_clusters_inner_time = []
n_clusters_iter = []

for n_clusters in n_cluster_set:

    timer = 0
    iter_counter = 0

    tic = time.perf_counter()

    kmeans = KMeans(n_clusters, n_init=loops, max_iter=500, tol=tol, verbose=True)
    centroids, iter_counter, inner_timer = kmeans.fit(dimm_data)

    toc = time.perf_counter()
    timer += toc - tic

    n_clusters_time.append(timer / loops)
    n_clusters_inner_time.append(inner_timer / loops)
    n_clusters_iter.append(iter_counter / loops)

times.append(n_clusters_time.copy())
inner_times.append(n_clusters_inner_time.copy())
iter.append(n_clusters_iter.copy())

if min_nclusters == max_nclusters:
    print("centroids:")
    print(centroids)

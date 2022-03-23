#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
from hurry.filesize import size
from sklearn.datasets import make_blobs
from tqdm import tqdm

from dpu_kmeans import KMeans, _dimm

nfeatures = 8
min_nclusters = 15
max_nclusters = 15
loops = 1
verbose = False
tol = 1e-4

npoints_str = "1e4"
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
inner_times = []
iter = []

_dimm.set_n_dpu(4)
_dimm.load_kernel("kmeans", verbose)

data, tags, centers = make_blobs(
    int(float(npoints_str)), 8, centers=15, random_state=42, return_centers=True
)

data = data.astype(np.float32)
print("data size for {} points : {}".format(npoints_str, size(sys.getsizeof(data))))

# _dimm.load_data(dimm_data, tol, verbose)

n_clusters_time = []
n_clusters_inner_time = []
n_clusters_iter = []

for n_clusters in tqdm(n_cluster_set, file=sys.stdout):

    timer = 0
    iter_counter = 0

    tic = time.perf_counter()

    kmeans = KMeans(n_clusters, n_init=loops, max_iter=500, tol=tol, verbose=True)
    centroids, iter_counter, inner_timer = kmeans.fit(data)

    toc = time.perf_counter()
    timer += toc - tic

    n_clusters_time.append(timer / loops)
    n_clusters_inner_time.append(inner_timer / loops)
    n_clusters_iter.append(iter_counter / loops)

times.append(n_clusters_time.copy())
inner_times.append(n_clusters_inner_time.copy())
iter.append(n_clusters_iter.copy())

# if min_nclusters == max_nclusters:
#     print("centroids:")
#     print(centroids)
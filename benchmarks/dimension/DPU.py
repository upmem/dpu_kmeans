#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import pandas as pd
from hurry.filesize import size
from sklearn.datasets import make_blobs
from tqdm import tqdm

from dpu_kmeans import DimmData, KMeans, _dimm

nfeatures = 8
min_nclusters = 3
max_nclusters = 32
loops = 10
verbose = False
tol = 1e-4

npoints = int(1e6)
n_dim_set = [4, 8, 16, 32]  # list(range(3, 33))
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
inner_times = []
iter = []

_dimm.load_kernel("kmeans", verbose)

for i_ndim, ndim in enumerate(n_dim_set):
    data, tags, centers = make_blobs(
        npoints, ndim, centers=15, random_state=42, return_centers=True
    )

    data = data.astype(np.float32)
    print("data size for {} dimensions : {}".format(ndim, size(sys.getsizeof(data))))

    dimm_data = DimmData(data)

    _dimm.load_data(dimm_data, tol, verbose)

    n_clusters_time = []
    n_clusters_inner_time = []
    n_clusters_iter = []

    for n_clusters in tqdm(n_cluster_set, file=sys.stdout):

        timer = 0
        iter_counter = 0

        tic = time.perf_counter()

        kmeans = KMeans(n_clusters, n_init=loops, max_iter=500, tol=tol, verbose=False)
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

    df_times = pd.DataFrame(
        times, index=n_dim_set[: i_ndim + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("DPU_times.pkl")

    df_times = pd.DataFrame(
        inner_times, index=n_dim_set[: i_ndim + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("DPU_inner_times.pkl")

    df_iter = pd.DataFrame(
        iter, index=n_dim_set[: i_ndim + 1], columns=n_cluster_set
    ).transpose()
    df_iter.to_pickle("DPU_iter.pkl")

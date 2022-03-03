#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
import numpy as np
import sys
import time
import pandas as pd
from hurry.filesize import size
from tqdm import tqdm
from dpu_kmeans import DimmData, KMeans, _dimm

nfeatures = 8
min_nclusters = 3
max_nclusters = 32
loops = 10
verbose = False
tol = 1e-4

test_set = ["1e4", "3e4", "1e5", "3e5", "1e6", "3e6", "1e7", "3e7", "1e8"]
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
inner_times = []
iter = []

_dimm.load_kernel("kmeans", verbose)

for i_npoints, npoints_str in enumerate(test_set):
    data, tags, centers = make_blobs(
        int(float(npoints_str)), 8, centers=15, random_state=42, return_centers=True
    )

    data = data.astype(np.float32)
    print("data size for {} points : {}".format(npoints_str, size(sys.getsizeof(data))))

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
        times, index=test_set[: i_npoints + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("DPU_times.pkl")

    df_times = pd.DataFrame(
        inner_times, index=test_set[: i_npoints + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("DPU_inner_times.pkl")

    df_iter = pd.DataFrame(
        iter, index=test_set[: i_npoints + 1], columns=n_cluster_set
    ).transpose()
    df_iter.to_pickle("DPU_iter.pkl")

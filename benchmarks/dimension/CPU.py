#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import pandas as pd
from hurry.filesize import size
from sklearn.cluster import k_means
from sklearn.datasets import make_blobs
from sklearnex import patch_sklearn
from tqdm import tqdm

patch_sklearn(verbose=False)

nfeatures = 8
min_nclusters = 3
max_nclusters = 32
loops = 10

npoints = int(1e6)
n_dim_set = [4, 8, 16, 32]  # list(range(3, 33))
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
iter = []

for i_ndim, ndim in enumerate(n_dim_set):
    data, tags, centers = make_blobs(
        npoints, ndim, centers=15, random_state=42, return_centers=True
    )

    data = data.astype(np.float32)
    print("data size for {} dimensions : {}".format(ndim, size(sys.getsizeof(data))))

    n_clusters_time = []
    n_clusters_iter = []

    for n_clusters in tqdm(n_cluster_set, file=sys.stdout):

        timer = 0
        iter_counter = 0
        inertia = np.inf

        for iloop in range(loops):
            tic = time.perf_counter()

            init = data[iloop * n_clusters : (iloop + 1) * n_clusters]

            results = k_means(
                data,
                n_clusters,
                init=init,
                n_init=1,
                return_n_iter=True,
                algorithm="full",
                tol=1e-4,
                max_iter=500,
                verbose=False,
                copy_x=False,
            )

            if results[2] < inertia:
                centroids = results[0]
                inertia = results[2]

            iter_counter += results[3]

            toc = time.perf_counter()
            timer += toc - tic

        n_clusters_time.append(timer / loops)
        n_clusters_iter.append(iter_counter / loops)

    times.append(n_clusters_time.copy())
    iter.append(n_clusters_iter.copy())

    if min_nclusters == max_nclusters:
        print("centroids:")
        print(centroids)

    df_times = pd.DataFrame(
        times, index=n_dim_set[: i_ndim + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("CPU_times.pkl")

    df_iter = pd.DataFrame(
        iter, index=n_dim_set[: i_ndim + 1], columns=n_cluster_set
    ).transpose()
    df_iter.to_pickle("CPU_iter.pkl")

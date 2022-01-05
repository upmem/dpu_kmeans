#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn.cluster import k_means
from sklearnex import patch_sklearn
import numpy as np
import sys
import time
import pandas as pd
from hurry.filesize import size
from tqdm import tqdm

patch_sklearn(verbose=False)

nfeatures = 8
min_nclusters = 3
max_nclusters = 32
loops = 10

test_set = ["1e4", "3e4", "1e5", "3e5", "1e6", "3e6", "1e7", "3e7", "1e8"]
n_cluster_set = list(range(min_nclusters, max_nclusters + 1))
times = []
iter = []

for i_npoints, npoints_str in enumerate(test_set):
    data, tags, centers = make_blobs(
        int(float(npoints_str)), 8, centers=15, random_state=42, return_centers=True
    )

    data = data.astype(np.float32)
    print("data size for {} points : {}".format(npoints_str, size(sys.getsizeof(data))))

    n_clusters_time = []
    n_clusters_iter = []

    for n_clusters in tqdm(n_cluster_set, file=sys.stdout):

        timer = 0
        iter_counter = 0
        inertia = np.inf

        for iloop in range(loops):
            init = data[iloop * n_clusters : (iloop + 1) * n_clusters]

            tic = time.perf_counter()
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
            toc = time.perf_counter()

            timer += toc - tic
            iter_counter += results[3]

            if results[2] < inertia:
                centroids = results[0]
                inertia = results[2]

        n_clusters_time.append(timer / loops)
        n_clusters_iter.append(iter_counter / loops)

    times.append(n_clusters_time.copy())
    iter.append(n_clusters_iter.copy())

    if min_nclusters == max_nclusters:
        print("centroids:")
        print(centroids)

    df_times = pd.DataFrame(
        times, index=test_set[: i_npoints + 1], columns=n_cluster_set
    ).transpose()
    df_times.to_pickle("CPU_times.pkl")

    df_iter = pd.DataFrame(
        iter, index=test_set[: i_npoints + 1], columns=n_cluster_set
    ).transpose()
    df_iter.to_pickle("CPU_iter.pkl")

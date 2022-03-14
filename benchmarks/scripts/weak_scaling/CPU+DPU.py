#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import pandas as pd
from hurry.filesize import size
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from dpu_kmeans import KMeans as DPU_KMeans
from dpu_kmeans import _dimm

nfeatures = 8
n_clusters = 16
n_init = 1
max_iter = 500
verbose = False
tol = 1e-4
random_state = 42

n_points_per_dpu = int(1e5)
n_dim = 16

n_dpu_set = [1, 2, 4, 8, 16, 32, 64]

DPU_times = []
DPU_inner_times = []
DPU_init_times = []
DPU_transform_and_load_times = []
DPU_iterations = []
DPU_scores = []
DPU_time_per_iter = []

CPU_times = []
CPU_iterations = []
CPU_scores = []
CPU_time_per_iter = []

for i_n_dpu, n_dpu in enumerate(tqdm(n_dpu_set, file=sys.stdout)):

    ##################################################
    #                   DATA GEN                     #
    ##################################################

    data, tags, centers = make_blobs(
        n_points_per_dpu * n_dpu,
        n_dim,
        centers=n_clusters,
        random_state=random_state,
        return_centers=True,
    )

    # data = data.astype(np.float32)
    print(f"raw data size for {n_dpu} dpus : {size(sys.getsizeof(data))}")

    ##################################################
    #                   DPU PERF                     #
    ##################################################

    # load the DPUS
    _dimm.free_dpus()
    tic = time.perf_counter()
    _dimm.set_n_dpu(n_dpu)
    _dimm.load_kernel("kmeans", verbose)
    toc = time.perf_counter()
    DPU_init_time = toc - tic

    # load the data in the DPUs
    tic = time.perf_counter()
    _dimm.load_data(data, verbose)
    toc = time.perf_counter()
    DPU_data_transform_and_load_time = toc - tic
    print(f"quantized data size for {n_dpu} dpus : {size(_dimm._data_size)}")

    # perform clustering on DPU
    tic = time.perf_counter()
    DPU_kmeans = DPU_KMeans(
        n_clusters,
        init="random",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        copy_x=False,
        random_state=random_state,
    )
    DPU_kmeans.fit(data)
    toc = time.perf_counter()

    DPU_centroids, DPU_iter_counter, DPU_inner_timer = (
        DPU_kmeans.cluster_centers_,
        DPU_kmeans.n_iter_,
        DPU_kmeans.dpu_run_time_,
    )
    DPU_timer = toc - tic

    ##################################################
    #                   CPU PERF                     #
    ##################################################

    # perform clustering on CPU
    tic = time.perf_counter()
    CPU_kmeans = KMeans(
        n_clusters,
        init="random",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        copy_x=False,
        random_state=random_state,
    )
    CPU_kmeans.fit(data)
    CPU_centroids, CPU_iter_counter = CPU_kmeans.cluster_centers_, CPU_kmeans.n_iter_
    toc = time.perf_counter()
    CPU_timer = toc - tic

    ##################################################
    #                   LOGGING                      #
    ##################################################

    DPU_times.append(DPU_timer)
    DPU_inner_times.append(DPU_inner_timer)
    DPU_iterations.append(DPU_iter_counter)
    DPU_time_per_iter.append(DPU_timer / DPU_iter_counter)
    DPU_init_times.append(DPU_init_time)
    DPU_transform_and_load_times.append(DPU_data_transform_and_load_time)

    CPU_times.append(CPU_timer)
    CPU_iterations.append(CPU_iter_counter)
    CPU_time_per_iter.append(CPU_timer / CPU_iter_counter)

    # rand index for CPU and DPU (measures the similarity of the clustering with the ground truth)
    DPU_scores.append(adjusted_rand_score(tags, DPU_kmeans.labels_))
    CPU_scores.append(adjusted_rand_score(tags, CPU_kmeans.labels_))

    # creating and exporting the dataframe at each iteration in case we crash early
    df = pd.DataFrame(
        {
            "DPU_times": DPU_times,
            "DPU_inner_times": DPU_inner_times,
            "DPU_iterations": DPU_iterations,
            "DPU_times_per_iter": DPU_time_per_iter,
            "DPU_init_times": DPU_init_times,
            "DPU_data_times": DPU_transform_and_load_times,
            "DPU_scores": DPU_scores,
            "CPU_times": CPU_times,
            "CPU_iterations": CPU_iterations,
            "CPU_scores": CPU_scores,
            "CPU_times_per_iter": CPU_time_per_iter,
        },
        index=n_dpu_set[: i_n_dpu + 1],
    )
    df.index.rename("DPUs")
    df.to_pickle("results.pkl")

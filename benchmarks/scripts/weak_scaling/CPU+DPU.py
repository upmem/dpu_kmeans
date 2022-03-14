#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
import numpy as np
import sys
import time
import pandas as pd
from hurry.filesize import size
from tqdm import tqdm
from dpu_kmeans import DimmData, KMeans as DPU_KMeans, _dimm
from sklearn.cluster import KMeans

nfeatures = 8
n_clusters = 16
loops = 1
verbose = False
tol = 1e-4

n_points_per_dpu = int(1e5)
n_dim = 16
n_cluster = 16

n_dpu_set = [1, 2, 4, 8, 16, 32, 64]

DPU_times = []
DPU_inner_times = []
DPU_init_times = []
DPU_transform_and_load_times = []
DPU_iterations = []

CPU_times = []
CPU_iterations = []

for i_n_dpu, n_dpu in enumerate(tqdm(n_dpu_set, file=sys.stdout)):
    data, tags, centers = make_blobs(
        n_points_per_dpu * n_dpu,
        n_dim,
        centers=n_clusters,
        random_state=42,
        return_centers=True,
    )

    data = data.astype(np.float32)
    print("data size for {} dpus : {}".format(n_dpu, size(sys.getsizeof(data))))

    # load the DPUS
    _dimm.free_dpus()
    tic = time.perf_counter()
    _dimm.set_n_dpu(n_dpu)
    _dimm.load_kernel("kmeans", verbose)
    toc = time.perf_counter()
    DPU_init_time = toc - tic

    # load the data
    tic = time.perf_counter()
    dimm_data = DimmData(data)
    _dimm.load_data(dimm_data, tol, verbose)
    toc = time.perf_counter()
    DPU_data_transform_and_load_time = toc - tic

    # perform clustering on DPU
    tic = time.perf_counter()
    DPU_kmeans = DPU_KMeans(
        n_clusters, n_init=loops, max_iter=500, tol=tol, verbose=False
    )
    DPU_centroids, DPU_iter_counter, DPU_inner_timer = DPU_kmeans.fit(dimm_data)
    toc = time.perf_counter()
    DPU_timer = toc - tic

    # perform clustering on CPU
    init = data[:n_clusters]
    tic = time.perf_counter()
    CPU_kmeans = KMeans(
        n_cluster,
        n_init=loops,
        init=init,
        max_iter=500,
        tol=tol,
        verbose=False,
        copy_x=False,
    )
    CPU_kmeans.fit(data)
    CPU_centroids, CPU_iter_counter = CPU_kmeans.cluster_centers_, CPU_kmeans.n_iter_
    toc = time.perf_counter()
    CPU_timer = toc - tic

    DPU_times.append(DPU_timer)
    DPU_inner_times.append(DPU_inner_timer)
    DPU_iterations.append(DPU_iter_counter)
    DPU_init_times.append(DPU_init_time)
    DPU_transform_and_load_times.append(DPU_data_transform_and_load_time)

    CPU_times.append(CPU_timer)
    CPU_iterations.append(CPU_iter_counter)

    # comparing CPU and DPU results
    relative_distance = np.linalg.norm(DPU_centroids - CPU_centroids) / np.linalg.norm(
        CPU_centroids
    )

    df = pd.DataFrame(
        {
            "DPU_times": DPU_times,
            "DPU_inner_times": DPU_times,
            "DPU_iterations": DPU_iterations,
            "DPU_init_times": DPU_init_times,
            "DPU_data_times": DPU_transform_and_load_times,
            "CPU_times": CPU_times,
            "CPU_iterations": CPU_iterations,
            "relative_distance": relative_distance,
        },
        index=n_dpu_set[: i_n_dpu + 1],
    )
    df.index.rename("DPUs")
    df.to_pickle("results.pkl")

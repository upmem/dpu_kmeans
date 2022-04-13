#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import pandas as pd
from hurry.filesize import size
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from tqdm import tqdm

n_clusters = 16
n_init = 10
max_iter = 500
verbose = False
tol = 1e-4
random_state = 42

n_points_per_dpu = int(1e5)
n_dim = 16

n_dpu_set = [1, 2, 4, 8, 16, 32, 64]

CPU_times = []
CPU_main_loop_timers = []
CPU_iterations = []
CPU_scores = []
CPU_time_per_iter = []
CPU_preprocessing_times = []
CPU_main_loop_timers = []

for i_n_dpu, n_dpu in enumerate(pbar := tqdm(n_dpu_set, file=sys.stdout)):
    pbar.set_description(f"{n_dpu} dpus, raw data size : ___")

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

    pbar.set_description(f"{n_dpu} dpus, raw data size : {size(sys.getsizeof(data))}")

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
        # algorithm="full",
    )
    CPU_kmeans.fit(data)
    toc = time.perf_counter()

    # read timers and results
    CPU_centroids = CPU_kmeans.cluster_centers_
    CPU_iter_counter = CPU_kmeans.n_iter_
    CPU_main_loop_timer = CPU_kmeans.main_loop_timer_
    CPU_preprocessing_timer = CPU_kmeans.preprocessing_timer_

    CPU_timer = toc - tic

    ##################################################
    #                   LOGGING                      #
    ##################################################

    CPU_times.append(CPU_timer)
    CPU_iterations.append(CPU_iter_counter)
    CPU_time_per_iter.append(CPU_main_loop_timer / CPU_iter_counter)
    CPU_main_loop_timers.append(CPU_main_loop_timer)
    CPU_preprocessing_times.append(CPU_preprocessing_timer)

    # rand index for CPU and DPU (measures the similarity of the clustering with the ground truth)
    CPU_scores.append(calinski_harabasz_score(data, CPU_kmeans.labels_))

    # creating and exporting the dataframe at each iteration in case we crash early
    df = pd.DataFrame(
        {
            "CPU_times": CPU_times,
            "CPU_preprocessing_times": CPU_preprocessing_times,
            "CPU_single_kmeans_times": CPU_main_loop_timers,
            "CPU_iterations": CPU_iterations,
            "CPU_times_one_iter": CPU_time_per_iter,
            "CPU_scores": CPU_scores,
        },
        index=n_dpu_set[: i_n_dpu + 1],
    )
    df.index.rename("DPUs")
    df.to_pickle("only_CPU_elkan.pkl")
    df.to_csv("only_CPU_elkan.csv")

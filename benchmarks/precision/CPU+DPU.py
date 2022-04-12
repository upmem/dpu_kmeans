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

from dpu_kmeans import KMeans as DPU_KMeans
from dpu_kmeans import _dimm

n_clusters = 16
n_init = 10
max_iter = 500
verbose = False
tol = 1e-4
random_state_set = list(range(42))

n_points_per_dpu = int(1e5)
n_dim = 16

n_dpu = 1

DPU_times = []
DPU_kernel_runtimes = []
DPU_preprocessing_times = []
DPU_init_times = []
DPU_main_loop_timers = []
DPU_iterations = []
DPU_scores = []
DPU_time_per_iter = []
DPU_inter_pim_core_times = []
DPU_cpu_pim_times = []

CPU_times = []
CPU_main_loop_timers = []
CPU_iterations = []
CPU_scores = []
CPU_time_per_iter = []
CPU_preprocessing_times = []
CPU_main_loop_timers = []

cross_scores = []


for i_random_state, random_state in enumerate(
    pbar := tqdm(random_state_set, file=sys.stdout)
):
    pbar.set_description(f"roll : {random_state}, raw data size : ___")

    ##################################################
    #                   DATA GEN                     #
    ##################################################
    n_points = n_points_per_dpu * n_dpu

    data, tags, centers = make_blobs(
        n_points,
        n_dim,
        centers=n_clusters,
        random_state=random_state,
        return_centers=True,
    )

    pbar.set_description(
        f"roll : {random_state}, raw data size : {size(sys.getsizeof(data))}"
    )

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
        algorithm="full",
    )
    CPU_kmeans.fit(data)
    toc = time.perf_counter()

    # read timers
    CPU_centroids = CPU_kmeans.cluster_centers_
    CPU_iter_counter = CPU_kmeans.n_iter_
    CPU_main_loop_timer = CPU_kmeans.main_loop_timer_
    CPU_preprocessing_timer = CPU_kmeans.preprocessing_timer_

    CPU_timer = toc - tic

    ##################################################
    #                   DPU PERF                     #
    ##################################################

    pbar.set_description(f"roll : {random_state}, quantized size : ___")

    # load the DPUS
    _dimm.free_dpus()
    tic = time.perf_counter()
    _dimm.set_n_dpu(n_dpu)
    _dimm.load_kernel("kmeans", verbose)
    toc = time.perf_counter()
    DPU_init_time = toc - tic

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
        reload_data=True,
    )
    DPU_kmeans.fit(data)
    toc = time.perf_counter()

    # read timers
    DPU_centroids = DPU_kmeans.cluster_centers_
    DPU_iter_counter = DPU_kmeans.n_iter_
    DPU_kernel_runtime = DPU_kmeans.dpu_run_time_
    DPU_main_loop_timer = DPU_kmeans.main_loop_timer_
    DPU_preprocessing_timer = DPU_kmeans.preprocessing_timer_
    DPU_cpu_pim_timer = DPU_kmeans.cpu_pim_time_

    DPU_timer = toc - tic

    pbar.set_description(
        f"roll : {random_state}, quantized size : {size(_dimm._data_size)}"
    )

    ##################################################
    #                   LOGGING                      #
    ##################################################

    DPU_times.append(DPU_timer)
    DPU_kernel_runtimes.append(DPU_kernel_runtime)
    DPU_preprocessing_times.append(DPU_preprocessing_timer)
    DPU_iterations.append(DPU_iter_counter)
    DPU_time_per_iter.append(DPU_main_loop_timer / DPU_iter_counter)
    DPU_init_times.append(DPU_init_time)
    DPU_main_loop_timers.append(DPU_main_loop_timer)
    DPU_inter_pim_core_times.append(DPU_main_loop_timer - DPU_kernel_runtime)
    DPU_cpu_pim_times.append(DPU_cpu_pim_timer)

    CPU_times.append(CPU_timer)
    CPU_iterations.append(CPU_iter_counter)
    CPU_time_per_iter.append(CPU_main_loop_timer / CPU_iter_counter)
    CPU_main_loop_timers.append(CPU_main_loop_timer)
    CPU_preprocessing_times.append(CPU_preprocessing_timer)

    # rand index for CPU and DPU (measures the similarity of the clustering with the ground truth)
    DPU_scores.append(calinski_harabasz_score(data, DPU_kmeans.labels_))
    CPU_scores.append(calinski_harabasz_score(data, CPU_kmeans.labels_))
    cross_scores.append(adjusted_rand_score(CPU_kmeans.labels_, DPU_kmeans.labels_))

    # creating and exporting the dataframe at each iteration in case we crash early
    df = pd.DataFrame(
        {
            "DPU_times": DPU_times,
            "DPU_init_times": DPU_init_times,
            "DPU_preprocessing_times": DPU_preprocessing_times,
            "DPU_cpu_pim_times": DPU_cpu_pim_times,
            "DPU_single_kmeans_times": DPU_main_loop_timers,
            "DPU_kernel_runtimes": DPU_kernel_runtimes,
            "DPU_inter_pim_core_times": DPU_inter_pim_core_times,
            "DPU_iterations": DPU_iterations,
            "DPU_times_one_iter": DPU_time_per_iter,
            "CPU_times": CPU_times,
            "CPU_preprocessing_times": CPU_preprocessing_times,
            "CPU_single_kmeans_times": CPU_main_loop_timers,
            "CPU_iterations": CPU_iterations,
            "CPU_times_one_iter": CPU_time_per_iter,
            "DPU_scores": DPU_scores,
            "CPU_scores": CPU_scores,
            "cross_scores": cross_scores,
        },
        index=random_state_set[: i_random_state + 1],
    )
    df.index.rename("random_state", inplace=True)
    df.to_pickle(f"precision_{n_dpu}.pkl")
    df.to_csv(f"precision_{n_dpu}.csv")

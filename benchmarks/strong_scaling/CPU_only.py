#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
patch_dir = os.path.realpath(os.path.join(script_dir, "../dropin_patch/"))
sys.path.insert(0, patch_dir)

import pandas as pd

# from sklearn.cluster import KMeans
from _kmeans_w_perf import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

n_clusters = 16
n_init = 10
max_iter = 500
verbose = False
tol = 1e-4
random_state = 42

n_points = int(1e5) * 256
n_dim = 16

CPU_times = []
CPU_main_loop_timers = []
CPU_iterations = []
CPU_scores = []
CPU_time_per_iter = []
CPU_preprocessing_times = []
CPU_main_loop_timers = []
CPU_train_times = []

cross_scores = []

##################################################
#                   DATA GEN                     #
##################################################

data, tags, centers = make_blobs(
    n_points,
    n_dim,
    centers=n_clusters,
    random_state=random_state,
    return_centers=True,
)

print(f"raw data size : {sys.getsizeof(data)}")

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
CPU_train_timer = CPU_kmeans.train_time_

CPU_timer = toc - tic

CPU_scores.append(calinski_harabasz_score(data, CPU_kmeans.labels_))

# logging
CPU_times.append(CPU_timer)
CPU_iterations.append(CPU_iter_counter)
CPU_time_per_iter.append(CPU_main_loop_timer / CPU_iter_counter)
CPU_main_loop_timers.append(CPU_main_loop_timer)
CPU_preprocessing_times.append(CPU_preprocessing_timer)
CPU_train_times.append(CPU_train_timer)


df = pd.DataFrame(
    {
        "CPU_times": CPU_times,
        "CPU_train_times": CPU_train_times,
        "CPU_preprocessing_times": CPU_preprocessing_times,
        "CPU_single_kmeans_times": CPU_main_loop_timers,
        "CPU_iterations": CPU_iterations,
        "CPU_times_one_iter": CPU_time_per_iter,
        "CPU_scores": CPU_scores,
    },
)
df.index.rename("DPUs")
df.to_pickle("strong_scaling_CPU.pkl")
df.to_csv("strong_scaling_CPU.csv")

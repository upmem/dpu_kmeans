#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import cudf
import numpy as np
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
from hurry.filesize import size
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score

n_clusters = 16
n_init = 10
max_iter = 500
verbose = False
tol = 1e-4
random_state = 42

n_points = int(1e5) * 256
n_dim = 16

GPU_times = []
GPU_transfer_times = []
GPU_init_timers = []
GPU_iterations = []
GPU_scores = []

CPU_times = []
CPU_iterations = []
CPU_scores = []

cross_scores = []


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({"fea%d" % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


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

print(f"raw data size : {size(sys.getsizeof(data))}")

##################################################
#                   GPU PERF                     #
##################################################

# load empty data to the GPU to pay the initialization cost
tic = time.perf_counter()
gpu_data = np2cudf(np.zeros((2, 2)))
toc = time.perf_counter()
GPU_init_timer = toc - tic

# load the data to the GPU
tic = time.perf_counter()
gpu_data = np2cudf(data)
toc = time.perf_counter()
GPU_transfer_timer = toc - tic

# perform clustering on DPU
tic = time.perf_counter()
GPU_kmeans = cuKMeans(
    n_clusters=n_clusters,
    init="random",
    n_init=n_init,
    max_iter=max_iter,
    tol=tol,
    verbose=verbose,
    random_state=random_state,
)
GPU_kmeans.fit(data)
toc = time.perf_counter()

# read output
(GPU_centroids, GPU_iter_counter,) = (
    GPU_kmeans.cluster_centers_,
    GPU_kmeans.n_iter_,
)
GPU_timer = toc - tic

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

# read output
(CPU_centroids, CPU_iter_counter,) = (
    CPU_kmeans.cluster_centers_,
    CPU_kmeans.n_iter_,
)
CPU_timer = toc - tic

##################################################
#                   LOGGING                      #
##################################################

GPU_times.append(GPU_timer)
GPU_transfer_times.append(GPU_transfer_timer)
GPU_init_timers.append(GPU_init_timer)
GPU_iterations.append(GPU_iter_counter)

CPU_times.append(CPU_timer)
CPU_iterations.append(CPU_iter_counter)

# rand index for CPU and DPU (measures the similarity of the clustering with the ground truth)
GPU_scores.append(calinski_harabasz_score(data, GPU_kmeans.labels_))
CPU_scores.append(calinski_harabasz_score(data, CPU_kmeans.labels_))
cross_scores.append(adjusted_rand_score(CPU_kmeans.labels_, GPU_kmeans.labels_))

df = pd.DataFrame(
    {
        "GPU_times": GPU_times,
        "GPU_init_time": GPU_init_timers,
        "GPU_transfer_times": GPU_transfer_times,
        "GPU_iterations": GPU_iterations,
        "CPU_times": CPU_times,
        "CPU_iterations": CPU_iterations,
        "GPU_scores": GPU_scores,
        "CPU_scores": CPU_scores,
        "cross_scores": cross_scores,
    },
)
df.to_pickle("strong_scaling_GPU_results.pkl")
df.to_csv("strong_scaling_GPU_results.csv")

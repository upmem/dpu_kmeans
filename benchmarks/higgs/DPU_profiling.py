#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import pandas as pd
from hurry.filesize import size
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from tqdm import tqdm

from dpu_kmeans import KMeans as DPU_KMeans
from dpu_kmeans import _dimm

n_clusters = 2
n_init = 10
max_iter = 500
verbose = False
tol = 1e-4
random_state = 42

n_dpu_set = [1024]  # [256, 512, 1024, 2048, 2524]

##################################################
#                   DATA READ                    #
##################################################

if len(sys.argv) >= 2:
    higgs_file = sys.argv[1]
else:
    higgs_file = "data/higgs.pq"
df = pd.read_parquet(higgs_file)

data, tags = np.require(
    df.iloc[:, 1:].to_numpy(dtype=np.float32), requirements=["C", "A", "O"]
), np.require(df.iloc[:, 0].to_numpy(dtype=int), requirements=["O"])

n_points, n_dim = data.shape

del df

print(f"raw data size : {size(sys.getsizeof(data))}")

for i_n_dpu, n_dpu in enumerate(pbar := tqdm(n_dpu_set, file=sys.stdout)):
    pbar.set_description(f"{n_dpu} dpus, raw data size : {size(sys.getsizeof(data))}")

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

    # perform clustering on DPU
    tic = time.perf_counter()
    DPU_kmeans = DPU_KMeans(
        n_clusters,
        reload_data=True,
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

    # read timers
    DPU_centroids = DPU_kmeans.cluster_centers_
    DPU_iter_counter = DPU_kmeans.n_iter_
    DPU_kernel_runtime = DPU_kmeans.dpu_run_time_
    DPU_main_loop_timer = DPU_kmeans.main_loop_timer_
    DPU_preprocessing_timer = DPU_kmeans.preprocessing_timer_
    DPU_cpu_pim_timer = DPU_kmeans.cpu_pim_time_

    DPU_timer = toc - tic

    pbar.set_description(f"{n_dpu} dpus, quantized size : {size(_dimm._data_size)}")

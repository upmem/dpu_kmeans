#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cProfile
import sys

from hurry.filesize import size
from sklearn.datasets import make_blobs

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

n_dpu = 64

data, tags, centers = make_blobs(
    n_points_per_dpu * n_dpu,
    n_dim,
    centers=n_clusters,
    random_state=random_state,
    return_centers=True,
)

# data = data.astype(np.float32)
print(f"raw data size for {n_dpu} dpus : {size(sys.getsizeof(data))}")

# load the DPUS
_dimm.set_n_dpu(n_dpu)
_dimm.load_kernel("kmeans", verbose)

# load the data in the DPUs
_dimm.load_data(data, verbose)
print(f"quantized data size for {n_dpu} dpus : {size(_dimm._data_size)}")

# perform clustering on DPU
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
cProfile.run("DPU_kmeans.fit(data)", "restats.prof")

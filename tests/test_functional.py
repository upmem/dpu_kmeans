#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for automated functional testing
Tests the K-Means at the service level"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from dpu_kmeans import KMeans as DPUKMeans
from dpu_kmeans import _dimm

N_CLUSTERS = 15


def test_clustering_dpu_then_cpu():
    """Make clustering on DPUs and then on CPU, and compare the results"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )

    init = data[:N_CLUSTERS]

    # Clustering with DPUs
    _dimm.set_n_dpu(4)
    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)

    # Clustering with CPU
    kmeans = KMeans(N_CLUSTERS, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    assert rand_score > 1 - 1e-2
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_cpu_then_dpu():
    """Make clustering on CPUs and then on DPU, and compare the results"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )

    init = data[:N_CLUSTERS]

    # Clustering with CPU
    kmeans = KMeans(N_CLUSTERS, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Clustering with DPUs
    _dimm.set_n_dpu(4)
    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    assert rand_score > 1 - 1e-2
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_dpu_then_dpu():
    """Make clustering on DPU twice, and compare the results"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )
    data_copy = data.copy()

    init = data[:N_CLUSTERS]

    # Clustering with DPUs
    _dimm.set_n_dpu(4)

    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)
    n_iter_1 = dpu_kmeans.n_iter_
    dpu_labels_1 = dpu_kmeans.labels_

    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data_copy)
    n_iter_2 = dpu_kmeans.n_iter_
    dpu_labels_2 = dpu_kmeans.labels_

    # Comparison
    rand_score = adjusted_rand_score(dpu_labels_1, dpu_labels_2)

    assert rand_score > 1 - 1e-4
    assert n_iter_1 == n_iter_2


if __name__ == "__main__":
    test_clustering_dpu_then_cpu()
    test_clustering_cpu_then_dpu()
    test_clustering_dpu_then_dpu()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for automated functional testing"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from dpu_kmeans import KMeans as DPUKMeans, _dimm

N_CLUSTERS = 15


def test_clustering_dpu_then_cpu():
    """Tests the K-Means at the service level"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )

    # Clustering with DPUs
    _dimm.set_n_dpu(4)

    init = data[:N_CLUSTERS]

    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)
    dpu_centroids = dpu_kmeans.cluster_centers_

    # Clustering with CPU
    kmeans = KMeans(N_CLUSTERS, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    # Comparison
    relative_distance = np.linalg.norm(dpu_centroids - centroids) / np.linalg.norm(
        centroids
    )
    assert relative_distance < 1e-3
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_cpu_then_dpu():
    """Tests the K-Means at the service level"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )

    # Clustering with CPU
    init = data[:N_CLUSTERS]
    kmeans = KMeans(N_CLUSTERS, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    # Clustering with DPUs
    _dimm.set_n_dpu(4)

    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)
    dpu_centroids = dpu_kmeans.cluster_centers_

    # Comparison
    relative_distance = np.linalg.norm(dpu_centroids - centroids) / np.linalg.norm(
        centroids
    )
    assert relative_distance < 1e-3
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_dpu_then_dpu():
    """Tests the K-Means at the service level"""

    # Generating data
    data = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)[0].astype(
        np.float32
    )
    data_copy = data.copy()

    # Clustering with DPUs
    _dimm.set_n_dpu(4)

    init = data[:N_CLUSTERS]
    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data)
    n_iter_1 = dpu_kmeans.n_iter_
    dpu_centroids_1 = dpu_kmeans.cluster_centers_

    dpu_kmeans = DPUKMeans(N_CLUSTERS, init=init, n_init=1, verbose=False)
    dpu_kmeans.fit(data_copy)
    dpu_centroids_2 = dpu_kmeans.cluster_centers_
    n_iter_2 = dpu_kmeans.n_iter_

    # Comparison
    relative_distance = np.linalg.norm(
        dpu_centroids_1 - dpu_centroids_2
    ) / np.linalg.norm(dpu_centroids_1)
    assert relative_distance < 1e-3
    assert n_iter_1 == n_iter_2


if __name__ == "__main__":
    test_clustering_dpu_then_cpu()
    test_clustering_cpu_then_dpu()
    test_clustering_dpu_then_dpu()

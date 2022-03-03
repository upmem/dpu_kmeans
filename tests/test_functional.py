#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cgi import test
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from dpu_kmeans import DimmData, KMeans as DPUKMeans, _dimm

N_CLUSTERS = 15


def test_clustering():
    """Tests the K-Means at the service level"""

    _dimm.set_n_dpu(4)
    # _dimm.load_kernel("kmeans")

    data, _ = make_blobs(int(1e4), 8, centers=N_CLUSTERS, random_state=42)

    data = data.astype(np.float32)

    dimm_data = DimmData(data)

    # _dimm.load_data(dimm_data)

    dpu_kmeans = DPUKMeans(N_CLUSTERS, n_init=1, verbose=False)
    centroids, _, _ = dpu_kmeans.fit(dimm_data)

    del dimm_data

    init = data[:N_CLUSTERS]
    kmeans = KMeans(N_CLUSTERS, init=init, n_init=1, algorithm="full")

    kmeans.fit(data)

    relative_distance = np.linalg.norm(
        centroids - kmeans.cluster_centers_
    ) / np.linalg.norm(kmeans.cluster_centers_)
    assert relative_distance < 1e-3


if __name__ == "__main__":
    test_clustering()

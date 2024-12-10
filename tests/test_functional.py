"""Module for automated functional testing.

Tests the K-Means at the service level

:Author: Sylvan Brocard <sbrocard@upmem.com>
:License: MIT
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from dpu_kmeans import KMeans as DPUKMeans


def test_clustering_dpu_then_cpu():
    """Make clustering on DPUs and then on CPU, and compare the results."""
    n_clusters = 15
    n_features = 8
    n_points = int(1e4)

    # Generating data
    data = make_blobs(n_points, n_features, centers=n_clusters, random_state=42)[
        0
    ].astype(
        np.float32,
    )

    rng = np.random.default_rng(42)
    init = rng.choice(data, n_clusters, replace=False)

    # Clustering with DPUs
    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data)

    # Clustering with CPU
    kmeans = KMeans(n_clusters, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    assert rand_score > 1 - 1e-2
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_cpu_then_dpu():
    """Make clustering on CPUs and then on DPU, and compare the results."""
    n_clusters = 15
    n_features = 8
    n_points = int(1e4)

    # Generating data
    data = make_blobs(n_points, n_features, centers=n_clusters, random_state=42)[
        0
    ].astype(
        np.float32,
    )

    rng = np.random.default_rng(42)
    init = rng.choice(data, n_clusters, replace=False)

    # Clustering with CPU
    kmeans = KMeans(n_clusters, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Clustering with DPUs
    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    assert rand_score > 1 - 1e-2
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_clustering_dpu_then_dpu():
    """Make clustering on DPU twice, and compare the results."""
    n_clusters = 15
    n_features = 8
    n_points = int(1e4)

    # Generating data
    data = make_blobs(n_points, n_features, centers=n_clusters, random_state=42)[
        0
    ].astype(
        np.float32,
    )
    data_copy = data.copy()

    rng = np.random.default_rng(42)
    init = rng.choice(data, n_clusters, replace=False)

    # Clustering with DPUs
    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data)
    n_iter_1 = dpu_kmeans.n_iter_
    dpu_labels_1 = dpu_kmeans.labels_

    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data_copy)
    n_iter_2 = dpu_kmeans.n_iter_
    dpu_labels_2 = dpu_kmeans.labels_

    # Comparison
    rand_score = adjusted_rand_score(dpu_labels_1, dpu_labels_2)

    assert rand_score > 1 - 1e-4
    assert n_iter_1 == n_iter_2


def test_large_dimensionality():
    """Test the clustering with a large features * clusters product."""
    n_clusters = 24
    n_features = 128
    n_points = int(1e4)

    # Generating data
    data = make_blobs(n_points, n_features, centers=n_clusters, random_state=42)[
        0
    ].astype(np.float32)

    rng = np.random.default_rng(42)
    init = rng.choice(data, n_clusters, replace=False)

    # Clustering with DPUs
    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data)

    # Clustering with CPU
    kmeans = KMeans(n_clusters, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    assert rand_score > 1 - 1e-2
    assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


def test_relocation():
    """Test the clustering with a bad initialization.

    The initialization is such that the clusters are not well separated.
    This will cause at least one relocation.
    """
    n_clusters = 15
    n_features = 8
    n_points = int(1e4)

    # Generating data
    data = make_blobs(n_points, n_features, centers=n_clusters, random_state=42)[
        0
    ].astype(np.float32)

    init = np.tile(data[0], (n_clusters, 1))

    # Clustering with DPUs
    dpu_kmeans = DPUKMeans(n_clusters, init=init, n_init=1, verbose=False, n_dpu=4)
    dpu_kmeans.fit(data)

    # Clustering with CPU
    kmeans = KMeans(n_clusters, init=init, n_init=1, algorithm="full")
    kmeans.fit(data)

    # Comparison
    rand_score = adjusted_rand_score(dpu_kmeans.labels_, kmeans.labels_)

    # assert rand_score > 1 - 1e-2
    # assert kmeans.n_iter_ * 2 / 3 < dpu_kmeans.n_iter_ < kmeans.n_iter_ * 1.5


if __name__ == "__main__":
    test_clustering_dpu_then_cpu()
    test_clustering_cpu_then_dpu()
    test_clustering_dpu_then_dpu()
    test_large_dimensionality()
    test_relocation()

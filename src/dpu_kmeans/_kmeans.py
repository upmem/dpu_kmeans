# -*- coding: utf-8 -*-
"""K-means clustering on DPU"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans as KMeansCPU
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.cluster._kmeans import _is_same_clustering
from sklearn.exceptions import ConvergenceWarning

from . import _dimm
from ._dimm import DimmData, load_data


def _kmeans_single_lloyd_dpu():
    pass


class KMeans(KMeansCPU):
    """KMeans estimator object

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    Examples
    --------

    >>> import numpy as np
    >>> from dpu_kmeans import DIMM_data, KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
    >>> dimm_data = DIMM_data(X)
    >>> kmeans = KMeans(2)
    >>> centroids, iterations, time = kmeans.fit(dimm_data)
    >>> print(centroids)
    [[ 0.9998627  2.       ]
    [10.000137   2.       ]]
    """

    def __init__(self, n_clusters: int = 8, *, n_dpu: int = 0, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.n_dpu = n_dpu
        self.n_iter_ = None
        self.time = None
        self.cluster_centers_ = None

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # transfer the data points to the DPUs
        load_data(X, verbose=self.verbose)

        kmeans_single = _kmeans_single_lloyd_dpu
        self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                x_squared_norms=x_squared_norms,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def _fit(self, X: DimmData):
        """Compute k-means clustering.

        Parameters
        ----------
        X : DIMM_data
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        Returns
        -------
        result : ndarray
            The centroids found by the clustering algorithm.
        iterations : int
            Number of iterations performed during the best run.
        time : float
            Total clustering time.
        """
        if self.n_dpu:
            _dimm.set_n_dpu(self.n_dpu)
        _dimm.load_kernel("kmeans", self.verbose)
        _dimm.load_data(X, self.tol, self.verbose)
        result, iterations, time = self._kmeans()

        result += X.feature_means

        self.n_iter_ = iterations
        self.time = _dimm.ctr.dpu_run_time()
        self.cluster_centers_ = result

        return result, iterations, time

    def fit_predict(self, X, y=None):
        """TODO: return clusterization labels"""

    def _kmeans(self):
        log_iterations = np.require(
            np.zeros(1, dtype=np.int32), requirements=["A", "C"]
        )
        log_time = np.require(np.zeros(1, dtype=np.float64), requirements=["A", "C"])

        clusters = _dimm.ctr.kmeans(
            self.n_clusters,
            self.n_clusters,
            False,
            self.verbose,
            self.n_init,
            self.max_iter,
            log_iterations,
            log_time,
        )

        return clusters, log_iterations[0], log_time[0]

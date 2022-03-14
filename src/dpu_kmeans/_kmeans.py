# -*- coding: utf-8 -*-
"""K-means clustering on DPU"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans as KMeansCPU
from sklearn.cluster._kmeans import (
    _is_same_clustering,
    _labels_inertia_threadpool_limit,
    _openmp_effective_n_threads,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.validation import _check_sample_weight

from . import _dimm


def _lloyd_iter_dpu(
    centers_old,
    centers_old_int,
    centers_new,
    centers_new_int,
    points_in_clusters,
):
    """Single iteration of K-means lloyd algorithm with dense input on DPU.

    Update centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    Returns
    -------
    center_shift_tot : float
        Distance between old and new centers.
    """
    dpu_iter = _dimm.ctr.lloyd_iter

    centers_old_int[:] = _dimm.ld.transform(centers_old)

    dpu_iter(
        centers_old_int,
        centers_new_int,
        points_in_clusters,
    )

    centers_new[:, :] = _dimm.ld.inverse_transform(centers_new_int)
    np.divide(
        centers_new,
        points_in_clusters[:, None],
        out=centers_new,
        where=points_in_clusters[:, None] != 0,
    )
    # centers_new /= points_in_clusters[:, None]

    center_shift_tot = np.linalg.norm(centers_new - centers_old, ord="fro") ** 2
    return center_shift_tot


def _kmeans_single_lloyd_dpu(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd on DPU, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : ndarray of shape (n_samples,), default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]
    dtype = _dimm.ld.dtype

    # transfer the number of clusters to the DPUs
    _dimm.ctr.load_n_clusters(n_clusters)

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_int = np.empty_like(centers, dtype=dtype)
    centers_new = np.empty_like(centers, dtype=float)
    centers_new_int = np.empty_like(centers, dtype=np.int64)
    points_in_clusters = np.empty(n_clusters, dtype=np.int32)

    # points_in_clusters_per_dpu = np.empty((n_dpu, n_clusters_round), dtype=np.int32)
    # partial_sums = np.empty((n_clusters, n_dpu, n_features), dtype=np.int64)

    if sp.issparse(X):
        raise ValueError("Sparse matrix not supported")
    else:
        lloyd_iter = _lloyd_iter_dpu

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            center_shift_tot = lloyd_iter(
                centers,
                centers_int,
                centers_new,
                centers_new_int,
                points_in_clusters,
            )

            if verbose:
                _, inertia = _labels_inertia_threadpool_limit(
                    X, sample_weight, x_squared_norms, centers, n_threads
                )
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            # Check for tol based convergence.
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

    labels, inertia = _labels_inertia_threadpool_limit(
        X, sample_weight, x_squared_norms, centers, n_threads
    )

    _dimm.ctr.deallocate_host_memory()
    return labels, inertia, centers, i + 1


class KMeans(KMeansCPU):
    """KMeans estimator object

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

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

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

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
        self.dpu_run_time_ = None
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

        # allocate DPUs if not yet done
        if self.n_dpu:
            _dimm.set_n_dpu(self.n_dpu)

        # load kmeans kernel if not yet done
        _dimm.load_kernel("kmeans", self.verbose)

        # transfer the data points to the DPUs
        _dimm.load_data(X, verbose=self.verbose)

        # reset perf timer
        _dimm.reset_timer(verbose=self.verbose)

        kmeans_single = _kmeans_single_lloyd_dpu
        self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for _ in range(self._n_init):
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
                f"Number of distinct clusters ({distinct_clusters}) found smaller than "
                "n_clusters ({self.n_clusters}). Possibly due to duplicate points "
                "in X.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.dpu_run_time_ = _dimm.get_dpu_run_time()
        return self

    # def _kmeans(self):
    #     log_iterations = np.require(
    #         np.zeros(1, dtype=np.int32), requirements=["A", "C"]
    #     )
    #     log_time = np.require(np.zeros(1, dtype=np.float64), requirements=["A", "C"])

    #     clusters = _dimm.ctr.kmeans(
    #         self.n_clusters,
    #         self.n_clusters,
    #         False,
    #         self.verbose,
    #         self.n_init,
    #         self.max_iter,
    #         log_iterations,
    #         log_time,
    #     )

    #     return clusters, log_iterations[0], log_time[0]

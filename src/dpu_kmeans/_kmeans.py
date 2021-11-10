# -*- coding: utf-8 -*-
"""K-means clustering on DPU"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator

from . import _dimm
from ._dimm import DIMM_data


class KMeans(BaseEstimator):
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

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        _dimm.load_kernel("kmeans", self.verbose)

    def fit(self, X: DIMM_data):
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
        _dimm.load_kernel("kmeans", self.verbose)
        _dimm.load_data(X, self.tol, self.verbose)
        result, iterations, time = self._kmeans()

        return result, iterations, time

    def fit_predict(self, X):
        pass

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

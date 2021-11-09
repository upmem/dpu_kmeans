# -*- coding: utf-8 -*-
"""K-means clustering on DPU"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator

from . import _dimm
from ._dimm import DIMM_data


class KMeans(BaseEstimator):
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

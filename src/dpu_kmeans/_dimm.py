"""DIMM memory manager module.

This module is intended to work like a singleton class,
hence the use of global variables.

:Author: Sylvan Brocard <sbrocard@upmem.com>
:License: MIT
"""

# pylint: disable=global-statement

import numpy as np
import xxhash

try:
    from importlib.resources import as_file, files
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import as_file, files

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._core import FEATURE_TYPE, Container

_kernels_lib = {"kmeans": files("dpu_kmeans").joinpath("dpu_program/kmeans_dpu_kernel")}

ctr = Container()


class LinearDiscretizer(TransformerMixin, BaseEstimator):
    """Transformer to quantize data for DIMMs."""

    def __init__(self) -> None:
        if FEATURE_TYPE == 8:
            self.dtype = np.int8
        elif FEATURE_TYPE == 16:
            self.dtype = np.int16
        elif FEATURE_TYPE == 32:
            self.dtype = np.int32
        self.input_dtype = None

    def fit(self, X, y=None):
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X = self._validate_data(X, dtype="numeric")
        self.input_dtype = X.dtype

        # Compute scale factor for quantization
        max_feature = np.max(np.abs(X))
        self.scale_factor = (
            np.iinfo(self.dtype).max / max_feature / 2.1
        )  # small safety margin to avoid rounding overflow

        return self

    def transform(self, X):
        """Discretize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be quantized.

        Returns
        -------
        Xt : ndarray, dtype={np.int8, np.int16, np.int32}
            Quantized data.

        """
        check_is_fitted(self)

        Xt = np.empty_like(X, dtype=self.dtype)
        np.multiply(X, self.scale_factor, out=Xt, order="C", casting="unsafe")
        return Xt

    def inverse_transform(self, Xt):
        """Transform discretized data back to original feature space.

        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.

        Returns
        -------
        Xinv : ndarray, dtype={np.float32, np.float64}
            Data in the original feature space.

        """
        check_is_fitted(self)

        # adding 0.5 to compensate for rounding previously
        return ((Xt + np.sign(Xt) * 0.5) / self.scale_factor).astype(self.input_dtype)


ld = LinearDiscretizer()  # linear discretization transformer


def load_kernel(kernel: str, verbose: int = False):
    """Load a given kernel into the allocated DPUs."""
    if ctr.binary_path != kernel:
        if verbose:
            print(f"loading new kernel : {kernel}")
        with as_file(_kernels_lib[kernel]) as dpu_binary:
            ctr.load_kernel(dpu_binary)


def load_data(X, verbose: int = False):
    """Load a dataset into the allocated DPUs."""
    # compute the checksum of X
    h = xxhash.xxh3_64()
    h.update(X)
    X_checksum = h.digest()

    if ctr.hash != X_checksum:
        if verbose:
            print("loading new data")
        Xt = ld.fit_transform(X)
        ctr.load_array_data(Xt, X_checksum)
    elif verbose:
        print("reusing previously loaded data")

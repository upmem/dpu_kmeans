# -*- coding: utf-8 -*-
"""DIMM memory manager module
This module is intended to work like a singleton class, hence the use of global variables."""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

# pylint: disable=global-statement

import atexit
import sys
import time

import numpy as np
import xxhash

try:
    from importlib.resources import as_file, files
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._core import FEATURE_TYPE, Container

_allocated = False  # whether the DPUs have been allocated
_kernel = None  # name of the currently loaded binary
_data_id = None  # ID of the currently loaded data
_data_checksum = None  # the checksum of the currently loaded data
_data_size = None  # size of the currently loaded data

_kernels_lib = {"kmeans": files("dpu_kmeans").joinpath("dpu_program/kmeans_dpu_kernel")}

ctr = Container()
ctr.set_nr_dpus(0)

_requested_dpus = 0


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
        """
        Fit the estimator.

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
        """
        Discretize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be quantized.
        Returns
        -------
        Xt : ndarray, dtype={np.float32, np.float64}
            Quantized data.
        """
        check_is_fitted(self)

        Xt = np.empty_like(X, dtype=self.dtype)
        np.multiply(X, self.scale_factor, out=Xt, order="C", casting="unsafe")
        return Xt

    def inverse_transform(self, Xt):
        """
        Transform discretized data back to original feature space.

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
        return ((Xt + 0.5) / self.scale_factor).astype(self.input_dtype)


ld = LinearDiscretizer()  # linear discretization transformer


def set_n_dpu(n_dpu: int):
    """Sets the number of DPUs to ask for during the allocation."""
    global _allocated
    global _requested_dpus
    if _allocated and _requested_dpus != n_dpu:
        free_dpus()
    if not _allocated:
        _requested_dpus = n_dpu
        ctr.set_nr_dpus(n_dpu)
        ctr.allocate()
        _allocated = True


def load_kernel(kernel: str, verbose: int = False):
    """Loads a given kernel into the allocated DPUs."""
    global _kernel
    global _allocated
    global _data_id
    global _data_checksum
    global _data_size
    if not _allocated:
        ctr.allocate()
        _allocated = True
    if not _kernel == kernel:
        if verbose:
            print(f"loading new kernel : {kernel}")
        _kernel = kernel
        ref = _kernels_lib[kernel]
        with as_file(ref) as dpu_binary:
            ctr.load_kernel(str(dpu_binary))
        _data_id = None
        _data_checksum = None
        _data_size = None


def load_data(X, verbose: int = False):
    """Loads a dataset into the allocated DPUs."""
    global _data_checksum
    global _data_size

    # compute the checksum of X
    h = xxhash.xxh3_64()
    h.update(X)
    X_checksum = h.digest()

    if _data_checksum != X_checksum:
        if verbose:
            print("loading new data")
        if _data_checksum:
            if verbose:
                print(f"freeing previous data : {_data_checksum}")
            ctr.free_data()
        _data_checksum = X_checksum
        Xt = ld.fit_transform(X)
        ctr.load_array_data(
            Xt,
            Xt.shape[0],
            Xt.shape[1],
            verbose,
        )
        _data_size = sys.getsizeof(Xt)
    elif verbose:
        print("reusing previously loaded data")


def reset_timer(verbose=False):
    """Resets the DPU execution timer."""
    if verbose:
        print("resetting inner timer")
    ctr.reset_timer()


def get_dpu_run_time():
    """Returns the DPU execution timer."""
    return ctr.get_dpu_run_time()


def get_cpu_pim_time():
    """Returns the time to load the data to the DPU memory."""
    return ctr.get_cpu_pim_time()


def get_pim_cpu_time():
    """Returns the time to get the inertia from the DPU memory."""
    return ctr.get_pim_cpu_time()


def free_dpus(verbose: int = False):
    """Frees all allocated DPUs."""
    global _allocated
    global _kernel
    global _data_id
    global _data_checksum
    global _data_size
    if _allocated:
        if verbose:
            print("freeing dpus")
        ctr.free_dpus()
        _allocated = False
        _kernel = None
        _data_id = None
        _data_checksum = None
        _data_size = None


atexit.register(free_dpus)

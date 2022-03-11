# -*- coding: utf-8 -*-
"""DIMM memory manager module
This module is intended to work like a singleton class, hence the use of global variables."""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

# pylint: disable=global-statement

import atexit
import numpy as np
import xxhash

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._core import Container
from ._core import FEATURE_TYPE

_allocated = False  # whether the DPUs have been allocated
_kernel = None  # name of the currently loaded binary
_data_id = None  # ID of the currently loaded data
_data_checksum = None  # the checksum of the currently loaded data

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

        # Compute scale factor for quantization
        max_feature = np.max(np.abs(X))
        self.scale_factor = np.iinfo(self.dtype).max / max_feature / 2

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
        return np.rint(
            X * self.scale_factor,
            order="C",
            out=Xt,
            casting="unsafe",
        )

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

        return Xt / self.scale_factor


ld = LinearDiscretizer()  # linear discretization transformer

'''
class DimmData:
    """Holder object for data loaded on the DIMM

    Parameters
    ----------
    data : str or ArrayLike
        The path to the data file, or a numeric iterable containing the data.
        For best performance, provide a contiguous float32 numpy array.

    is_binary_file : bool
        True if the data is in binary format, False otherwise.
        Unused if type is "array".

    Atrributes
    ----------
    data_id : str or int
        Path to the data file, or ID of the underlying data array.

    npoints : int
        Number of points in the data set.

    nfeatures : int
        Number of features in the data set.

    type : str
        Form of the data ("file" or "array").

    X : numpy.ndarray[np.float32]
        Data as a numpy array usable by the compiled library.
    """

    def __init__(self, data, is_binary_file=False):
        self.data_id = None
        self.npoints, self.nfeatures = None, None
        self.type = ""
        self.is_binary_file = is_binary_file
        self._X_int = None
        self.scale_factor = None
        self.feature_means = None
        self.avg_variance = None
        self.X = data

    @property
    def X(self):
        """Get X"""
        return self._X

    @property
    def X_int(self):
        """Get X_int"""
        return self._X_int

    @X.setter
    def X(self, X):
        """Set X"""
        if isinstance(X, str):
            self._data_id = X
            self.type = "file"
        else:
            if not (isinstance(X, np.ndarray)) or not (
                X.flags.c_contiguous and X.flags.aligned and X.dtype == np.float32
            ):
                print(
                    "Converting input data. "
                    + "Provide a contiguous float32 ndarray to avoid this extra step."
                )
                self._X = np.require(X, dtype=np.float32, requirements=["A", "C"])
            else:
                self._X = X
            self.data_id = id(self._X)
            self.npoints, self.nfeatures = self.X.shape
            self.type = "array"

            self.feature_means = np.mean(self._X, axis=0)

            if FEATURE_TYPE == 8:
                quantized_data_type = np.int8
            elif FEATURE_TYPE == 16:
                quantized_data_type = np.int16
            elif FEATURE_TYPE == 32:
                quantized_data_type = np.int32

            # Compute scale factor for quantization
            max_feature = np.max(np.abs(self._X))
            self.scale_factor = np.iinfo(quantized_data_type).max / max_feature / 2

            # Compute average of variance
            variances = np.var(self._X, axis=0)
            self.avg_variance = np.mean(variances)

            self._X_int = np.zeros_like(self._X, dtype=quantized_data_type)
            np.rint(
                (self._X - self.feature_means) * self.scale_factor,
                order="C",
                out=self._X_int,
                casting="unsafe",
            )

    def __del__(self):
        global _data_id
        if self.data_id == _data_id:
            _data_id = None
            ctr.free_data(self.type == "file", False)
'''


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


def load_data(X, verbose: int = False):
    """Loads a dataset into the allocated DPUs."""
    global _data_checksum

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
    elif verbose:
        print("reusing previously loaded data")


def free_dpus(verbose: int = False):
    """Frees all allocated DPUs."""
    global _allocated
    global _kernel
    global _data_id
    global _data_checksum
    if _allocated:
        if verbose:
            print("freeing dpus")
        ctr.free_dpus()
        _allocated = False
        _kernel = None
        _data_id = None
        _data_checksum = None


atexit.register(free_dpus)

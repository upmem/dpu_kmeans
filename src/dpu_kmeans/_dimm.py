# -*- coding: utf-8 -*-
"""DIMM memory manager module"""

# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

from collections.abc import Iterable
import numpy as np
import atexit

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from ._core import Container

_kernel = ""  # name of the currently loaded binary

_data_id = None  # ID of the currently loaded data

_kernels_lib = {"kmeans": files("dpu_kmeans").joinpath("dpu_program/kmeans_dpu_kernel")}

ctr = Container()


class DIMM_data:
    """Holder object for data loaded on the DIMM

    Parameters
    ----------
    data : str or ArrayLike
        The path to the data file, or a numeric iterable containing the data.
        For best performance, provide a contiguous float32 numpy array.

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

    is_binary_file : bool
        True if the data is in binary format, False otherwise.
        Unused if type is "array".
    """

    def __init__(self, data, is_binary_file=False):
        self.data_id = None
        self.npoints, self.nfeatures = None, None
        self.type = ""
        self.X = data
        self.is_binary_file = is_binary_file

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        if isinstance(X, str):
            self._data_id = X
            self.type = "file"
        else:
            if not (isinstance(X, np.ndarray)) or not (
                X.flags.c_contiguous and X.flags.aligned and X.dtype == np.float32
            ):
                print(
                    "Converting input data. Provide a contiguous float32 ndarray to avoid this extra step."
                )
                self._X = np.require(X, dtype=np.float32, requirements=["A", "C"])
            else:
                self._X = X
            self.data_id = id(self._X)
            self.npoints, self.nfeatures = self.X.shape
            self.type = "array"

    def __del__(self):
        global ctr
        global _data_id
        if self.data_id == _data_id:
            _data_id = None
            ctr.free_data(self.type == "file", False)


def load_kernel(kernel: str, verbose: int):
    global ctr
    global _kernel
    if not _kernel == kernel:
        if verbose:
            print(f"loading new kernel : {kernel}")
        _kernel = kernel
        ref = _kernels_lib[kernel]
        with as_file(ref) as DPU_BINARY:
            ctr.load_kernel(str(DPU_BINARY))


def load_data(data: DIMM_data, tol: float, verbose: int):
    global ctr
    global _data_id
    if _data_id != data.data_id:
        if verbose:
            print("loading new data")
        if _data_id:
            if verbose:
                print(f"freeing previous data : {_data_id}")
            ctr.free_data(data.type == "file", True)
        _data_id = data.data_id
        if data.type == "file":
            if verbose:
                print(f"reading data from file {data.data_id}")
            ctr.load_file_data(data.data_id, data.is_binary_file, tol, verbose)
        elif data.type == "array":
            if verbose:
                print("reading data from array")
            ctr.load_array_data(data.X, data.npoints, data.nfeatures, tol, verbose)


def free_dpus():
    print("freeing dpus")
    global ctr
    ctr.free_dpus()


atexit.register(free_dpus)

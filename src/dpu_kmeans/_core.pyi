"""

        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans._core

        .. autosummary::
           :toctree: _generate

           Container
    
"""
from __future__ import annotations
import numpy
import os
import typing
__all__ = ['Container', 'FEATURE_TYPE']
class Container:
    """
    
            Container object to interface with the DPUs
        
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def allocate(self, arg0: int) -> None:
        ...
    def free_dpus(self) -> None:
        ...
    def lloyd_iter(self, arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.int64], arg2: numpy.ndarray[numpy.int32]) -> None:
        ...
    def load_array_data(self, arg0: numpy.ndarray[numpy.int16], arg1: str) -> None:
        ...
    def load_kernel(self, arg0: os.PathLike) -> None:
        ...
    def load_n_clusters(self, arg0: int) -> None:
        ...
    def reset_timer(self) -> None:
        ...
    @property
    def allocated(self) -> bool:
        ...
    @property
    def binary_path(self) -> os.PathLike | None:
        ...
    @property
    def cpu_pim_time(self) -> float:
        ...
    @property
    def data_size(self) -> int | None:
        ...
    @property
    def dpu_run_time(self) -> float:
        ...
    @property
    def hash(self) -> bytes | None:
        ...
    @property
    def inertia(self) -> int:
        ...
    @property
    def nr_dpus(self) -> int:
        ...
    @property
    def pim_cpu_time(self) -> float:
        ...
FEATURE_TYPE: int = 16
__version__: str = '0.2.4'

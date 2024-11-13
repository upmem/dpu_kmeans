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
__all__ = ['Container', 'FEATURE_TYPE']
class Container:
    """
    
            Container object to interface with the DPUs
        
    """
    nr_dpus: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def allocate(self) -> None:
        ...
    def compute_inertia(self, arg0: numpy.ndarray[numpy.int16]) -> int:
        ...
    def free_dpus(self) -> None:
        ...
    def lloyd_iter(self, arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.int64], arg2: numpy.ndarray[numpy.int32]) -> None:
        ...
    def load_array_data(self, arg0: numpy.ndarray[numpy.int16], arg1: int, arg2: int) -> None:
        ...
    def load_kernel(self, arg0: os.PathLike) -> None:
        ...
    def load_n_clusters(self, arg0: int) -> None:
        ...
    def reset_timer(self) -> None:
        ...
    @property
    def cpu_pim_time(self) -> float:
        ...
    @property
    def dpu_run_time(self) -> float:
        ...
    @property
    def pim_cpu_time(self) -> float:
        ...
FEATURE_TYPE: int = 16
__version__: str = '0.2.3'

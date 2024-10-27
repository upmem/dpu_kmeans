"""

        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans._core

        .. autosummary::
           :toctree: _generate

           add
           subtract
           checksum
           Container
    
"""
from __future__ import annotations
import numpy
__all__ = ['Container', 'FEATURE_TYPE', 'add', 'checksum', 'subtract']
class Container:
    """
    
            Container object to interface with the DPUs
        
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def allocate(self) -> None:
        ...
    def allocate_host_memory(self) -> None:
        ...
    def compute_inertia(self, arg0: numpy.ndarray[numpy.int16]) -> int:
        ...
    def deallocate_host_memory(self) -> None:
        ...
    def free_data(self) -> None:
        ...
    def free_dpus(self) -> None:
        ...
    def get_cpu_pim_time(self) -> float:
        ...
    def get_dpu_run_time(self) -> float:
        ...
    def get_nr_dpus(self) -> int:
        ...
    def get_pim_cpu_time(self) -> float:
        ...
    def lloyd_iter(self, arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.int64], arg2: numpy.ndarray[numpy.int32]) -> None:
        ...
    def load_array_data(self, arg0: numpy.ndarray[numpy.int16], arg1: int, arg2: int) -> None:
        ...
    def load_kernel(self, arg0: str) -> None:
        ...
    def load_n_clusters(self, arg0: int) -> None:
        ...
    def reset_timer(self) -> None:
        ...
    def set_nr_dpus(self, arg0: int) -> None:
        ...
def add(arg0: int, arg1: int) -> int:
    """
            Add two numbers
    
            Some other explanation about the add function.
    """
def checksum(arg0: str) -> int:
    """
            Checksum test on dpus
    """
def subtract(arg0: int, arg1: int) -> int:
    """
            Subtract two numbers
    
            Some other explanation about the subtract function.
    """
FEATURE_TYPE: int = 16
__version__: str = '0.2.1'

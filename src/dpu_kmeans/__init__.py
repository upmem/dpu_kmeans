"""DPU KMeans package.

This package provides a KMeans algorithm implemented on a DPU.

:Authors: Sylvan Brocard <sbrocard@upmem.com>
:License: MIT
"""

from ._core import __version__
from ._kmeans import KMeans

__all__ = ["__version__", "KMeans"]

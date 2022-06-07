# -*- coding: utf-8 -*-

# Authors: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

from ._checksum import test_checksum
from ._core import __doc__, __version__, add, subtract
from ._kmeans import KMeans


def add2(x: int, y: int) -> int:
    return add(x, y)

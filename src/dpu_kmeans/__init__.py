# -*- coding: utf-8 -*-
from ._checksum import test_checksum
from ._core import __doc__, __version__, add, subtract
from ._kmeans import KMeans


def add2(x: int, y: int) -> int:
    return add(x, y)

# -*- coding: utf-8 -*-
from ._core import __doc__, __version__, add, subtract, call_home, dpu_test, checksum

# from .base_tree import printbin
from .base_tree import test_dpu_bin, test_checksum


def add2(x: int, y: int) -> int:
    return add(x, y)

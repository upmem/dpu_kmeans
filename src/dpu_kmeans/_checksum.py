# -*- coding: utf-8 -*-

# Authors: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT

try:
    from importlib.resources import as_file, files
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from ._core import checksum


def test_checksum():
    ref = files("dpu_kmeans").joinpath("dpu_program/trivial_checksum_example")
    with as_file(ref) as path:
        return f"{checksum(str(path)):#0{10}x}"

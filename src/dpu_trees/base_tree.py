# -*- coding: utf-8 -*-

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from . import dpu_test, checksum


def test_dpu_bin():
    ref = files("dpu_trees").joinpath("dpu_program/helloworld")
    with as_file(ref) as path:
        dpu_test(str(path))


def test_checksum():
    ref = files("dpu_trees").joinpath("dpu_program/trivial_checksum_example")
    with as_file(ref) as path:
        return f"{checksum(str(path)):#0{10}x}"

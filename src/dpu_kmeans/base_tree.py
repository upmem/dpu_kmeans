# -*- coding: utf-8 -*-

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from ._core import dpu_test, checksum, kmeans_c


def test_dpu_bin():
    ref = files("dpu_kmeans").joinpath("dpu_program/helloworld")
    with as_file(ref) as path:
        dpu_test(str(path))


def test_checksum():
    ref = files("dpu_kmeans").joinpath("dpu_program/trivial_checksum_example")
    with as_file(ref) as path:
        return f"{checksum(str(path)):#0{10}x}"


def test_kmeans(
    filename: str,
    isBinaryFile: bool,
    threshold: float = 0.0001,
    max_nclusters: int = 5,
    min_nclusters: int = 5,
    isRMSE: bool = False,
    isOutput: bool = True,
    nloops: int = 1,
):
    ref = files("dpu_kmeans").joinpath("dpu_program/kmeans_dpu_kernel")
    with as_file(ref) as DPU_BINARY:
        kmeans_c(
            filename,
            isBinaryFile,
            threshold,
            max_nclusters,
            min_nclusters,
            isRMSE,
            isOutput,
            nloops,
            str(DPU_BINARY),
        )

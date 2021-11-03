# -*- coding: utf-8 -*-

import numpy as np
from collections.abc import Iterable
from typing import Union
from os.path import basename, dirname, splitext
from os import PathLike
from datetime import datetime

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file

from ._core import dpu_test, checksum, kmeans_cpp


def test_dpu_bin():
    ref = files("dpu_kmeans").joinpath("dpu_program/helloworld")
    with as_file(ref) as path:
        dpu_test(str(path))


def test_checksum():
    ref = files("dpu_kmeans").joinpath("dpu_program/trivial_checksum_example")
    with as_file(ref) as path:
        return f"{checksum(str(path)):#0{10}x}"


def test_kmeans(
    input: Union[str, PathLike, Iterable],
    is_binary_file: bool = True,
    threshold: float = 0.0001,
    max_nclusters: int = 5,
    min_nclusters: int = 5,
    isRMSE: bool = False,
    isOutput: bool = True,
    nloops: int = 1,
    log_name: str = "",
):
    npoints, nfeatures = 0, 0
    if isinstance(input, (str, PathLike)):
        filename, file_input, data = input, True, []
        if not log_name:
            log_name = (
                dirname(filename)
                + "/kmeanstime_dpu_"
                + splitext(basename(filename))[0]
                + ".log"
            )
    else:
        filename, file_input, data = "", False, input
        npoints, nfeatures = data.shape
        if not log_name:
            log_name = (
                "kmeanstime_dpu_"
                + str(datetime.now().date())
                + "_"
                + str(datetime.now().time()).replace(":", ".")
                + ".log"
            )

    ref = files("dpu_kmeans").joinpath("dpu_program/kmeans_dpu_kernel")
    with as_file(ref) as DPU_BINARY:
        result = kmeans_cpp(
            data,
            filename,
            file_input,
            is_binary_file,
            threshold,
            max_nclusters,
            min_nclusters,
            isRMSE,
            isOutput,
            npoints,
            nfeatures,
            nloops,
            str(DPU_BINARY),
            log_name,
        )
    return result

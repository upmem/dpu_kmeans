# -*- coding: utf-8 -*-

import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages
from setuptools_scm import get_version

VERSION = get_version(local_scheme="no-local-version")
VERSION = "".join([c for c in VERSION if c.isdigit() or c == "."])

# compilation of the host library
setup(
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/dpu_kmeans",
    include_package_data=True,
    install_requires=[
        "numpy",
        "scikit-learn",
        "importlib_resources;python_version<'3.9'",
        "xxhash",
    ],
    extras_require={
        "test": ["pytest"],
        "benchmarks": ["pytest", "pandas", "pyarrow"],
    },
    zip_safe=False,
    cmake_args=[
        "-DNR_TASKLETS=16",  # number of parallel tasklets on each DPU
        f"-DVERSION={VERSION}",
    ],
)

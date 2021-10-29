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

version = get_version(local_scheme="no-local-version")
version = "".join([c for c in version if c.isdigit() or c == "."])

# compilation of the host library
setup(
    name="dpu_kmeans",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
    },
    description="a package for the k-means algorithm on DPU",
    author="Sylvan Brocard",
    author_email="sylvan.brocard@gmail.com",
    url="https://github.com/upmem/dpu_kmeans",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/dpu_kmeans",
    include_package_data=True,
    extras_require={
        "test": ["pytest"],
        ':python_version < "3.9"': ["importlib_resources"],
    },
    zip_safe=False,
    cmake_args=[
        "-DNR_TASKLETS=16",  # number of parallel tasklets on each DPU
        f"-DVERSION={version}",
    ],
)

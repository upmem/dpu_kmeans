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

# compilation of the host library
setup(
    name="dpu_trees",
    version="0.0.4-alpha",
    description="a package for tree algorithms on DPU",
    author="Sylvan Brocard",
    author_email="sylvan.brocard@gmail.com",
    url="https://github.com/SylvanBrocard/dpu_trees",
    download_url="https://github.com/SylvanBrocard/dpu_trees/archive/refs/tags/v0.0.2-alpha.tar.gz",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/dpu_trees",
    include_package_data=True,
    extras_require={
        "test": ["pytest"],
        ':python_version < "3.9"': ["importlib_resources"],
    },
    zip_safe=False,
)

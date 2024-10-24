dpu_kmeans README
=================

| CI         | status                                                       |
| ---------- | ------------------------------------------------------------ |
| pip builds | [![Pip Actions Status][actions-pip-badge]][actions-pip-link] |
| docs       | [![Doc Status][actions-docs-badge]][actions-docs-link]       |

A project built with [pybind11](https://github.com/pybind/pybind11) and [scikit-build](https://github.com/scikit-build/scikit-build), running the KMeans algorithm on in-memory processors with the UPMEM SDK.

[actions-pip-link]:        https://github.com/upmem/dpu_kmeans/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/upmem/dpu_kmeans/workflows/Pip/badge.svg
[actions-docs-link]:       https://github.com/upmem/dpu_kmeans/actions?query=workflow%3Adocumentation
[actions-docs-badge]:      https://github.com/upmem/dpu_kmeans/workflows/documentation/badge.svg

Documentation
-------------

- [API documentation](https://upmem.github.io/dpu_kmeans/)

Installation
------------

- install the [UPMEM SDK](https://sdk.upmem.com/)
- `pip install dpu-kmeans`

If you get an error about RANLIB, create a symlink to `llvm-ar`:

```bash
sudo ln -s /usr/bin/llvm-ar /usr/bin/llvm-ranlib
```

Usage
-----

```python
import numpy as np
from dpu_kmeans import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])

kmeans = KMeans(2)
kmeans.fit(X)

print(kmeans.cluster_centers_)
```

Output:

```python
[[ 0.99986267  2.        ]
 [10.00013733  2.        ]]
```

Development
-----------

- clone this repository
- install the [UPMEM SDK](https://sdk.upmem.com/)
- install the build requirements in [`pyproject.toml`](pyproject.toml)
- `cd dpu_kmeans`
- `pre-commit install`
- `pip install -e .`

OR

- clone this repository
- open folder in VS Code
- start in Dev Container

to debug: `python setup.py develop --build-type Debug`

*Note:* The dev container is for development only and uses the PIM simulator.

Templating
----------

To use this project as a base for your own UPMEM DIMM project:

- click on "Use this template" in github
- create a new project from this one
- turn off Conda and Wheels workflows in github actions as they are not operational right now
- change folder `src/dpu_kmeans` to `src/<your_project>`
- change project name (all instances of `dpu_kmeans`) and info in:
  - README.md
  - setup.cfg
  - setup.py (`cmake_install_dir="src/dpu_kmeans"`)
  - .gitignore (`src/dpu_kmeans/dpu_program/`)
  - CMakeLists.txt (`project(dpu_kmeans VERSION ${VERSION})`)
  - conda.recipe/meta.yaml (optional)
  - docs (optional)
- if you intend to use github actions to auto-publish to pypi, update the project secrets as described in [Publishing package distribution releases using GitHub Actions CI/CD workflows](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

Testing
-------

- clone this repository
- install the [UPMEM SDK](https://sdk.upmem.com/)
- install [nox](https://nox.thea.codes/)
- `cd dpu_trees`
- `nox`

OR

- clone this repository
- open folder in VS Code
- start in Dev Container
- `nox`

*Note:* `nox`, `python setup.py` and `pip` might fail if you executed `pip install -e .` previously, delete the `_skbuild` cache or run `python setup.py clean` to solve.

Test call
---------

```python
import dpu_kmeans
dpu_kmeans.test_checksum()
```

Expected return: `0x007f8000`

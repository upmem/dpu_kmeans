dpu_trees
==============

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Actions Status][actions-pip-badge]][actions-pip-link] |

A project built with [pybind11](https://github.com/pybind/pybind11) and [scikit-build](https://github.com/scikit-build/scikit-build), running the KMeans algorithm on in-memory processors with the UPMEM SDK.

[actions-badge]:           https://github.com/SylvanBrocard/dpu_trees/workflows/Tests/badge.svg
[actions-conda-link]:      https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3AConda
[actions-conda-badge]:     https://github.com/SylvanBrocard/dpu_trees/workflows/Conda/badge.svg
[actions-pip-link]:        https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/SylvanBrocard/dpu_trees/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/SylvanBrocard/dpu_trees/workflows/Wheels/badge.svg

Installation
------------

- install the [UPMEM SDK](https://sdk.upmem.com/)
- `pip install dpu-kmeans`

Usage
-----

```python
import numpy as np
from dpu_kmeans import DIMM_data, KMeans

X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
dimm_data = DIMM_data(X)

kmeans = KMeans(2)

centroids, iterations, time = kmeans.fit(dimm_data)
print(centroids)
```

Output:

```python
[[ 0.9998627  2.       ]
 [10.000137   2.       ]]
```

Alternatively you can import data from a CSV file:

```python
dimm_data = DIMM_data("/path/to/data")
```

Development
-----------

- clone this repository
- install the [UPMEM SDK](https://sdk.upmem.com/)
- install the build requirements in [`pyproject.toml`](pyproject.toml)
- `cd dpu_kmeans`
- `pre-commit install`
- `pip install -e .`
- `python setup.py clean`

OR

- clone this repository
- open folder in VS Code
- start in Dev Container

to debug: `python setup.py develop --build-type Debug`

*Note:* The dev container is for development only and uses the PIM simulator.

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

[`cibuildwheel`]:          https://cibuildwheel.readthedocs.io

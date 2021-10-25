dpu_trees
==============

|      CI              | status |
|----------------------|--------|
| conda.recipe         | [![Conda Actions Status][actions-conda-badge]][actions-conda-link] |
| pip builds           | [![Pip Actions Status][actions-pip-badge]][actions-pip-link] |
| wheel                | [![Wheels Actions Status][actions-wheels-badge]][actions-wheels-link] |

A project built with [pybind11](https://github.com/pybind/pybind11) and scikit-build, running DPU programs with the UPMEM SDK.

[actions-badge]:           https://github.com/SylvanBrocard/dpu_trees/workflows/Tests/badge.svg
[actions-conda-link]:      https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3AConda
[actions-conda-badge]:     https://github.com/SylvanBrocard/dpu_trees/workflows/Conda/badge.svg
[actions-pip-link]:        https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/SylvanBrocard/dpu_trees/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/SylvanBrocard/dpu_trees/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/SylvanBrocard/dpu_trees/workflows/Wheels/badge.svg

Installation
------------

- `pip install dpu-trees`

OR

- clone this repository
- install the [UPMEM SDK](https://sdk.upmem.com/)
- `pip install ./dpu_trees`

Development
-----------

- clone this repository
- install the [UPMEM SDK](https://sdk.upmem.com/)
- `cd dpu_trees`
- `pre-commit install`
- `python3 setup.py develop`
- `python setup.py clean`

OR

- clone this repository
- open folder in VS Code
- start in Dev Container

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

*Note:* `nox` and `pip` will fail if you executed `setup.py install` or `setup.py develop`, delete the cache by running `python setup.py clean` to solve.

Test call
---------

```python
import dpu_trees
dpu_trees.add(1, 2)
```

[`cibuildwheel`]:          https://cibuildwheel.readthedocs.io

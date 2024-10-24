# Author: Sylvan Brocard <sbrocard@upmem.com>
# License: MIT
"""Module for testing the _core module stub generation."""

import subprocess
from pathlib import Path


def test_stub(tmp_path):
    """Test _core module stub generation and verify its correctness."""
    # generate stub
    result = subprocess.run(
        ["pybind11-stubgen", "dpu_kmeans._core", "--output", tmp_path],
        check=True,
    )
    assert result.returncode == 0, f"Failed to generate stub: {result.stderr}"

    # compare the generated stub with the distributed one
    with Path("src/dpu_kmeans/_core.pyi").open(
        "r",
        encoding="utf-8",
    ) as distributed_stub:
        expected = distributed_stub.read()
    with Path(f"{tmp_path}/dpu_kmeans/_core.pyi").open(
        "r",
        encoding="utf-8",
    ) as generated_stub:
        generated = generated_stub.read()
    assert expected == generated, (
        "Generated stub is different from the distributed one, "
        "re-run `pybind11-stubgen dpu_kmeans._core --output src`"
    )

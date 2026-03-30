import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

import naturalneighbor
from naturalneighbor import griddata
import naturalneighbor.naturalneighbor as nn_mod


cuda_only = pytest.mark.skipif(
    not naturalneighbor.CUDA_AVAILABLE,
    reason="CUDA extension not built or not importable",
)


def known_cube(side_length=1):
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return corners * side_length


@cuda_only
@pytest.mark.parametrize(
    "grid_ranges",
    [
        [[0, 1, 2j], [0, 1, 2j], [0, 1, 2j]],
        [[0, 1, 4j], [0, 1, 7j], [0, 1, 10j]],
    ],
)
def test_cuda_interp_on_known_points(grid_ranges):
    known_points = known_cube()
    known_values = np.random.rand(8)

    actual = griddata(
        known_points,
        known_values,
        grid_ranges,
        backend="cuda",
    )

    for value, point in zip(known_values, known_points):
        i, j, k = point.astype(int)
        assert_allclose(actual[-i, -j, -k], value, rtol=0, atol=1e-8)


@cuda_only
def test_cuda_interp_constant_values():
    known_points = known_cube()
    known_values = np.ones((8,)) * 7

    interp_grid_ranges = [
        [0, 1.5, 0.5],
        [0, 1.5, 0.5],
        [0, 1.5, 0.5],
    ]

    actual = griddata(
        known_points,
        known_values,
        interp_grid_ranges,
        backend="cuda",
    )

    expected = np.ones_like(actual) * 7
    assert_allclose(actual, expected, rtol=0, atol=1e-8)


@cuda_only
@pytest.mark.parametrize("num_points", [10, 11])
def test_cuda_cube_symmetrical(num_points):
    known_points = known_cube()
    known_values = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    interp_grid_ranges = [
        [0, 1, num_points * 1j],
        [0, 1, num_points * 1j],
        [0, 1, num_points * 1j],
    ]

    interp_values = griddata(
        known_points,
        known_values,
        interp_grid_ranges,
        backend="cuda",
    )

    middle = int(math.floor(num_points / 2))
    q1 = interp_values[0:middle, 0:middle, :]
    q2 = interp_values[0:middle, : -middle - 1 : -1, :]
    q3 = interp_values[: -middle - 1 : -1, 0:middle, :]
    q4 = interp_values[: -middle - 1 : -1, : -middle - 1 : -1, :]

    assert_allclose(q1, q2, rtol=0, atol=1e-9)
    assert_allclose(q1, q3, rtol=0, atol=1e-9)
    assert_allclose(q1, q4, rtol=0, atol=1e-9)


@cuda_only
def test_cuda_backend_matches_cpu():
    """CUDA brute-force NN + per-voxel accumulation must be bit-identical to CPU."""
    np.random.seed(0)
    points = np.random.rand(12, 3).astype(np.float64)
    values = np.random.rand(12).astype(np.float64)
    grid_ranges = [
        [0, 1, 5j],
        [0, 1, 5j],
        [0, 1, 5j],
    ]

    cpu_out = griddata(points, values, grid_ranges, backend="cpu")
    cuda_out = griddata(points, values, grid_ranges, backend="cuda")

    assert_allclose(cpu_out, cuda_out, rtol=0, atol=0)


def test_backend_cpu_when_cuda_available():
    """Explicit backend='cpu' must not use CUDA."""
    np.random.seed(1)
    points = np.random.rand(6, 3).astype(np.float64)
    values = np.random.rand(6).astype(np.float64)
    grid_ranges = [
        [0, 1, 3j],
        [0, 1, 3j],
        [0, 1, 3j],
    ]
    out = griddata(points, values, grid_ranges, backend="cpu")
    assert out.shape == (3, 3, 3)


def test_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        griddata(
            np.random.rand(2, 3),
            np.random.rand(2),
            [[0, 1, 2j], [0, 1, 2j], [0, 1, 2j]],
            backend="gpu",
        )


@cuda_only
def test_cuda_import_error_when_forcing_cuda_without_ext(monkeypatch):
    saved = nn_mod.ccudanaturalneighbor
    try:
        nn_mod.ccudanaturalneighbor = None
        with pytest.raises(ImportError, match="CUDA backend"):
            griddata(
                known_cube(),
                np.random.rand(8),
                [[0, 1, 2j], [0, 1, 2j], [0, 1, 2j]],
                backend="cuda",
            )
    finally:
        nn_mod.ccudanaturalneighbor = saved

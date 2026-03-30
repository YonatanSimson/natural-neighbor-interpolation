"""
Comparison of natural neighbor and linear barycentric interpolation.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import scipy.interpolate

mpl.use("Agg")  # so it can run on Travis without a display
import matplotlib.pyplot as plt

import naturalneighbor

_DEMO_DIR = Path(__file__).resolve().parent


def error_str(errors):
    numerical_error = errors[~np.isnan(errors)]
    mean_err = np.mean(numerical_error)
    std_err = np.std(numerical_error)
    max_err = np.max(numerical_error)
    return "(Mean={:.2f}, Std={:.2f} Max={:.2f})".format(mean_err, std_err, max_err)


def compare_interp_for_func(
    func,
    func_as_string,
    image_name,
    xmax=60,
    ymax=None,
    zmax=None,
    z_slice=None,
    num_known_points=100,
    transpose_slice=False,
):
    if ymax is None:
        ymax = xmax
    if zmax is None:
        zmax = xmax
    if z_slice is None:
        z_slice = 20
    final_shape = (xmax, ymax, zmax)

    scales = np.array([xmax, ymax, zmax], dtype=np.float64)
    known_points = np.round(np.random.rand(num_known_points, 3) * scales)

    grid_ranges = [
        [0, xmax, 1],
        [0, ymax, 1],
        [0, zmax, 1],
    ]

    grid = np.mgrid[0:xmax:1, 0:ymax:1, 0:zmax:1]

    known_values = np.array([func(*point) for point in known_points], dtype=np.float64)
    true_values = np.reshape([func(x, y, z) for x, y, z in zip(*grid)], final_shape)

    linear_interp = scipy.interpolate.griddata(known_points, known_values, tuple(grid), method="linear")

    nn_interp = naturalneighbor.griddata(known_points, known_values, grid_ranges)
    nn_interp[np.isnan(linear_interp)] = float("nan")

    nn_interp_slice = nn_interp[:, :, z_slice]
    linear_interp_slice = linear_interp[:, :, z_slice]
    true_values_slice = true_values[:, :, z_slice]

    if transpose_slice:
        nn_interp_slice = nn_interp_slice.T
        linear_interp_slice = linear_interp_slice.T
        true_values_slice = true_values_slice.T

    nn_interp_err = np.abs(nn_interp_slice - true_values_slice)
    linear_interp_err = np.abs(linear_interp_slice - true_values_slice)

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(true_values_slice)
    ax1.set_title("True Values\n{}".format(func_as_string))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(nn_interp_err)
    nn_error_str = error_str(nn_interp_err)
    ax2.set_title("Natural Neighbor Abs Error\n{}".format(nn_error_str))

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(linear_interp_err)
    linear_error_str = error_str(linear_interp_err)
    ax3.set_title("Linear Barycentric Abs Error\n{}".format(linear_error_str))

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(nn_interp_slice)
    ax5.set_title("Natural Neighbor Values")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(linear_interp_slice)
    ax6.set_title("Linear Barycentric Values")

    plt.savefig(_DEMO_DIR / image_name, dpi=100)


def staircase_height_variable_m(x, tread_widths_m, rise_heights_m):
    """
    Piecewise-constant height along +x (meters): each tread has its own width; each
    riser its own height. On tread i, elevation is sum(rise[0:i]).
    """
    tread_widths_m = np.asarray(tread_widths_m, dtype=np.float64).ravel()
    rise_heights_m = np.asarray(rise_heights_m, dtype=np.float64).ravel()
    n = tread_widths_m.size
    if rise_heights_m.size != n:
        raise ValueError("tread_widths_m and rise_heights_m must have the same length")

    boundaries = np.concatenate([[0.0], np.cumsum(tread_widths_m)])
    # Elevation on tread i before climbing riser i: sum(rise[0:i])
    h_on_tread = np.cumsum(np.concatenate([[0.0], rise_heights_m[:-1]]))

    x_arr = np.asarray(x, dtype=np.float64)
    scalar = x_arr.ndim == 0
    xf = np.atleast_1d(x_arr)
    idx = np.searchsorted(boundaries, xf, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    out = h_on_tread[idx]
    return out.item() if scalar else out


if __name__ == "__main__":
    np.random.seed(100)

    compare_interp_for_func(
        (lambda x, y, z: np.sin(y / 10) + np.sin(x / 10)),
        "sin(y/10) + sin(x/10)",
        "sin_sin_comparison.png",
    )

    compare_interp_for_func(
        (lambda x, y, z: x + np.sin(x / 10) / 10),
        "x + sin(x/10)/10",
        "linear_comparison.png",
    )

    # Physical staircase: tread widths in [0.5, 3] m (relative), scaled to run_length_m
    # along +x; variable riser heights (meters). Grid: 0.1 m per index; y,z span 1 m.
    cell_m = 0.1
    run_length_m = 10.0
    lateral_width_m = 1.0
    rng_stair = np.random.default_rng(101)
    tread_widths_m = 0.5 + (3.0 - 0.5) * rng_stair.random(10)
    tread_widths_m[0] = 0.5
    tread_widths_m[9] = 3.0
    tread_widths_m *= run_length_m / np.sum(tread_widths_m)
    rise_heights_m = np.array([0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.09, 0.12, 0.08, 0.11], dtype=np.float64)
    total_run_m = float(np.sum(tread_widths_m))
    xmax = int(np.ceil(total_run_m / cell_m)) + 1
    ymax = int(np.ceil(lateral_width_m / cell_m)) + 1
    zmax = ymax
    stair_label = "10 steps, {:.0f} m run along x, tread {:.2f}–{:.2f} m, rise {:.2f}–{:.2f} m " "(0.1×index→m)".format(
        run_length_m,
        float(np.min(tread_widths_m)),
        float(np.max(tread_widths_m)),
        float(np.min(rise_heights_m)),
        float(np.max(rise_heights_m)),
    )
    compare_interp_for_func(
        lambda x, y, z: staircase_height_variable_m(cell_m * x, tread_widths_m, rise_heights_m),
        stair_label,
        "staircase_steps_comparison.png",
        xmax=xmax,
        ymax=ymax,
        zmax=zmax,
        z_slice=zmax // 2,
        num_known_points=800,
        # imshow uses dim0=vertical, dim1=horizontal; without transpose, long x-run was vertical.
        transpose_slice=True,
    )

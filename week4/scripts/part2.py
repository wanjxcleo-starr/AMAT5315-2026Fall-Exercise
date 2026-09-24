#!/usr/bin/env python3
"""Run the current learning sheet's Week 4 Part 2 comparisons and plot."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


WEEK4 = Path(__file__).resolve().parents[1]
TAYLOR_GREEN = WEEK4 / "artifacts" / "taylor-green"
EVIDENCE = WEEK4 / "evidence"


def relative_velocity_error(recorded: dict, exact: dict) -> float:
    """Return ||(u,v) - (u_exact,v_exact)||_2 / ||(u_exact,v_exact)||_2."""
    recorded_u = np.asarray(recorded["u"], dtype=float)
    recorded_v = np.asarray(recorded["v"], dtype=float)
    exact_u = np.asarray(exact["u"], dtype=float)
    exact_v = np.asarray(exact["v"], dtype=float)
    if not (
        recorded_u.shape == recorded_v.shape == exact_u.shape == exact_v.shape
    ):
        raise ValueError("recorded and exact velocity arrays must have matching shapes")
    numerator = np.sqrt(
        np.sum((recorded_u - exact_u) ** 2 + (recorded_v - exact_v) ** 2)
    )
    denominator = np.sqrt(np.sum(exact_u**2 + exact_v**2))
    if denominator == 0.0:
        raise ValueError("exact velocity norm is zero")
    return float(numerator / denominator)


def derivative_study() -> dict:
    completed = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "--bin", "part2-study"],
        cwd=WEEK4,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def validate_derivative_study(document: dict) -> None:
    rows = document["derivatives"]
    for row in rows:
        fourier_error = row["fourier_n32"]
        if fourier_error >= 1.0e-10:
            raise RuntimeError(
                f"{row['derivative']} Fourier error {fourier_error:.6e} is not below 1e-10"
            )
        ratio = row["finite_difference_n32"] / row["finite_difference_n64"]
        if not 3.8 <= ratio <= 4.1:
            raise RuntimeError(
                f"{row['derivative']} finite-difference error ratio {ratio:.6f} is not near 4"
            )
    if len(rows) != 4:
        raise RuntimeError(f"expected four derivative comparisons, got {len(rows)}")


def write_exact_field() -> dict:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--bin",
            "field",
            "--",
            "taylor-green",
            "--n",
            "64",
            "--nu",
            "0.1",
            "--t",
            "1",
        ],
        cwd=WEEK4,
        check=True,
        capture_output=True,
        text=True,
    )
    exact_path = TAYLOR_GREEN / "exact-t1.json"
    exact_path.write_text(completed.stdout, encoding="utf-8")
    return json.loads(completed.stdout)


def load_first_and_last_frames() -> tuple[dict, dict]:
    fields_path = TAYLOR_GREEN / "fields.jsonl"
    first = None
    last = None
    with fields_path.open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if first is None:
                first = frame
            last = frame
    if first is None or last is None:
        raise ValueError(f"{fields_path} contains no frames")
    return first, last


def print_derivative_table(document: dict) -> None:
    labels = {
        "dx": "dx g",
        "dxx": "dxx g",
        "dxdy": "dxdy g",
        "laplacian": "Laplacian g",
    }
    print(
        f"{'Derivative':<14} {'FD N=32':>12} {'FD N=64':>12} "
        f"{'ratio':>9} {'Fourier N=32':>16}"
    )
    for row in document["derivatives"]:
        coarse = row["finite_difference_n32"]
        fine = row["finite_difference_n64"]
        print(
            f"{labels[row['derivative']]:<14} {coarse:12.8f} {fine:12.8f} "
            f"{coarse / fine:9.5f} {row['fourier_n32']:16.8e}"
        )


def write_taylor_green_figure(first: dict, last: dict) -> Path:
    n = int(round(math.sqrt(len(first["omega"]))))
    if n * n != len(first["omega"]) or len(last["omega"]) != n * n:
        raise ValueError("Taylor-Green frames are not square grids of equal size")
    coordinates = 2.0 * math.pi * np.arange(n) / n
    x, y = np.meshgrid(coordinates, coordinates)
    frames = [first, last]
    vorticity_limit = max(
        float(np.max(np.abs(np.asarray(frame["omega"])))) for frame in frames
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    image = None
    arrow_stride = 4
    for axis, frame in zip(axes, frames, strict=True):
        omega = np.asarray(frame["omega"]).reshape((n, n))
        u = np.asarray(frame["u"]).reshape((n, n))
        v = np.asarray(frame["v"]).reshape((n, n))
        image = axis.imshow(
            omega,
            origin="lower",
            extent=(0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
            cmap="coolwarm",
            vmin=-vorticity_limit,
            vmax=vorticity_limit,
            interpolation="bilinear",
        )
        axis.quiver(
            x[::arrow_stride, ::arrow_stride],
            y[::arrow_stride, ::arrow_stride],
            u[::arrow_stride, ::arrow_stride],
            v[::arrow_stride, ::arrow_stride],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=2.7,
            width=0.004,
        )
        axis.set_xticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
        axis.set_yticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"t={frame['t']:g}, max|ω|={float(np.max(np.abs(omega))):.3f}"
        )
        axis.set_aspect("equal")
    figure.colorbar(image, ax=axes, label="vorticity ω", shrink=0.88)
    figure.suptitle("Taylor–Green decay, n=64, ν=0.1; shared colour scale")
    destination = EVIDENCE / "taylor-green.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    derivatives = derivative_study()
    validate_derivative_study(derivatives)
    print_derivative_table(derivatives)
    exact = write_exact_field()
    first, last = load_first_and_last_frames()
    velocity_error = relative_velocity_error(last, exact)
    print(f"Taylor-Green final relative velocity error: {velocity_error:.12e}")
    if velocity_error >= 1.0e-5:
        raise RuntimeError(
            f"Taylor-Green velocity error {velocity_error:.6e} is not below 1e-5"
        )
    destination = write_taylor_green_figure(first, last)
    print(f"wrote {destination.relative_to(WEEK4)}")


if __name__ == "__main__":
    main()

"""Render three dimer energy-error runs supplied as CSV on standard input."""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


EXPECTED_STEPS = {"euler_500": 500, "verlet_500": 500, "verlet_5000": 5000}


def read_series() -> dict[str, list[tuple[float, float]]]:
    reader = csv.DictReader(sys.stdin)
    if reader.fieldnames != ["run", "step", "time", "relative_error"]:
        raise ValueError("expected run,step,time,relative_error CSV header")

    series: dict[str, list[tuple[float, float]]] = {
        name: [] for name in EXPECTED_STEPS
    }
    for row in reader:
        name = row["run"]
        if name not in series:
            raise ValueError(f"unknown run: {name}")
        step = int(row["step"])
        time = float(row["time"])
        error = float(row["relative_error"])
        if step != len(series[name]) or not math.isfinite(time) or not math.isfinite(error):
            raise ValueError(f"invalid observation in {name} at step {step}")
        if not math.isclose(time, step * 0.01, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"unexpected time in {name} at step {step}")
        series[name].append((time, error))

    for name, steps in EXPECTED_STEPS.items():
        if len(series[name]) != steps + 1:
            raise ValueError(f"{name} has {len(series[name])} rows; expected {steps + 1}")
        if series[name][0][1] != 0.0:
            raise ValueError(f"{name} must start with zero relative error")
    return series


def build_figure(series: dict[str, list[tuple[float, float]]]) -> Figure:
    euler = series["euler_500"]
    verlet_short = series["verlet_500"]
    verlet_long = series["verlet_5000"]
    verlet_max = max(abs(error) for _, error in verlet_short[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    fig.suptitle("Lennard–Jones dimer: total-energy error", y=0.98, fontsize=15)

    axes[0].plot([time for time, _ in euler], [error for _, error in euler],
                 color="#bb5a34", linewidth=1.5, label="Euler")
    axes[0].plot([time for time, _ in verlet_short],
                 [error for _, error in verlet_short],
                 color="#276b9a", linewidth=1.5, linestyle="--",
                 label="velocity-Verlet", zorder=3)
    axes[0].axhline(0.0, color="#506070", linewidth=0.8, zorder=1)
    axes[0].set(title="Euler vs velocity-Verlet · 500 steps", xlabel="Time",
                ylabel=r"Signed $(E-E_0)/|E_0|$")
    axes[0].set_xlim(0.0, 5.0)
    axes[0].legend(loc="upper left", framealpha=0.95)

    axes[1].plot([time for time, _ in verlet_long],
                 [1000.0 * error for _, error in verlet_long],
                 color="#276b9a", linewidth=1.1)
    axes[1].axhline(0.0, color="#506070", linewidth=0.8)
    axes[1].axvspan(0.0, 5.0, color="#dceefa", alpha=0.55, zorder=-1)
    axes[1].set(title="Velocity-Verlet · 5000 steps", xlabel="Time",
                ylabel=r"Signed $1000\,(E-E_0)/|E_0|$")
    axes[1].set_xlim(0.0, 50.0)

    for axis in axes:
        axis.grid(alpha=0.22)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.19, wspace=0.24)
    fig.text(0.75, 0.035, f"Verlet max |error| in first 500 steps: {verlet_max:.3e}",
             ha="center", fontsize=9, color="#214861")
    return fig


def render(series: dict[str, list[tuple[float, float]]], output: Path) -> None:
    fig = build_figure(series)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_dimer.py OUTPUT.png")
    render(read_series(), Path(sys.argv[1]))

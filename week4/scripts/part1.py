#!/usr/bin/env python3
"""Generate the Week 4 Part 1 stability and accuracy figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402


WEEK4 = Path(__file__).resolve().parents[1]
EVIDENCE = WEEK4 / "evidence"


def stability_function(method: str, z):
    """Evaluate the stability polynomial named by ``method``."""
    if method == "euler":
        return 1.0 + z
    if method == "midpoint":
        return 1.0 + z + z**2 / 2.0
    if method == "rk4":
        return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    raise ValueError(f"unknown method: {method}")


def load_study() -> dict:
    completed = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "--bin", "line-study"],
        cwd=WEEK4,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def validate_study(document: dict) -> None:
    expected_slopes = {
        "euler": 1.0,
        "midpoint": 2.0,
        "rk4": 4.0,
        "equal_weight_rk4": 2.0,
    }
    for method, expected in expected_slopes.items():
        actual = document["accuracy"]["slopes"][method]
        if abs(actual - expected) > 0.15 * expected:
            raise RuntimeError(
                f"{method} slope {actual:.6f} is not within 15% of {expected:g}"
            )

    study = document["stability"]
    real = np.asarray(study["real_axis"])
    imaginary = np.asarray(study["imaginary_axis"])
    real_grid, imaginary_grid = np.meshgrid(real, imaginary)
    analytic = np.abs(stability_function("rk4", real_grid + 1j * imaginary_grid))
    measured = np.asarray(study["rk4_growth"])
    discrepancy = float(np.max(np.abs(measured - analytic)))
    if discrepancy > 1.0e-12:
        raise RuntimeError(
            f"measured RK4 growth differs from its polynomial by {discrepancy:.3e}"
        )


def write_stability_figure(document: dict) -> Path:
    study = document["stability"]
    real = np.asarray(study["real_axis"])
    imaginary = np.asarray(study["imaginary_axis"])
    real_grid, imaginary_grid = np.meshgrid(real, imaginary)
    z = real_grid + 1j * imaginary_grid
    measured_growth = np.asarray(study["rk4_growth"])

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    color_mesh = axes[0].pcolormesh(
        real,
        imaginary,
        np.maximum(measured_growth, 1.0e-3),
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=0.1, vmax=10.0),
    )
    contour_styles = {
        "rk4": ("black", "-", 1.8),
        "euler": ("white", "--", 1.2),
        "midpoint": ("white", ":", 1.5),
    }
    for method, (color, line_style, width) in contour_styles.items():
        axes[0].contour(
            real_grid,
            imaginary_grid,
            np.abs(stability_function(method, z)),
            levels=[1.0],
            colors=[color],
            linestyles=[line_style],
            linewidths=[width],
        )
        axes[0].plot(
            [],
            [],
            color=color,
            linestyle=line_style,
            linewidth=width,
            label=f"|R_{method}| = 1",
        )

    markers = [("o", "#ff9f1c"), ("^", "#e71d36")]
    for mode_set, (marker, color) in zip(study["modes"], markers, strict=True):
        points = np.asarray(mode_set["points"])
        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            label=f"line modes, Δt={mode_set['dt']:.3f}",
        )
    axes[0].axhline(0.0, color="0.7", linewidth=0.5)
    axes[0].axvline(0.0, color="0.7", linewidth=0.5)
    axes[0].set_xlim(-4.0, 1.0)
    axes[0].set_ylim(-4.0, 4.0)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Re(z)")
    axes[0].set_ylabel("Im(z)")
    axes[0].set_title("Measured RK4 growth per step")
    axes[0].legend(fontsize=7, loc="upper right")
    figure.colorbar(color_mesh, ax=axes[0], label="|y₁/y₀|", shrink=0.84)

    pulse_image = None
    for axis, key, title in [
        (axes[1], "stable_pulse", "stable: Δt=0.045"),
        (axes[2], "unstable_pulse", "unstable: Δt=0.056"),
    ]:
        pulse = study[key]
        states = np.asarray(pulse["states"])
        pulse_image = axis.imshow(
            states,
            extent=(0.0, 2.0 * math.pi, pulse["times"][-1], 0.0),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("t (increases downward)")
        axis.set_xticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
        axis.set_title(title)
    figure.colorbar(pulse_image, ax=axes[1:], label="u(x,t)", shrink=0.84)
    figure.suptitle("Advection–diffusion stability: n=64, c=1, ν=0.05")

    destination = EVIDENCE / "line-stability.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _periodic_curve(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.append(x, 2.0 * math.pi), np.append(values, values[0])


def write_accuracy_figure(document: dict) -> Path:
    study = document["accuracy"]
    x = np.asarray(study["x"])
    exact_x, exact_values = _periodic_curve(x, np.asarray(study["exact_profile"]))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    axes[0].plot(exact_x, exact_values, "k-", linewidth=2.2, label="exact")
    for profile in study["profiles"]:
        plot_x, values = _periodic_curve(x, np.asarray(profile["values"]))
        axes[0].plot(
            plot_x,
            values,
            linewidth=1.4,
            label=f"{profile['label']} (err={profile['max_error']:.2e})",
        )
    axes[0].set_xlim(0.0, 2.0 * math.pi)
    axes[0].set_xticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, 2π)")
    axes[0].set_title("One-lap pulse propagation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    time_steps = np.asarray(study["time_steps"])
    display_names = {
        "euler": "Euler",
        "midpoint": "explicit midpoint",
        "rk4": "RK4",
        "equal_weight_rk4": "equal-weight RK4",
    }
    for method, display_name in display_names.items():
        errors = np.asarray(study["convergence_errors"][method])
        slope = study["slopes"][method]
        coefficients = np.polyfit(np.log(time_steps), np.log(errors), 1)
        fit = np.exp(coefficients[1]) * time_steps ** coefficients[0]
        (points,) = axes[1].loglog(
            time_steps,
            errors,
            "o",
            label=f"{display_name}, slope {slope:.2f}",
        )
        axes[1].loglog(
            time_steps,
            fit,
            "--",
            color=points.get_color(),
            linewidth=1.0,
        )
    axes[1].set_xlabel("time step Δt")
    axes[1].set_ylabel("maximum error at t=1")
    axes[1].set_title("Temporal convergence with Fourier derivatives")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Advection–diffusion accuracy, n=64, c=1")

    destination = EVIDENCE / "line-accuracy.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def print_summary(document: dict) -> None:
    print("One-lap maximum errors:")
    for profile in document["accuracy"]["profiles"]:
        print(f"  {profile['label']}: {profile['max_error']:.12e}")
    print("Fitted temporal slopes:")
    for method in ("euler", "midpoint", "rk4", "equal_weight_rk4"):
        print(f"  {method}: {document['accuracy']['slopes'][method]:.6f}")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    document = load_study()
    validate_study(document)
    stability_path = write_stability_figure(document)
    accuracy_path = write_accuracy_figure(document)
    print_summary(document)
    print(f"wrote {stability_path.relative_to(WEEK4)}")
    print(f"wrote {accuracy_path.relative_to(WEEK4)}")


if __name__ == "__main__":
    main()

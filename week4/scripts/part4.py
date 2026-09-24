#!/usr/bin/env python3
"""Run and plot the current learning sheet's Week 4 Part 4 experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


WEEK4 = Path(__file__).resolve().parents[1]
ARTIFACTS = WEEK4 / "artifacts"
EVIDENCE = WEEK4 / "evidence"
BIN = WEEK4 / "target" / "release"


@dataclass(frozen=True)
class RunSpec:
    case: str
    n: int
    nu: float
    dt: float
    t_end: float
    output_directory: Path


@dataclass(frozen=True)
class StoredRun:
    spec: RunSpec
    fields_path: Path
    reused_from: Path | None


def relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return ||actual-reference||_2 / ||reference||_2."""
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm((actual - reference).ravel()) / denominator)


def fit_log_slope(time_steps: np.ndarray, errors: np.ndarray) -> float:
    """Fit log(error) = q log(dt) + constant and return q."""
    if np.any(time_steps <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("time steps and errors must be positive")
    return float(np.polyfit(np.log(time_steps), np.log(errors), 1)[0])


def richardson_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    refinement_ratio: float = 2.0,
    order: int = 4,
) -> float:
    """Estimate the fine solution's relative error from a nested pair."""
    return relative_error(coarse, fine) / (refinement_ratio**order - 1.0)


def choose_time_step(
    base_dt: float,
    base_error: float,
    candidates: Iterable[float],
    tolerance: float,
    order: int = 4,
) -> tuple[float | None, dict[float, float]]:
    """Choose the largest candidate with a power-law error below tolerance."""
    predictions = {
        candidate: base_error * (candidate / base_dt) ** order
        for candidate in candidates
    }
    acceptable = [
        candidate
        for candidate, prediction in predictions.items()
        if prediction < tolerance
    ]
    chosen = max(acceptable) if acceptable else None
    return chosen, predictions


def frame_at_time(fields_path: Path, target_time: float) -> dict[str, object]:
    """Load the frame at target_time without reading the movie into memory."""
    with fields_path.open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if math.isclose(float(frame["t"]), target_time, abs_tol=1.0e-12):
                return frame
    raise RuntimeError(f"no t={target_time:g} frame in {fields_path}")


def metadata_matches(spec: RunSpec, directory: Path) -> bool:
    metadata_path = directory / "run.json"
    fields_path = directory / "fields.jsonl"
    if not metadata_path.is_file() or not fields_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_band = [2, 6] if spec.case == "random" else None
    return (
        metadata["case"] == spec.case
        and metadata["n"] == spec.n
        and metadata["method"] == "rk4"
        and math.isclose(metadata["nu"], spec.nu)
        and math.isclose(metadata["dt"], spec.dt)
        and metadata.get("seed") == (2026 if spec.case == "random" else None)
        and metadata.get("k_band") == expected_band
        and metadata["t_end"] >= spec.t_end
        and fields_path.stat().st_size > 0
    )


def field_command(spec: RunSpec, exact_time: float | None = None) -> list[str]:
    if spec.case == "taylor-green":
        command = [str(BIN / "field"), "taylor-green", "--n", str(spec.n)]
        if exact_time is not None:
            command += ["--nu", str(spec.nu), "--t", str(exact_time)]
        return command
    if spec.case == "random":
        return [
            str(BIN / "field"),
            "random",
            "--n",
            str(spec.n),
            "--seed",
            "2026",
            "--k-min",
            "2",
            "--k-max",
            "6",
        ]
    raise ValueError(f"unknown case: {spec.case}")


def run_case(spec: RunSpec, reuse_directories: Iterable[Path] = ()) -> StoredRun:
    candidates = [spec.output_directory, *reuse_directories]
    for directory in candidates:
        if metadata_matches(spec, directory):
            fields_path = directory / "fields.jsonl"
            frame_at_time(fields_path, spec.t_end)
            reused_from = None if directory == spec.output_directory else directory
            print(f"reuse {fields_path.relative_to(WEEK4)}", flush=True)
            return StoredRun(spec, fields_path, reused_from)

    spec.output_directory.mkdir(parents=True, exist_ok=True)
    initial = subprocess.run(
        field_command(spec), check=True, capture_output=True, text=True
    ).stdout
    print(
        f"run {spec.case}: N={spec.n}, dt={spec.dt:g}, t_end={spec.t_end:g}",
        flush=True,
    )
    completed = subprocess.run(
        [
            str(BIN / "fluid"),
            "--method",
            "rk4",
            "--nu",
            str(spec.nu),
            "--dt",
            str(spec.dt),
            "--t-end",
            str(spec.t_end),
            "--every",
            str(spec.t_end),
            "--out",
            str(spec.output_directory),
        ],
        input=initial,
        capture_output=True,
        text=True,
    )
    table_path = spec.output_directory / "energy.tsv"
    table_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{spec.case} dt={spec.dt:g} failed with exit "
            f"{completed.returncode}: {completed.stderr}"
        )
    fields_path = spec.output_directory / "fields.jsonl"
    frame_at_time(fields_path, spec.t_end)
    return StoredRun(spec, fields_path, None)


def dt_label(dt: float) -> str:
    return format(dt, "g")


def velocity(frame: dict[str, object]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(frame["u"], dtype=float),
            np.asarray(frame["v"], dtype=float),
        )
    )


def vorticity(frame: dict[str, object]) -> np.ndarray:
    return np.asarray(frame["omega"], dtype=float)


def fitted_curve(steps: np.ndarray, errors: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
    return np.exp(intercept) * steps**slope


def plot_order(
    steps: np.ndarray,
    errors: np.ndarray,
    slope: float,
    storage_floor: float,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    axis.loglog(steps, errors, "o", markersize=7, label="measured RK4 error")
    order = np.argsort(steps)
    axis.loglog(
        steps[order],
        fitted_curve(steps, errors)[order],
        "-",
        label=fr"log-log fit, $q={slope:.3f}$",
    )
    axis.axhline(
        storage_floor,
        color="0.4",
        linestyle="--",
        label="6-decimal storage floor (bound)",
    )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel("relative final velocity error")
    axis.set_title("Taylor–Green temporal convergence at t=2")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(EVIDENCE / "order.png", dpi=180)
    plt.close(figure)


def plot_convergence(
    steps: np.ndarray,
    errors: np.ndarray,
    slope: float,
    predictions: dict[float, float],
    chosen: float | None,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    order = np.argsort(steps)
    axis.loglog(steps, errors, "o", markersize=7, label="measured")
    axis.loglog(
        steps[order],
        fitted_curve(steps, errors)[order],
        "-",
        label=fr"fit, $q={slope:.3f}$",
    )
    prediction_steps = np.asarray(sorted(predictions))
    prediction_errors = np.asarray([predictions[dt] for dt in prediction_steps])
    axis.loglog(
        prediction_steps,
        prediction_errors,
        "s--",
        label="Richardson prediction",
    )
    axis.axhline(5.0e-6, color="0.4", linestyle=":", label=r"target $5\times10^{-6}$")
    if chosen is None:
        axis.text(
            0.34,
            0.06,
            "no candidate meets the target",
            transform=axis.transAxes,
            color="tab:red",
        )
    else:
        axis.annotate(
            f"chosen dt={chosen:g}",
            (chosen, predictions[chosen]),
            xytext=(8, 9),
            textcoords="offset points",
        )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel("relative final vorticity error")
    axis.set_title("Random-flow temporal convergence at t=2")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(EVIDENCE / "convergence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cargo", "build", "--release", "--bins"], cwd=WEEK4, check=True
    )

    taylor_steps = np.asarray([0.4, 0.25, 0.2])
    taylor_runs = [
        run_case(
            RunSpec(
                case="taylor-green",
                n=8,
                nu=0.5,
                dt=float(dt),
                t_end=2.0,
                output_directory=ARTIFACTS
                / "order"
                / f"rk4-dt{dt_label(float(dt))}",
            )
        )
        for dt in taylor_steps
    ]
    exact_spec = RunSpec(
        case="taylor-green",
        n=8,
        nu=0.5,
        dt=0.0,
        t_end=2.0,
        output_directory=ARTIFACTS / "order",
    )
    exact_document = subprocess.run(
        field_command(exact_spec, exact_time=2.0),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (ARTIFACTS / "order" / "exact-t2.json").write_text(
        exact_document, encoding="utf-8"
    )
    exact_velocity = velocity(json.loads(exact_document))
    taylor_errors = np.asarray(
        [
            relative_error(
                velocity(frame_at_time(run.fields_path, 2.0)), exact_velocity
            )
            for run in taylor_runs
        ]
    )
    taylor_slope = fit_log_slope(taylor_steps, taylor_errors)
    storage_floor = (
        math.sqrt(exact_velocity.size) * 0.5e-6 / np.linalg.norm(exact_velocity)
    )
    plot_order(taylor_steps, taylor_errors, taylor_slope, storage_floor)

    random_steps = np.asarray([0.02, 0.0125, 0.01])
    random_runs: dict[float, StoredRun] = {}
    for dt in random_steps:
        reuse = [ARTIFACTS / "random"] if math.isclose(dt, 0.01) else []
        random_runs[float(dt)] = run_case(
            RunSpec(
                case="random",
                n=128,
                nu=0.004,
                dt=float(dt),
                t_end=2.0,
                output_directory=ARTIFACTS
                / "convergence"
                / f"rk4-dt{dt_label(float(dt))}",
            ),
            reuse,
        )
    reference = run_case(
        RunSpec(
            case="random",
            n=128,
            nu=0.004,
            dt=0.0025,
            t_end=2.0,
            output_directory=ARTIFACTS / "convergence" / "rk4-dt0.0025",
        )
    )
    reference_omega = vorticity(frame_at_time(reference.fields_path, 2.0))
    random_omega = {
        dt: vorticity(frame_at_time(run.fields_path, 2.0))
        for dt, run in random_runs.items()
    }
    random_errors = np.asarray(
        [relative_error(random_omega[float(dt)], reference_omega) for dt in random_steps]
    )
    random_slope = fit_log_slope(random_steps, random_errors)
    richardson = richardson_error(random_omega[0.02], random_omega[0.01])
    chosen, predictions = choose_time_step(
        base_dt=0.01,
        base_error=richardson,
        candidates=random_steps,
        tolerance=5.0e-6,
    )
    plot_convergence(random_steps, random_errors, random_slope, predictions, chosen)

    result = {
        "case": "random",
        "n": 128,
        "nu": 0.004,
        "t_end": 2.0,
        "method": "rk4",
        "seed": 2026,
        "k_band": [2, 6],
        "reference_dt": 0.0025,
        "runs": [
            {
                "dt": float(dt),
                "relative_omega_error": float(error),
                "fields": str(random_runs[float(dt)].fields_path.relative_to(WEEK4)),
                "reused": random_runs[float(dt)].reused_from is not None,
            }
            for dt, error in zip(random_steps, random_errors, strict=True)
        ],
        "slope": random_slope,
        "richardson": {
            "coarse_dt": 0.02,
            "fine_dt": 0.01,
            "order": 4,
            "estimated_fine_error": richardson,
            "tolerance": 5.0e-6,
            "predictions": {
                dt_label(dt): prediction for dt, prediction in predictions.items()
            },
            "chosen_dt": chosen,
            "chosen_measured_error": (
                None
                if chosen is None
                else float(
                    random_errors[np.where(np.isclose(random_steps, chosen))[0][0]]
                )
            ),
        },
    }
    (EVIDENCE / "convergence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print("Taylor–Green relative velocity errors:")
    for dt, error in zip(taylor_steps, taylor_errors, strict=True):
        print(f"  dt={dt:g}: {error:.12e}")
    print(f"  fitted slope: {taylor_slope:.6f}")
    print(f"  six-decimal storage floor bound: {storage_floor:.12e}")
    print("Random relative vorticity errors:")
    for dt, error in zip(random_steps, random_errors, strict=True):
        print(f"  dt={dt:g}: {error:.12e}")
    print(f"  fitted slope: {random_slope:.6f}")
    print(f"  Richardson estimate at dt=0.01: {richardson:.12e}")
    for dt, prediction in predictions.items():
        print(f"  predicted dt={dt:g}: {prediction:.12e}")
    if chosen is None:
        print("  chosen dt: none (all predictions are at least 5e-6)")
    else:
        print(f"  chosen dt: {chosen:g}")

    failures: list[str] = []
    if abs(taylor_slope - 4.0) > 0.15 * 4.0:
        failures.append(
            f"Taylor–Green slope {taylor_slope:.6f} is not within 15% of 4"
        )
    if not 3.7 <= random_slope <= 4.3:
        failures.append(f"random slope {random_slope:.6f} is outside [3.7, 4.3]")
    if predictions[0.02] < 5.0e-6:
        failures.append("Richardson prediction does not reject dt=0.02")
    if chosen is None:
        failures.append("no candidate has predicted error below 5e-6")
    else:
        chosen_index = np.where(np.isclose(random_steps, chosen))[0][0]
    if chosen is not None and random_errors[chosen_index] >= 5.0e-6:
        failures.append(
            f"chosen dt={chosen:g} measured error {random_errors[chosen_index]:.6e} "
            "does not meet 5e-6"
        )
    if failures:
        print(f"Part 4 acceptance failed: {'; '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

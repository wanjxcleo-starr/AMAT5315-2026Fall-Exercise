#!/usr/bin/env python3
"""Run and plot the current learning sheet's Week 4 Part 3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


WEEK4 = Path(__file__).resolve().parents[1]
ARTIFACTS = WEEK4 / "artifacts"
EVIDENCE = WEEK4 / "evidence"
BIN = WEEK4 / "target" / "release"


@dataclass
class EnergyTable:
    times: np.ndarray
    energy: np.ndarray
    enstrophy: np.ndarray
    nonfinite_time: float | None


@dataclass(frozen=True)
class RunSpec:
    name: str
    case: str
    method: str
    n: int
    nu: float
    dt: float
    requested_t_end: float
    actual_t_end: float
    every: float
    output_directory: Path
    table_path: Path
    perturbed: bool = False


@dataclass
class RunOutcome:
    spec: RunSpec
    table: EnergyTable
    return_code: int


def parse_energy_table(text: str) -> EnergyTable:
    times: list[float] = []
    energy: list[float] = []
    enstrophy: list[float] = []
    nonfinite_time = None
    for line_number, line in enumerate(text.splitlines()):
        if line_number == 0 or not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise ValueError(f"malformed energy-table line: {line!r}")
        time, current_energy, current_enstrophy = map(float, columns)
        if not math.isfinite(current_energy):
            nonfinite_time = time
            break
        times.append(time)
        energy.append(current_energy)
        enstrophy.append(current_enstrophy)
    return EnergyTable(
        times=np.asarray(times),
        energy=np.asarray(energy),
        enstrophy=np.asarray(enstrophy),
        nonfinite_time=nonfinite_time,
    )


def relative_vorticity_distance(
    unperturbed: np.ndarray, perturbed: np.ndarray
) -> float:
    denominator = float(np.linalg.norm(unperturbed.ravel()))
    if denominator == 0.0:
        raise ValueError("unperturbed vorticity norm is zero")
    return float(np.linalg.norm((unperturbed - perturbed).ravel()) / denominator)


def aligned_end_time(requested: float, dt: float) -> float:
    """Extend a horizon to the first whole fixed step at or beyond it."""
    steps = math.ceil(requested / dt - 64.0 * np.finfo(float).eps)
    return steps * dt


def field_command(case: str, n: int) -> list[str]:
    if case == "taylor-green":
        return [str(BIN / "field"), "taylor-green", "--n", str(n)]
    if case == "random":
        return [
            str(BIN / "field"),
            "random",
            "--n",
            str(n),
            "--seed",
            "2026",
            "--k-min",
            "2",
            "--k-max",
            "6",
        ]
    raise ValueError(f"unknown case: {case}")


def initial_field(case: str, n: int, perturbed: bool = False) -> str:
    generated = subprocess.run(
        field_command(case, n), check=True, capture_output=True, text=True
    ).stdout
    if not perturbed:
        return generated
    return subprocess.run(
        [str(BIN / "perturb-field")],
        input=generated,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def metadata_matches(spec: RunSpec) -> bool:
    metadata_path = spec.output_directory / "run.json"
    fields_path = spec.output_directory / "fields.jsonl"
    if not metadata_path.is_file() or not fields_path.is_file() or not spec.table_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata["case"] == spec.case
        and metadata["n"] == spec.n
        and metadata["method"] == spec.method
        and math.isclose(metadata["nu"], spec.nu)
        and math.isclose(metadata["dt"], spec.dt)
        and math.isclose(metadata["t_end"], spec.actual_t_end)
        and math.isclose(metadata["snapshot_every"], spec.every)
        and fields_path.stat().st_size > 0
    )


def run_case(spec: RunSpec) -> RunOutcome:
    if metadata_matches(spec):
        table = parse_energy_table(spec.table_path.read_text(encoding="utf-8"))
        return_code = 1 if table.nonfinite_time is not None else 0
        print(f"reuse {spec.name}")
        return RunOutcome(spec, table, return_code)

    spec.output_directory.mkdir(parents=True, exist_ok=True)
    spec.table_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"run {spec.name}: dt={spec.dt:g}, requested t_end={spec.requested_t_end:g}, "
        f"fixed-step t_end={spec.actual_t_end:g}"
    )
    completed = subprocess.run(
        [
            str(BIN / "fluid"),
            "--method",
            spec.method,
            "--nu",
            str(spec.nu),
            "--dt",
            str(spec.dt),
            "--t-end",
            str(spec.actual_t_end),
            "--every",
            str(spec.every),
            "--out",
            str(spec.output_directory),
        ],
        input=initial_field(spec.case, spec.n, spec.perturbed),
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"{spec.name} failed with exit {completed.returncode}: {completed.stderr}"
        )
    spec.table_path.write_text(completed.stdout, encoding="utf-8")
    return RunOutcome(spec, parse_energy_table(completed.stdout), completed.returncode)


def make_spec(
    name: str,
    case: str,
    method: str,
    n: int,
    nu: float,
    dt: float,
    requested_t_end: float,
    every: float,
    group: str,
    perturbed: bool = False,
) -> RunSpec:
    actual_t_end = aligned_end_time(requested_t_end, dt)
    return RunSpec(
        name=name,
        case=case,
        method=method,
        n=n,
        nu=nu,
        dt=dt,
        requested_t_end=requested_t_end,
        actual_t_end=actual_t_end,
        every=every,
        output_directory=ARTIFACTS / group / name,
        table_path=ARTIFACTS / group / f"{name}.tsv",
        perturbed=perturbed,
    )


def run_stability_cases() -> tuple[RunOutcome, RunOutcome, RunOutcome, RunOutcome, RunOutcome, RunOutcome]:
    unstable_taylor = run_case(
        make_spec(
            "taylor-green",
            "taylor-green",
            "rk4",
            64,
            0.1,
            0.04,
            4.0,
            0.1,
            "unstable",
        )
    )
    taylor_stable = run_case(
        make_spec(
            "taylor-green-rk4-dt0.032",
            "taylor-green",
            "rk4",
            64,
            0.1,
            0.032,
            8.0,
            0.5,
            "scan",
        )
    )
    taylor_unstable = run_case(
        make_spec(
            "taylor-green-rk4-dt0.033",
            "taylor-green",
            "rk4",
            64,
            0.1,
            0.033,
            8.0,
            0.5,
            "scan",
        )
    )

    random_outcomes = [
        run_case(
            make_spec(
                f"random-rk4-dt{dt:.3f}",
                "random",
                "rk4",
                128,
                0.004,
                dt,
                10.0,
                0.5,
                "scan",
            )
        )
        for dt in (0.038, 0.040)
    ]
    next_lower = 0.036
    while not any(outcome.return_code == 0 for outcome in random_outcomes):
        random_outcomes.append(
            run_case(
                make_spec(
                    f"random-rk4-dt{next_lower:.3f}",
                    "random",
                    "rk4",
                    128,
                    0.004,
                    next_lower,
                    10.0,
                    0.5,
                    "scan",
                )
            )
        )
        next_lower -= 0.002
    next_upper = 0.042
    while not any(outcome.return_code == 1 for outcome in random_outcomes):
        random_outcomes.append(
            run_case(
                make_spec(
                    f"random-rk4-dt{next_upper:.3f}",
                    "random",
                    "rk4",
                    128,
                    0.004,
                    next_upper,
                    10.0,
                    0.5,
                    "scan",
                )
            )
        )
        next_upper += 0.002
    random_stable = max(
        (outcome for outcome in random_outcomes if outcome.return_code == 0),
        key=lambda outcome: outcome.spec.dt,
    )
    random_unstable = min(
        (outcome for outcome in random_outcomes if outcome.return_code == 1),
        key=lambda outcome: outcome.spec.dt,
    )
    if random_stable.spec.dt >= random_unstable.spec.dt:
        raise RuntimeError("random RK4 runs do not form an ordered stability bracket")

    random_euler = run_case(
        make_spec(
            "random-euler-dt0.010",
            "random",
            "euler",
            128,
            0.004,
            0.01,
            10.0,
            0.5,
            "scan",
        )
    )
    return (
        unstable_taylor,
        taylor_stable,
        taylor_unstable,
        random_stable,
        random_unstable,
        random_euler,
    )


def run_sensitivity_cases() -> dict[str, RunOutcome]:
    outcomes = {}
    for case, n, nu in [("taylor-green", 64, 0.1), ("random", 128, 0.004)]:
        for perturbed in (False, True):
            suffix = "perturbed" if perturbed else "original"
            name = f"{case}-{suffix}"
            outcomes[name] = run_case(
                make_spec(
                    name,
                    case,
                    "rk4",
                    n,
                    nu,
                    0.01,
                    20.0,
                    0.5,
                    "sensitivity",
                    perturbed=perturbed,
                )
            )
    return outcomes


def initial_random_speed() -> float:
    document = json.loads(initial_field("random", 128))
    u = np.asarray(document["u"])
    v = np.asarray(document["v"])
    return float(np.max(np.hypot(u, v)))


def validate_stability(
    unstable_taylor: RunOutcome,
    taylor_stable: RunOutcome,
    taylor_unstable: RunOutcome,
    random_stable: RunOutcome,
    random_unstable: RunOutcome,
    random_euler: RunOutcome,
    maximum_speed: float,
) -> tuple[float, float]:
    predicted_taylor = 2.785 / (0.1 * 2.0 * math.floor(64 / 3) ** 2)
    predicted_random = 2.83 / (
        maximum_speed * math.sqrt(2.0) * math.floor(128 / 3)
    )
    if unstable_taylor.table.nonfinite_time is None or unstable_taylor.table.nonfinite_time >= 4.0:
        raise RuntimeError("Taylor-Green dt=0.04 did not become non-finite before t=4")
    if taylor_stable.return_code != 0 or taylor_unstable.return_code != 1:
        raise RuntimeError("Taylor-Green dt=0.032 and 0.033 did not bracket the limit")
    if max(
        abs(taylor_stable.spec.dt / predicted_taylor - 1.0),
        abs(taylor_unstable.spec.dt / predicted_taylor - 1.0),
    ) > 0.05:
        raise RuntimeError("Taylor-Green measured bracket is not within 5% of prediction")
    if not (
        random_stable.spec.dt >= predicted_random
        and random_unstable.spec.dt <= 3.0 * predicted_random
    ):
        raise RuntimeError("random measured bracket is not between one and three times its bound")
    if random_euler.table.nonfinite_time is None or random_euler.table.nonfinite_time >= 2.0:
        raise RuntimeError("random Euler dt=0.01 did not fail within two time units")
    return predicted_taylor, predicted_random


def plot_energy_curve(axis, outcome: RunOutcome, label: str, **kwargs) -> None:
    axis.semilogy(outcome.table.times, outcome.table.energy, label=label, **kwargs)
    if outcome.table.nonfinite_time is not None:
        axis.axvline(
            outcome.table.nonfinite_time,
            color=kwargs.get("color", None),
            linestyle=":",
            alpha=0.7,
        )
        if outcome.table.energy.size:
            axis.plot(
                outcome.table.nonfinite_time,
                outcome.table.energy[-1],
                "x",
                color=kwargs.get("color", None),
            )


def write_blowup_figure(
    taylor_stable: RunOutcome,
    taylor_unstable: RunOutcome,
    random_stable: RunOutcome,
    random_unstable: RunOutcome,
    random_euler: RunOutcome,
    predicted_taylor: float,
    predicted_random: float,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    plot_energy_curve(axes[0], taylor_stable, "RK4, dt=0.032", color="#1f77b4")
    plot_energy_curve(axes[0], taylor_unstable, "RK4, dt=0.033", color="#d62728")
    exact_time = np.linspace(0.0, 8.0, 300)
    axes[0].semilogy(exact_time, 0.25 * np.exp(-0.4 * exact_time), "k--", label="exact")
    axes[0].set_title(
        f"Taylor–Green, predicted critical dt={predicted_taylor:.4f}"
    )
    axes[0].set_xlim(0.0, 8.0)

    plot_energy_curve(
        axes[1],
        random_stable,
        f"RK4, dt={random_stable.spec.dt:.3f}",
        color="#1f77b4",
    )
    plot_energy_curve(
        axes[1],
        random_unstable,
        f"RK4, dt={random_unstable.spec.dt:.3f}",
        color="#d62728",
    )
    plot_energy_curve(
        axes[1], random_euler, "Euler, dt=0.010", color="#2ca02c", linestyle="--"
    )
    axes[1].set_title(f"random, advective bound dt={predicted_random:.4f}")
    axes[1].set_xlim(0.0, 10.0)
    for axis in axes:
        axis.set_xlabel("time t")
        axis.set_ylabel("energy E(t)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    destination = EVIDENCE / "blowup.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def paired_distances(original_path: Path, perturbed_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    distances: list[float] = []
    with original_path.open(encoding="utf-8") as original_stream, perturbed_path.open(
        encoding="utf-8"
    ) as perturbed_stream:
        for original_line, perturbed_line in zip(
            original_stream, perturbed_stream, strict=True
        ):
            original = json.loads(original_line)
            perturbed = json.loads(perturbed_line)
            if original["step"] != perturbed["step"]:
                raise ValueError("sensitivity snapshots have mismatched steps")
            times.append(float(original["t"]))
            distances.append(
                relative_vorticity_distance(
                    np.asarray(original["omega"]), np.asarray(perturbed["omega"])
                )
            )
    return np.asarray(times), np.asarray(distances)


def write_sensitivity_figure(outcomes: dict[str, RunOutcome]) -> tuple[Path, dict[str, np.ndarray]]:
    series = {}
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for case, label, color in [
        ("taylor-green", "Taylor–Green", "#1f77b4"),
        ("random", "random", "#d62728"),
    ]:
        times, distances = paired_distances(
            outcomes[f"{case}-original"].spec.output_directory / "fields.jsonl",
            outcomes[f"{case}-perturbed"].spec.output_directory / "fields.jsonl",
        )
        series[case] = distances
        positive = np.where(distances > 0.0, distances, np.nan)
        axis.semilogy(times, positive, "o-", markersize=3, label=label, color=color)
        zero_indices = np.flatnonzero(distances == 0.0)
        if zero_indices.size:
            axis.annotate(
                "equal at six-decimal storage",
                xy=(times[zero_indices[0]], np.nanmin(positive)),
                xytext=(8, 14),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": color},
            )
    axis.set_xlabel("time t")
    axis.set_ylabel("||omega_original - omega_perturbed|| / ||omega_original||")
    axis.set_xlim(0.0, 20.0)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    axis.set_title("Sensitivity to the prescribed initial vorticity ripple")
    destination = EVIDENCE / "sensitivity.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination, series


def load_selected_random_frames() -> list[dict]:
    wanted = {0.0, 2.0, 5.0, 10.0}
    frames = []
    with (ARTIFACTS / "random" / "fields.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if float(frame["t"]) in wanted:
                frames.append(frame)
    if [float(frame["t"]) for frame in frames] != [0.0, 2.0, 5.0, 10.0]:
        raise ValueError("baseline random recording lacks t=0,2,5,10 frames")
    return frames


def frame_energy_and_enstrophy(frame: dict) -> tuple[float, float]:
    u = np.asarray(frame["u"])
    v = np.asarray(frame["v"])
    omega = np.asarray(frame["omega"])
    return float(0.5 * np.mean(u * u + v * v)), float(0.5 * np.mean(omega * omega))


def write_random_figure() -> Path:
    frames = load_selected_random_frames()
    n = 128
    vorticity_limit = float(np.max(np.abs(np.asarray(frames[0]["omega"]))))
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), constrained_layout=True)
    image = None
    for axis, frame in zip(axes, frames, strict=True):
        omega = np.asarray(frame["omega"]).reshape((n, n))
        energy, enstrophy = frame_energy_and_enstrophy(frame)
        image = axis.imshow(
            omega,
            origin="lower",
            extent=(0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
            cmap="coolwarm",
            vmin=-vorticity_limit,
            vmax=vorticity_limit,
            interpolation="bilinear",
        )
        axis.set_title(
            f"t={frame['t']:g}\nE={energy:.3f}, Z={enstrophy:.3f}", fontsize=10
        )
        axis.set_xticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
        axis.set_yticks([0.0, math.pi, 2.0 * math.pi], ["0", "π", "2π"])
        axis.set_xlabel("x")
        axis.set_aspect("equal")
    axes[0].set_ylabel("y")
    figure.colorbar(image, ax=axes, label="vorticity omega", shrink=0.82)
    figure.suptitle(
        f"Random flow decay; shared scale +/-{vorticity_limit:.2f}", fontsize=13
    )
    destination = EVIDENCE / "random.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main() -> None:
    subprocess.run(
        ["cargo", "build", "--quiet", "--release", "--bins"], cwd=WEEK4, check=True
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if not (ARTIFACTS / "random.tsv").exists():
        shutil.copyfile(EVIDENCE / "decay.txt", ARTIFACTS / "random.tsv")

    maximum_speed = initial_random_speed()
    stability = run_stability_cases()
    predicted_taylor, predicted_random = validate_stability(*stability, maximum_speed)
    sensitivity = run_sensitivity_cases()
    if any(outcome.return_code != 0 for outcome in sensitivity.values()):
        raise RuntimeError("a sensitivity run became non-finite")

    blowup_path = write_blowup_figure(
        stability[1],
        stability[2],
        stability[3],
        stability[4],
        stability[5],
        predicted_taylor,
        predicted_random,
    )
    sensitivity_path, sensitivity_series = write_sensitivity_figure(sensitivity)
    random_path = write_random_figure()
    random_growth = sensitivity_series["random"][-1] / sensitivity_series["random"][0]
    taylor_final = sensitivity_series["taylor-green"][-1]
    if random_growth <= 10.0:
        raise RuntimeError(f"random perturbation grew only {random_growth:.3f}-fold")
    if taylor_final >= sensitivity_series["taylor-green"][0]:
        raise RuntimeError("Taylor-Green perturbation did not decay toward its storage floor")

    print(f"random initial maximum speed: {maximum_speed:.12f}")
    print(f"Taylor-Green diffusive prediction: {predicted_taylor:.8f}")
    print(
        f"Taylor-Green measured bracket: {stability[1].spec.dt:.3f} stable, "
        f"{stability[2].spec.dt:.3f} unstable at "
        f"t={stability[2].table.nonfinite_time:g}"
    )
    print(f"random advective bound: {predicted_random:.8f}")
    print(
        f"random measured bracket: {stability[3].spec.dt:.3f} stable, "
        f"{stability[4].spec.dt:.3f} unstable at "
        f"t={stability[4].table.nonfinite_time:g}"
    )
    print(f"Euler non-finite time: {stability[5].table.nonfinite_time:g}")
    print(f"random sensitivity growth factor at t=20: {random_growth:.6f}")
    print(f"Taylor-Green final stored sensitivity: {taylor_final:.12e}")
    for path in (blowup_path, sensitivity_path, random_path):
        print(f"wrote {path.relative_to(WEEK4)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the authorized dt=0.008 supplement to Week 4 Part 4."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from part4 import (  # noqa: E402
    ARTIFACTS,
    EVIDENCE,
    WEEK4,
    RunSpec,
    frame_at_time,
    relative_error,
    richardson_error,
    run_case,
    vorticity,
)


SUPPLEMENT_DT = 0.008
TOLERANCE = 5.0e-6


def validate_prescribed_result(document: dict[str, object]) -> None:
    """Protect the original three-candidate result from the supplement."""
    runs = document.get("runs")
    if not isinstance(runs, list) or [run.get("dt") for run in runs] != [
        0.02,
        0.0125,
        0.01,
    ]:
        raise RuntimeError("prescribed candidates must remain 0.02, 0.0125, 0.01")
    richardson = document.get("richardson")
    if not isinstance(richardson, dict) or richardson.get("chosen_dt") is not None:
        raise RuntimeError("prescribed chosen_dt must remain null")


def supplemental_record(
    dt: float,
    predicted_error: float,
    measured_error: float,
    tolerance: float,
    fields: str,
) -> dict[str, object]:
    """Describe a supplement without changing the prescribed candidate choice."""
    predicted_passes = predicted_error < tolerance
    measured_passes = measured_error < tolerance
    return {
        "label": "原候选全部未达标后的补充实验",
        "dt": dt,
        "predicted_error": predicted_error,
        "measured_error": measured_error,
        "tolerance": tolerance,
        "predicted_below_tolerance": predicted_passes,
        "measured_below_tolerance": measured_passes,
        "both_below_tolerance": predicted_passes and measured_passes,
        "fields": fields,
        "not_a_global_maximum_claim": True,
    }


def frame_arrays_match(
    first: dict[str, object], second: dict[str, object]
) -> bool:
    return all(
        np.array_equal(np.asarray(first[name]), np.asarray(second[name]))
        for name in ("u", "v", "omega")
    )


def fields_path(document: dict[str, object], dt: float) -> Path:
    runs = document["runs"]
    if not isinstance(runs, list):
        raise RuntimeError("convergence runs are malformed")
    for run in runs:
        if isinstance(run, dict) and math.isclose(float(run["dt"]), dt):
            return WEEK4 / str(run["fields"])
    raise RuntimeError(f"missing prescribed dt={dt:g} fields")


def plot_supplement(
    document: dict[str, object], predicted_error: float, measured_error: float
) -> None:
    runs = document["runs"]
    if not isinstance(runs, list):
        raise RuntimeError("convergence runs are malformed")
    candidate_steps = np.asarray([float(run["dt"]) for run in runs])
    candidate_errors = np.asarray(
        [float(run["relative_omega_error"]) for run in runs]
    )

    figure, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    axis.loglog(
        candidate_steps,
        candidate_errors,
        "o",
        color="tab:blue",
        label="prescribed candidates (measured)",
    )
    axis.loglog(
        SUPPLEMENT_DT,
        measured_error,
        "*",
        color="tab:red",
        markersize=12,
        label="supplement dt=0.008 (measured)",
    )
    axis.loglog(
        SUPPLEMENT_DT,
        predicted_error,
        "D",
        markerfacecolor="none",
        markeredgecolor="tab:red",
        markersize=8,
        label="supplement dt=0.008 (predicted)",
    )
    axis.axhline(
        TOLERANCE,
        color="0.4",
        linestyle=":",
        label=r"target $5\times10^{-6}$",
    )
    axis.annotate(
        "supplement after all prescribed\ncandidates failed",
        (SUPPLEMENT_DT, measured_error),
        xytext=(18, 20),
        textcoords="offset points",
        color="tab:red",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    axis.text(
        0.98,
        0.04,
        "original chosen_dt remains null",
        ha="right",
        transform=axis.transAxes,
        color="0.3",
    )
    axis.set_xlabel(r"time step $\Delta t$")
    axis.set_ylabel("relative final vorticity error")
    axis.set_title("Part 4 supplemental time-step check at t=2")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.savefig(EVIDENCE / "convergence-supplemental.png", dpi=180)
    plt.close(figure)


def main() -> None:
    convergence_path = EVIDENCE / "convergence.json"
    document = json.loads(convergence_path.read_text(encoding="utf-8"))
    validate_prescribed_result(document)

    coarse = vorticity(frame_at_time(fields_path(document, 0.02), 2.0))
    fine = vorticity(frame_at_time(fields_path(document, 0.01), 2.0))
    richardson = richardson_error(coarse, fine)
    recorded_richardson = float(
        document["richardson"]["estimated_fine_error"]  # type: ignore[index]
    )
    if not math.isclose(richardson, recorded_richardson, rel_tol=1.0e-12):
        raise RuntimeError("retained fields do not reproduce the recorded Richardson estimate")
    predicted_error = richardson * (SUPPLEMENT_DT / 0.01) ** 4

    subprocess.run(
        ["cargo", "build", "--release", "--bins"], cwd=WEEK4, check=True
    )
    supplement = run_case(
        RunSpec(
            case="random",
            n=128,
            nu=0.004,
            dt=SUPPLEMENT_DT,
            t_end=2.0,
            output_directory=ARTIFACTS / "convergence" / "rk4-dt0.008",
        )
    )
    reference_path = ARTIFACTS / "convergence" / "rk4-dt0.0025" / "fields.jsonl"
    supplemental_initial = frame_at_time(supplement.fields_path, 0.0)
    reference_initial = frame_at_time(reference_path, 0.0)
    if not frame_arrays_match(supplemental_initial, reference_initial):
        raise RuntimeError("supplement and reference do not share the same initial field")

    supplemental_omega = vorticity(frame_at_time(supplement.fields_path, 2.0))
    reference_omega = vorticity(frame_at_time(reference_path, 2.0))
    measured_error = relative_error(supplemental_omega, reference_omega)
    record = supplemental_record(
        dt=SUPPLEMENT_DT,
        predicted_error=predicted_error,
        measured_error=measured_error,
        tolerance=TOLERANCE,
        fields=str(supplement.fields_path.relative_to(WEEK4)),
    )
    record.update(
        {
            "n": 128,
            "nu": 0.004,
            "t_end": 2.0,
            "method": "rk4",
            "seed": 2026,
            "k_band": [2, 6],
            "reference_dt": 0.0025,
            "prediction_base_dt": 0.01,
            "prediction_order": 4,
        }
    )
    document["supplemental_experiment"] = record
    convergence_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_supplement(document, predicted_error, measured_error)

    print("原候选全部未达标后的补充实验")
    print(f"  dt: {SUPPLEMENT_DT:g}")
    print(f"  predicted relative error: {predicted_error:.12e}")
    print(f"  measured relative error:  {measured_error:.12e}")
    print(f"  predicted < 5e-6: {record['predicted_below_tolerance']}")
    print(f"  measured < 5e-6:  {record['measured_below_tolerance']}")
    print(f"  both < 5e-6:      {record['both_below_tolerance']}")
    print("  original chosen_dt remains null")
    if not record["both_below_tolerance"]:
        print("supplement failed; no further scan is authorized", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Recompute Week 2 fluid diagnostics from recorded positions and velocities."""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path


CUTOFF = 2.5
DRIFT_LIMIT = 2e-3
TARGET_TEMPERATURE = 0.5
TEMPERATURE_ERROR_LIMIT = 0.05
CHI2_PER_DOF_LIMIT = 2.0
ENERGY_ABS_TOLERANCE = 1e-8
ENERGY_REL_TOLERANCE = 1e-10


def pair_potential(positions: list[list[float]], box: list[float]) -> float:
    """Sum the shifted Lennard-Jones potential over minimum-image pairs."""
    shift = 4.0 * (CUTOFF**-12 - CUTOFF**-6)
    terms = []
    for i, (xi, yi) in enumerate(positions):
        for xj, yj in positions[i + 1 :]:
            dx = xj - xi
            dy = yj - yi
            dx -= box[0] * math.floor(dx / box[0] + 0.5)
            dy -= box[1] * math.floor(dy / box[1] + 0.5)
            r2 = dx * dx + dy * dy
            if r2 == 0.0:
                raise ValueError(f"overlapping atoms in saved positions: {i}")
            if r2 < CUTOFF * CUTOFF:
                terms.append(4.0 * (r2**-6 - r2**-3) - shift)
    return math.fsum(terms)


def checked_energy(name: str, stored: float, recomputed: float, frame: int) -> float:
    if not math.isfinite(stored) or not math.isclose(
        stored,
        recomputed,
        rel_tol=ENERGY_REL_TOLERANCE,
        abs_tol=ENERGY_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"frame {frame} {name} mismatch: stored={stored}, recomputed={recomputed}"
        )
    return abs(stored - recomputed)


def diagnostics(run_folder: Path) -> tuple[float, float, float, float, float, int]:
    metadata = json.loads((run_folder / "run.json").read_text())
    n = metadata["n"]
    box = metadata["box"]
    dt = metadata["dt"]
    sample_every = metadata["sample_every"]
    steps = metadata["steps"]
    if not isinstance(n, int) or n < 2:
        raise ValueError("run.json: n must be at least two")
    if not isinstance(box, list) or len(box) != 2 or any(
        not math.isfinite(length) or length <= 2 * CUTOFF for length in box
    ):
        raise ValueError("run.json: box must have two finite lengths above twice the cutoff")
    if not isinstance(sample_every, int) or sample_every <= 0:
        raise ValueError("run.json: sample_every must be positive")
    if not isinstance(steps, int) or steps <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("run.json: steps and dt must be positive")

    energies = []
    squared_speeds = []
    max_potential_difference = 0.0
    max_kinetic_difference = 0.0
    with (run_folder / "traj.jsonl").open() as trajectory:
        for frame_number, line in enumerate(trajectory, start=1):
            frame = json.loads(line)
            step = frame_number * sample_every
            if frame["step"] != step or not math.isclose(
                frame["t"], step * dt, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"frame {frame_number}: incorrect production step or time")
            positions = frame["pos"]
            velocities = frame["vel"]
            if len(positions) != n or len(velocities) != n:
                raise ValueError(f"frame {frame_number}: expected {n} positions and velocities")
            for point, velocity in zip(positions, velocities):
                if len(point) != 2 or len(velocity) != 2:
                    raise ValueError(f"frame {frame_number}: atoms must have two components")
                if any(not math.isfinite(value) for value in point + velocity):
                    raise ValueError(f"frame {frame_number}: non-finite position or velocity")
                if any(not 0 <= point[axis] < box[axis] for axis in range(2)):
                    raise ValueError(f"frame {frame_number}: position outside periodic box")
                squared_speeds.append(velocity[0] ** 2 + velocity[1] ** 2)

            potential = pair_potential(positions, box)
            kinetic = math.fsum(
                (vx * vx + vy * vy) / 2.0 for vx, vy in velocities
            )
            max_potential_difference = max(
                max_potential_difference,
                checked_energy("E_pot", frame["E_pot"], potential, frame_number),
            )
            max_kinetic_difference = max(
                max_kinetic_difference,
                checked_energy("E_kin", frame["E_kin"], kinetic, frame_number),
            )
            energies.append(potential + kinetic)

    if len(energies) != steps // sample_every or not energies:
        raise ValueError("traj.jsonl: missing or unexpected saved frames")
    if energies[0] == 0.0:
        raise ValueError("first saved total energy is zero; secular drift is undefined")
    k = max(1, len(energies) // 10)
    drift = abs(
        math.fsum(energies[-k:]) / k - math.fsum(energies[:k]) / k
    ) / abs(energies[0])
    t_speed = math.fsum(squared_speeds) / (2.0 * len(squared_speeds))
    if not math.isfinite(t_speed) or t_speed <= 0:
        raise ValueError("saved velocities have zero or non-finite speed temperature")
    edges = [
        math.sqrt(-2.0 * t_speed * math.log1p(-index / 24.0))
        for index in range(1, 24)
    ]
    counts = [0] * 24
    for speed2 in squared_speeds:
        counts[bisect.bisect_right(edges, math.sqrt(speed2))] += 1
    expected = len(squared_speeds) / 24.0
    chi2_per_22 = math.fsum(
        (observed - expected) ** 2 / expected for observed in counts
    ) / 22.0
    return (
        drift,
        t_speed,
        chi2_per_22,
        max_potential_difference,
        max_kinetic_difference,
        k,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    arguments = parser.parse_args()
    try:
        drift, t_speed, chi2_per_22, pot_difference, kin_difference, k = diagnostics(
            arguments.run_folder
        )
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1

    temperature_error = abs(t_speed - TARGET_TEMPERATURE)
    checks = [
        ("secular_drift", drift, DRIFT_LIMIT),
        ("abs(T_speed - 0.5)", temperature_error, TEMPERATURE_ERROR_LIMIT),
        ("chi2/22", chi2_per_22, CHI2_PER_DOF_LIMIT),
    ]
    for name, value, limit in checks:
        status = "PASS" if value < limit else "FAIL"
        print(f"{name} = {value:.9g} ; limit < {limit:.9g} ; {status}")
    print(f"T_speed = {t_speed:.9g} ; first/last window k = {k}")
    print(
        "stored-energy cross-check: "
        f"max |E_pot difference| = {pot_difference:.3g}, "
        f"max |E_kin difference| = {kin_difference:.3g} "
        f"(abs_tol={ENERGY_ABS_TOLERANCE:g}, rel_tol={ENERGY_REL_TOLERANCE:g})"
    )
    return 0 if all(value < limit for _, value, limit in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

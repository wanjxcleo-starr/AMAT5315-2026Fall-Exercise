"""Behavior tests for independent Week 2 trajectory diagnostics."""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "scripts" / "check.py"
SPEEDS = [
    0.145098, 0.254044, 0.331664, 0.397025, 0.455675, 0.510179,
    0.562008, 0.612122, 0.661221, 0.709863, 0.758528, 0.807667,
    0.857734, 0.909219, 0.962685, 1.018817, 1.078495, 1.142914,
    1.213798, 1.293822, 1.387549, 1.503916, 1.665109, 1.967537,
]
POTENTIAL = -0.8746483964470763  # one minimum-image pair at separation 1.2
KINETIC = sum(speed * speed for speed in SPEEDS) / 2


def write_fixture(folder: Path, *, frame_count: int = 2, potential_offsets=None,
                  kinetic_offsets=None) -> None:
    metadata = {
        "n": 24, "rho": 24 / 1_000_000, "box": [1000.0, 1000.0],
        "dt": 1.0, "temperature": 0.5, "eq_steps": 0,
        "steps": frame_count, "sample_every": 1, "seed": 0,
        "integrator": "velocity-verlet",
    }
    (folder / "run.json").write_text(json.dumps(metadata) + "\n")
    positions = [[0.0, 0.0], [998.8, 0.0]] + [[3.0 * i, 0.0] for i in range(1, 23)]
    velocities = [[speed, 0.0] for speed in SPEEDS]
    frames = []
    for index in range(frame_count):
        frames.append(json.dumps({
            "step": index + 1, "t": float(index + 1),
            "pos": positions, "vel": velocities,
            "E_pot": POTENTIAL + (potential_offsets or {}).get(index, 0.0),
            "E_kin": KINETIC + (kinetic_offsets or {}).get(index, 0.0),
        }))
    (folder / "traj.jsonl").write_text("\n".join(frames) + "\n")


def run_check(folder: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(folder)],
        text=True, capture_output=True,
    )


def measured(text: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)} = ([0-9.eE+-]+) ; limit < ", text, re.M)
    if match is None:
        raise AssertionError(f"missing {label} in {text!r}")
    return float(match.group(1))


class CheckTests(unittest.TestCase):
    def test_periodic_pair_and_rayleigh_quantiles_pass_from_saved_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_fixture(folder)
            result = run_check(folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(measured(result.stdout, "secular_drift"), 0.0)
            self.assertAlmostEqual(
                measured(result.stdout, "abs(T_speed - 0.5)"),
                0.0071840165, places=7,
            )
            self.assertEqual(measured(result.stdout, "chi2/22"), 0.0)
            self.assertIn("limit < 0.002", result.stdout)
            self.assertIn("limit < 0.05", result.stdout)
            self.assertIn("limit < 2", result.stdout)

    def test_stored_potential_is_cross_checked_but_not_used_for_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_fixture(folder, potential_offsets={1: 5e-9})
            result = run_check(folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(measured(result.stdout, "secular_drift"), 0.0)
            write_fixture(folder, potential_offsets={1: 0.001})
            mismatch = run_check(folder)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("E_pot", mismatch.stderr)

    def test_stored_kinetic_energy_is_cross_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_fixture(folder, kinetic_offsets={0: 0.001})
            result = run_check(folder)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("E_kin", result.stderr)


if __name__ == "__main__":
    unittest.main()

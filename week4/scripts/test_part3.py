#!/usr/bin/env python3
"""Tests for the Week 4 Part 3 experiment helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part3 import parse_energy_table, relative_vorticity_distance  # noqa: E402


class Part3Tests(unittest.TestCase):
    def test_energy_table_keeps_finite_curve_and_nonfinite_stop_time(self) -> None:
        table = parse_energy_table(
            "t\tE\tZ\n0\t0.5\t1.0\n0.5\t0.4\t0.8\n0.7\tinf\tinf\n"
        )

        np.testing.assert_allclose(table.times, [0.0, 0.5])
        np.testing.assert_allclose(table.energy, [0.5, 0.4])
        self.assertEqual(table.nonfinite_time, 0.7)

    def test_relative_vorticity_distance_uses_the_unperturbed_norm(self) -> None:
        unperturbed = np.array([3.0, 4.0])
        perturbed = np.array([0.0, 4.0])

        self.assertAlmostEqual(
            relative_vorticity_distance(unperturbed, perturbed), 0.6
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for the Week 4 Part 2 comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part2 import relative_velocity_error, validate_derivative_study  # noqa: E402


class Part2PlotTests(unittest.TestCase):
    def test_relative_velocity_error_uses_both_components(self) -> None:
        recorded = {"u": [3.0, 0.0], "v": [0.0, 0.0]}
        exact = {"u": [3.0, 4.0], "v": [0.0, 0.0]}

        self.assertAlmostEqual(relative_velocity_error(recorded, exact), 0.8)

    def test_derivative_validation_rejects_a_failed_fourier_threshold(self) -> None:
        document = {
            "derivatives": [
                {
                    "derivative": "dx",
                    "finite_difference_n32": 0.16,
                    "finite_difference_n64": 0.04,
                    "fourier_n32": 2.0e-10,
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "Fourier"):
            validate_derivative_study(document)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for the Week 4 Part 4 convergence helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part4 import (  # noqa: E402
    choose_time_step,
    fit_log_slope,
    relative_error,
    richardson_error,
)


class Part4Tests(unittest.TestCase):
    def test_relative_error_uses_the_reference_norm(self) -> None:
        actual = np.array([0.0, 4.0])
        reference = np.array([3.0, 4.0])

        self.assertAlmostEqual(relative_error(actual, reference), 0.6)

    def test_log_fit_recovers_fourth_order(self) -> None:
        steps = np.array([0.4, 0.25, 0.2])
        errors = 0.125 * steps**4

        self.assertAlmostEqual(fit_log_slope(steps, errors), 4.0)

    def test_richardson_error_uses_fine_solution_norm(self) -> None:
        coarse = np.array([1.15, 0.0])
        fine = np.array([1.0, 0.0])

        self.assertAlmostEqual(richardson_error(coarse, fine), 0.01)

    def test_step_choice_is_largest_prediction_below_tolerance(self) -> None:
        chosen, predictions = choose_time_step(
            base_dt=0.01,
            base_error=1.0e-6,
            candidates=[0.02, 0.0125, 0.01],
            tolerance=5.0e-6,
        )

        self.assertEqual(chosen, 0.0125)
        self.assertAlmostEqual(predictions[0.02], 16.0e-6)
        self.assertAlmostEqual(predictions[0.0125], 2.44140625e-6)

    def test_step_choice_reports_when_no_candidate_meets_tolerance(self) -> None:
        chosen, predictions = choose_time_step(
            base_dt=0.01,
            base_error=6.0e-6,
            candidates=[0.02, 0.0125, 0.01],
            tolerance=5.0e-6,
        )

        self.assertIsNone(chosen)
        self.assertAlmostEqual(predictions[0.01], 6.0e-6)


if __name__ == "__main__":
    unittest.main()

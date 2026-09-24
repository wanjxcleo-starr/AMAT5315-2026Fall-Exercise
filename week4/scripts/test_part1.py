#!/usr/bin/env python3
"""Tests for the Week 4 Part 1 plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part1 import stability_function  # noqa: E402


class Part1PlotTests(unittest.TestCase):
    def test_stability_functions_have_the_learning_sheet_axis_crossings(self) -> None:
        self.assertAlmostEqual(abs(stability_function("euler", -2.0)), 1.0, places=12)
        self.assertAlmostEqual(abs(stability_function("midpoint", -2.0)), 1.0, places=12)
        self.assertLess(abs(abs(stability_function("rk4", -2.785)) - 1.0), 1.0e-3)
        self.assertLess(abs(abs(stability_function("rk4", 2.83j)) - 1.0), 5.0e-3)


if __name__ == "__main__":
    unittest.main()

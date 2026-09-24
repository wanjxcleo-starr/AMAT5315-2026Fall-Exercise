#!/usr/bin/env python3
"""Tests for the bounded Week 4 Part 4 supplemental experiment."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part4_supplement import (  # noqa: E402
    supplemental_record,
    validate_prescribed_result,
)


class Part4SupplementTests(unittest.TestCase):
    def test_prescribed_result_rejects_a_supplement_mixed_into_candidates(self) -> None:
        document = {
            "runs": [
                {"dt": 0.02},
                {"dt": 0.0125},
                {"dt": 0.01},
                {"dt": 0.008},
            ],
            "richardson": {"chosen_dt": None},
        }

        with self.assertRaisesRegex(RuntimeError, "prescribed candidates"):
            validate_prescribed_result(document)

    def test_prescribed_result_requires_the_null_choice_to_remain(self) -> None:
        document = {
            "runs": [{"dt": 0.02}, {"dt": 0.0125}, {"dt": 0.01}],
            "richardson": {"chosen_dt": 0.01},
        }

        with self.assertRaisesRegex(RuntimeError, "chosen_dt"):
            validate_prescribed_result(document)

    def test_both_errors_must_be_strictly_below_the_tolerance(self) -> None:
        record = supplemental_record(
            dt=0.008,
            predicted_error=5.0e-6,
            measured_error=4.0e-6,
            tolerance=5.0e-6,
            fields="artifacts/convergence/rk4-dt0.008/fields.jsonl",
        )

        self.assertFalse(record["predicted_below_tolerance"])
        self.assertTrue(record["measured_below_tolerance"])
        self.assertFalse(record["both_below_tolerance"])

    def test_supplement_passes_only_when_both_errors_pass(self) -> None:
        record = supplemental_record(
            dt=0.008,
            predicted_error=2.3e-6,
            measured_error=2.2e-6,
            tolerance=5.0e-6,
            fields="artifacts/convergence/rk4-dt0.008/fields.jsonl",
        )

        self.assertEqual(
            record["label"], "原候选全部未达标后的补充实验"
        )
        self.assertTrue(record["both_below_tolerance"])


if __name__ == "__main__":
    unittest.main()

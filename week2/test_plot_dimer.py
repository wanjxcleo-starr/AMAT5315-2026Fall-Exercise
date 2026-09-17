"""Check the dimer renderer without creating the student's final image."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("plot_dimer.py")


def sample_csv(include_long_run: bool = True) -> str:
    rows = ["run,step,time,relative_error"]
    runs = [("euler_500", 500), ("verlet_500", 500)]
    if include_long_run:
        runs.append(("verlet_5000", 5000))
    for name, steps in runs:
        for step in range(steps + 1):
            if name == "euler_500":
                error = 0.6 * step / steps
            else:
                error = -0.0002 * math.sin(step / 11) ** 2
            rows.append(f"{name},{step},{step * 0.01:.2f},{error:.17g}")
    return "\n".join(rows) + "\n"


class DimerPlotTests(unittest.TestCase):
    def test_left_panel_compares_signed_500_step_errors_with_legend(self) -> None:
        import plot_dimer

        series = {
            "euler_500": [(0.0, 0.0), (0.01, 0.5), (0.02, 0.8)],
            "verlet_500": [(0.0, 0.0), (0.01, -0.0002), (0.02, 0.0001)],
            "verlet_5000": [(0.0, 0.0), (0.01, -0.0002), (50.0, 0.0001)],
        }
        builder = getattr(plot_dimer, "build_figure", None)
        self.assertIsNotNone(builder)
        figure = builder(series)
        try:
            left, right = figure.axes
            self.assertEqual(left.get_title(), "Euler vs velocity-Verlet · 500 steps")
            self.assertEqual(
                [text.get_text() for text in left.get_legend().get_texts()],
                ["Euler", "velocity-Verlet"],
            )
            lines = {line.get_label(): list(line.get_ydata()) for line in left.get_lines()}
            self.assertEqual(lines["Euler"], [0.0, 0.5, 0.8])
            self.assertEqual(lines["velocity-Verlet"], [0.0, -0.0002, 0.0001])
            self.assertEqual(list(right.get_lines()[0].get_ydata()), [0.0, -0.2, 0.1])
            self.assertIn("1000", right.get_ylabel())
        finally:
            plot_dimer.plt.close(figure)

    def test_complete_series_writes_png_at_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(output)],
                input=sample_csv(),
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(output.stat().st_size, 1000)

    def test_missing_long_run_is_rejected_without_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(output)],
                input=sample_csv(include_long_run=False),
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

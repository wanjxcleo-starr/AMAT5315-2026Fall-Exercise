"""Exercise the public dimer command with an image path in /tmp."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WEEK2 = Path(__file__).resolve().parent


class DimerExampleTests(unittest.TestCase):
    def test_example_creates_png_in_requested_temporary_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            environment = os.environ.copy()
            environment["PATH"] = (
                str(Path(sys.executable).parent)
                + os.pathsep
                + environment.get("PATH", "")
            )
            result = subprocess.run(
                [
                    "cargo", "run", "--quiet", "--manifest-path", "md/Cargo.toml",
                    "--example", "dimer", "--", str(output),
                ],
                cwd=WEEK2,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("Euler final signed error", result.stdout)
            self.assertIn("Verlet 500-step max absolute error", result.stdout)


if __name__ == "__main__":
    unittest.main()

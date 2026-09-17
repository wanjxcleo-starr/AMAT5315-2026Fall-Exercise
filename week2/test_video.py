"""Check the Week 2 fluid animation without an external video encoder."""

import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import video  # noqa: E402


SCRIPT = Path(video.__file__)
METADATA = {
    "n": 2, "rho": 0.02, "box": [10.0, 10.0], "dt": 0.5,
    "temperature": 0.5, "eq_steps": 0, "steps": 2,
    "sample_every": 1, "seed": 2026, "integrator": "velocity-verlet",
}
FRAMES = [
    {"step": 1, "t": 0.5, "pos": [[0.1, 1.0], [9.9, 1.0]], "vel": [[0.0, 0.0]] * 2},
    {"step": 2, "t": 1.0, "pos": [[0.0, 1.0], [3.0, 1.0]], "vel": [[0.0, 0.0]] * 2},
]


def write_run(folder: Path) -> None:
    (folder / "run.json").write_text(json.dumps(METADATA) + "\n")
    (folder / "traj.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in FRAMES) + "\n"
    )


class CapturingWriter:
    def __init__(self) -> None:
        self.positions = []
        self.profile_maxima = []
        self.figure = None

    @contextlib.contextmanager
    def saving(self, figure, _path, dpi):
        self.figure = figure
        self.dpi = dpi
        yield self

    def grab_frame(self) -> None:
        self.figure.canvas.draw()
        self.positions.append(self.figure.axes[0].collections[0].get_offsets().copy())
        self.profile_maxima.append(max(self.figure.axes[1].lines[0].get_ydata()))


class OversizeWriter:
    @contextlib.contextmanager
    def saving(self, _figure, path, dpi):
        yield self
        Path(path).write_bytes(b"x" * video.MAX_MP4_BYTES)

    def grab_frame(self) -> None:
        pass


class VideoTests(unittest.TestCase):
    def test_minimum_image_pair_builds_cumulative_normalized_gr(self) -> None:
        radii, profiles = video.cumulative_gr(
            [frame["pos"] for frame in FRAMES], [10.0, 10.0], 0.02,
            bin_width=0.5,
        )
        self.assertEqual(len(radii), 10)
        self.assertEqual(len(profiles), len(FRAMES))
        self.assertAlmostEqual(profiles[0][0], 200 / math.pi, places=8)
        self.assertAlmostEqual(profiles[1][0], 100 / math.pi, places=8)
        self.assertEqual(profiles[0][1], 0.0)

    def test_each_saved_state_becomes_one_side_by_side_video_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder)
            metadata, frames = video.load_run(folder)
            self.assertEqual(len(frames), 2)
            writer = CapturingWriter()
            with mock.patch.object(video, "FFMpegWriter", return_value=writer):
                video.render_video(metadata, frames, folder / "sample.mp4")
            self.assertEqual(len(writer.positions), 2)
            self.assertEqual(len(writer.figure.axes), 2)
            self.assertEqual(writer.positions[0].tolist(), FRAMES[0]["pos"])
            self.assertEqual(writer.positions[1].tolist(), FRAMES[1]["pos"])
            self.assertIn("g(r)", writer.figure.axes[1].get_title())
            self.assertGreaterEqual(
                writer.figure.axes[1].get_ylim()[1], max(writer.profile_maxima)
            )

    def test_oversize_encode_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder)
            output = folder / "sample.mp4"
            output.write_bytes(b"previous video")
            error_output = io.StringIO()
            with (
                mock.patch.object(video, "FFMpegWriter", return_value=OversizeWriter()),
                mock.patch.object(video.shutil, "which", return_value="/tmp/ffmpeg"),
                mock.patch.object(sys, "argv", ["video.py", str(folder), str(output)]),
                contextlib.redirect_stderr(error_output),
            ):
                self.assertEqual(video.main(), 1)
            self.assertEqual(output.read_bytes(), b"previous video")
            self.assertIn("required size", error_output.getvalue())

    def test_missing_ffmpeg_is_reported_without_creating_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_run(folder)
            output = folder / "missing.mp4"
            env = dict(os.environ, PATH="", MPLCONFIGDIR=str(folder / "mplconfig"))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(folder), str(output)],
                text=True, capture_output=True, env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ffmpeg", result.stderr.lower())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

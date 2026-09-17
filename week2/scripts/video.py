"""Animate saved atoms beside a running radial distribution function."""

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


MAX_MP4_BYTES = 5_000_000


def load_run(folder: Path) -> tuple[dict, list[dict]]:
    """Read every recorded production frame in step order."""
    metadata = json.loads((folder / "run.json").read_text())
    n = metadata["n"]
    rho = metadata["rho"]
    box = np.asarray(metadata["box"], dtype=float)
    dt = metadata["dt"]
    sample_every = metadata["sample_every"]
    steps = metadata["steps"]
    if not isinstance(n, int) or n < 2 or not math.isfinite(rho) or rho <= 0:
        raise ValueError("run.json needs positive n and rho")
    if box.shape != (2,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("run.json needs two positive finite box lengths")
    if not isinstance(steps, int) or steps <= 0 or not isinstance(sample_every, int) or sample_every <= 0:
        raise ValueError("run.json needs positive steps and sample_every")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("run.json needs positive finite dt")

    frames = []
    with (folder / "traj.jsonl").open() as trajectory:
        for number, line in enumerate(trajectory, start=1):
            frame = json.loads(line)
            positions = np.asarray(frame["pos"], dtype=float)
            expected_step = number * sample_every
            if frame["step"] != expected_step or not math.isclose(
                frame["t"], expected_step * dt, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"frame {number}: incorrect production step or time")
            if positions.shape != (n, 2) or not np.all(np.isfinite(positions)):
                raise ValueError(f"frame {number}: expected {n} finite two-dimensional positions")
            if np.any(positions < 0) or np.any(positions >= box):
                raise ValueError(f"frame {number}: position outside periodic box")
            frames.append(frame)
    if not frames or len(frames) != steps // sample_every:
        raise ValueError("traj.jsonl: missing or unexpected saved frames")
    return metadata, frames


def cumulative_gr(
    position_frames: list[list[list[float]]],
    box: list[float],
    rho: float,
    bin_width: float = 0.1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Count nearest-image pairs twice and normalize by n*rho*shell area*frames."""
    if not position_frames or not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("need saved positions and a positive radial bin width")
    box_array = np.asarray(box, dtype=float)
    r_max = float(np.min(box_array)) / 2.0
    bins = math.ceil(r_max / bin_width)
    edges = np.linspace(0.0, r_max, bins + 1)
    radii = (edges[:-1] + edges[1:]) / 2.0
    shell_areas = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    n = len(position_frames[0])
    i, j = np.triu_indices(n, k=1)
    counts = np.zeros(bins, dtype=float)
    profiles = []
    for frame_number, positions in enumerate(position_frames, start=1):
        points = np.asarray(positions, dtype=float)
        if points.shape != (n, 2):
            raise ValueError(f"frame {frame_number}: inconsistent atom count")
        separation = points[j] - points[i]
        separation -= box_array * np.rint(separation / box_array)
        distances = np.sqrt(np.sum(separation * separation, axis=1))
        pairs, _ = np.histogram(distances, bins=edges)
        counts += 2.0 * pairs
        profiles.append(counts.copy() / (frame_number * n * rho * shell_areas))
    return radii, profiles


def render_video(metadata: dict, frames: list[dict], output: Path) -> None:
    """Send one drawn side-by-side figure to FFmpeg for each saved frame."""
    box = metadata["box"]
    radii, profiles = cumulative_gr(
        [frame["pos"] for frame in frames], box, metadata["rho"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=100)
    fig.suptitle("Two-dimensional Lennard–Jones fluid", fontsize=14)
    atoms = axes[0].scatter([], [], s=34, color="#176a79", edgecolors="white", linewidths=0.4)
    axes[0].set(
        title="Saved atom positions", xlabel="x", ylabel="y",
        xlim=(0, box[0]), ylim=(0, box[1]),
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.15)

    profile_line, = axes[1].plot(radii, np.zeros_like(radii), color="#bf5b32", linewidth=2)
    axes[1].axhline(1.0, color="#667381", linewidth=1, linestyle="--")
    axes[1].set(
        title="Running g(r) from saved positions", xlabel="r", ylabel="g(r)",
        xlim=(0, radii[-1]),
        ylim=(0, max(3.5, max(float(np.max(profile)) for profile in profiles) * 1.15)),
    )
    axes[1].grid(alpha=0.2)
    caption = fig.text(0.5, 0.02, "", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18, wspace=0.3)

    writer = FFMpegWriter(
        fps=20, codec="libx264", bitrate=700,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with writer.saving(fig, str(output), dpi=100):
            for number, (frame, profile) in enumerate(zip(frames, profiles), start=1):
                atoms.set_offsets(frame["pos"])
                profile_line.set_ydata(profile)
                caption.set_text(
                    f"Production t = {frame['t']:.2f}  ·  frame {number}/{len(frames)}"
                )
                writer.grab_frame()
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("mp4", type=Path)
    arguments = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg executable is required; install FFmpeg and put ffmpeg on PATH", file=sys.stderr)
        return 1
    try:
        metadata, frames = load_run(arguments.run_folder)
        with tempfile.TemporaryDirectory(
            prefix=".fluid-video-", dir=arguments.mp4.parent
        ) as directory:
            temporary_mp4 = Path(directory) / arguments.mp4.name
            render_video(metadata, frames, temporary_mp4)
            size = temporary_mp4.stat().st_size
            if size >= MAX_MP4_BYTES:
                raise ValueError(f"MP4 is {size} bytes; required size is below {MAX_MP4_BYTES}")
            temporary_mp4.replace(arguments.mp4)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"video failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {arguments.mp4} ({len(frames)} video frames, {size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

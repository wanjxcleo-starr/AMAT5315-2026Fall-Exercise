"""Plot Rust-computed Lennard-Jones pair energy and force around a fixed atom.

Run from the repository root with Matplotlib installed: python3 week2/plot_field.py
"""

import csv
import io
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle


WEEK2 = Path(__file__).resolve().parent
OUTPUT = WEEK2 / "field.png"
R0 = 2.0 ** (1.0 / 6.0)


def main() -> None:
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "field_data"],
        cwd=WEEK2 / "md",
        check=True,
        capture_output=True,
        text=True,
    )
    grid = []
    arrows = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        if row["kind"] == "grid":
            grid.append((float(row["x"]), float(row["y"]), float(row["energy"])))
        elif row["kind"] == "arrow":
            arrows.append((float(row["x"]), float(row["y"]), float(row["fx"]), float(row["fy"])))
        else:
            raise ValueError(f"unknown row kind: {row['kind']}")

    size = math.isqrt(len(grid))
    if size * size != len(grid) or not arrows:
        raise ValueError("missing grid or force samples")
    grid = np.asarray(grid)
    arrow = np.asarray(arrows)
    energy = np.ma.masked_invalid(grid[:, 2].reshape(size, size))
    x_values = grid[:size, 0]
    y_values = grid[::size, 1]
    half_step = (x_values[1] - x_values[0]) / 2.0
    extent = (x_values[0] - half_step, x_values[-1] + half_step,
              y_values[0] - half_step, y_values[-1] + half_step)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    fig.subplots_adjust(left=0.10, right=0.84, top=0.89, bottom=0.12)
    fig.suptitle("Lennard-Jones pair energy and force", fontsize=17, fontweight="bold")
    ax.set_title(r"Fixed atom at the origin · $\epsilon=\sigma=1$ · no cutoff", fontsize=11)
    image = ax.imshow(
        np.ma.clip(energy, -1.0, 2.0),
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0),
        interpolation="bilinear",
    )

    force_size = np.hypot(arrow[:, 2], arrow[:, 3])
    arrow_length = 0.16 + 0.16 * np.tanh(force_size / 4.0)
    dx = arrow[:, 2] * arrow_length / force_size
    dy = arrow[:, 3] * arrow_length / force_size
    outward = arrow[:, 0] * arrow[:, 2] + arrow[:, 1] * arrow[:, 3] > 0.0
    for mask, color in [(outward, "#fff0b5"), (~outward, "#10253f")]:
        ax.quiver(
            arrow[mask, 0], arrow[mask, 1], dx[mask], dy[mask],
            angles="xy", scale_units="xy", scale=1, pivot="mid",
            color=color, edgecolors="white", linewidths=0.4,
            width=0.006, headwidth=4.5, headlength=5.0, zorder=4,
        )

    ax.add_patch(Circle((0, 0), R0, fill=False, edgecolor="#ffd166",
                        linestyle=(0, (5, 3)), linewidth=2.2, zorder=5))
    ax.scatter([0], [0], s=200, c="#152338", edgecolors="white", linewidths=1.5, zorder=6)
    ax.plot([], [], color="#c68a00", linestyle="--", linewidth=2, label=r"Equilibrium radius $r_0=2^{1/6}$")
    ax.legend(loc="upper right", framealpha=0.93, fontsize=10)
    ax.set(xlabel=r"$x/\sigma$", ylabel=r"$y/\sigma$", aspect="equal")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label(r"Pair energy $U(r)$ (clipped above $+2$)")
    fig.text(0.5, 0.045,
             "Arrows show force direction; their lengths are compressed for visibility. The origin is masked.",
             ha="center", fontsize=9, color="#334155")
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

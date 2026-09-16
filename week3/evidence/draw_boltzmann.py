"""Draw the energy histograms and their Boltzmann log ratio."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WEEK3 = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("boltzmann.png")
SITES = 4096
BIN_WIDTH = 40
MIN_COUNT = 5


def total_energies(run_name: str, temperature: float) -> list[float]:
    path = WEEK3 / "runs" / run_name / "series.jsonl"
    energies = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if row["L"] != 64 or not math.isclose(row["T"], temperature, abs_tol=1e-6):
                raise ValueError(f"unexpected L or T in {path}:{line_number}")
            energy = row["E"] * SITES
            if not math.isfinite(energy):
                raise ValueError(f"non-finite energy in {path}:{line_number}")
            energies.append(energy)
    if not energies:
        raise ValueError(f"no measured rows in {path}")
    return energies


def common_bins(first: list[float], second: list[float]) -> list[int]:
    lowest = min(min(first), min(second))
    highest = max(max(first), max(second))
    # True energies are multiples of 4. Edges at 2 mod 40 keep the
    # six-decimal rounding of E away from any bin boundary.
    first_edge = math.floor(lowest / BIN_WIDTH) * BIN_WIDTH + 2
    if first_edge > lowest:
        first_edge -= BIN_WIDTH
    count = math.floor((highest - first_edge) / BIN_WIDTH) + 1
    return [first_edge + index * BIN_WIDTH for index in range(count + 1)]


def histogram(values: list[float], edges: list[int]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for energy in values:
        index = math.floor((energy - edges[0]) / BIN_WIDTH)
        counts[index] += 1
    return counts


def draw() -> None:
    first = total_energies("T3.0", 3.0)
    second = total_energies("T3.1", 3.1)
    edges = common_bins(first, second)
    first_counts = histogram(first, edges)
    second_counts = histogram(second, edges)

    centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
    shared = [
        (center, first_count, second_count)
        for center, first_count, second_count in zip(centers, first_counts, second_counts)
        if first_count >= MIN_COUNT and second_count >= MIN_COUNT
    ]
    if not shared:
        raise ValueError("no bins contain at least five rows from each run")

    ratio_x = [center for center, _, _ in shared]
    ratio_y = [
        math.log(second_count / len(second)) - math.log(first_count / len(first))
        for _, first_count, second_count in shared
    ]
    slope = 1 / 3.0 - 1 / 3.1
    # Fit only the vertical offset; the reference line keeps the requested slope.
    weights = [first_count * second_count / (first_count + second_count) for _, first_count, second_count in shared]
    weight_sum = sum(weights)
    reference_x = sum(weight * x for weight, x in zip(weights, ratio_x)) / weight_sum
    reference_y = sum(weight * y for weight, y in zip(weights, ratio_y)) / weight_sum

    fig, (hist_ax, ratio_ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]}
    )
    fig.suptitle(r"$64\times64$ Ising model: energy and Boltzmann ratio", fontsize=15)
    hist_ax.stairs(first_counts, edges, fill=True, alpha=0.4, color="#2563a6", label=f"T = 3.0 (n = {len(first):,})")
    hist_ax.stairs(second_counts, edges, fill=True, alpha=0.4, color="#df7b28", label=f"T = 3.1 (n = {len(second):,})")
    hist_ax.set_ylabel("Rows per 40-energy bin")
    hist_ax.legend(frameon=False)
    hist_ax.grid(axis="y", alpha=0.25)

    ratio_ax.scatter(ratio_x, ratio_y, s=40, color="#173b65", zorder=3, label="Log probability ratio")
    line_x = [min(ratio_x), max(ratio_x)]
    line_y = [reference_y + slope * (x - reference_x) for x in line_x]
    ratio_ax.plot(line_x, line_y, color="#bf3d30", linewidth=2, label=rf"Slope $1/3.0-1/3.1={slope:.5f}$ (offset fitted)")
    ratio_ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.5)
    ratio_ax.set_xlabel(r"Total energy $E_{\mathrm{tot}}=4096E$")
    ratio_ax.set_ylabel(r"$\log[P_{3.1}(E)/P_{3.0}(E)]$")
    ratio_ax.legend(frameon=False, loc="upper left")
    ratio_ax.grid(alpha=0.25)

    fig.text(
        0.5,
        0.02,
        f"Common 40-unit bins; log ratio uses {len(shared)} bins with at least {MIN_COUNT} rows from each run.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.14, hspace=0.09)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUTPUT}; {len(shared)} ratio bins; reference slope {slope:.8f}")


if __name__ == "__main__":
    draw()

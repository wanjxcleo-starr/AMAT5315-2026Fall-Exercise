# Week 2: molecular dynamics

The Rust crate in `md/` simulates a two-dimensional Lennard–Jones fluid with
velocity-Verlet integration. It provides naive all-pairs and cell-list force
paths, deterministic recording, and a linear temperature ramp. The accompanying
Python scripts validate saved runs and produce figures and videos.

## Environment and basic reproduction

The reported measurements used WSL2 (Linux
6.18.33.2-microsoft-standard-WSL2), an Intel Core i7-8565U with 8 logical
CPUs, Python 3.14.4, NumPy 2.5.3, and Rust 1.98.1. The videos used Matplotlib
3.11.2 and FFmpeg 7.0.2. The profiling results used `samply` 0.13.1.

From `week2/`, with NumPy and Matplotlib installed in the active Python
environment, build, test, and reproduce the standard run with:

```sh
cargo test --manifest-path md/Cargo.toml
cargo build --release --manifest-path md/Cargo.toml
make reproduce
python3 scripts/check.py artifacts > check.txt
```

`scripts/video.py` additionally requires an `ffmpeg` executable on `PATH`.
Generated `artifacts/` directories and Rust build output are not committed.

## Python and Rust timing

The comparison used the
[course NumPy script](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week2-sim.py),
whose SHA-256 was
`ec03acaf7e28fed74f4faa28a1b30e57924a0c6a1af4afd04a50f3f22ba9f7bc`.
Dependencies and compilation were completed before timing. Each program ran
three times sequentially with its own output directory, and every run produced
200 trajectory frames.

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy `week2-sim.py` | 23.73 | 23.44–39.78 |
| Rust debug | 16.66 | 16.42–27.91 |
| Rust release | 2.89 | 2.60–3.04 |

| Program | Run 1 (s) | Run 2 (s) | Run 3 (s) |
| --- | ---: | ---: | ---: |
| NumPy `week2-sim.py` | 39.78 | 23.44 | 23.73 |
| Rust debug | 27.91 | 16.66 | 16.42 |
| Rust release | 2.60 | 2.89 | 3.04 |

The first NumPy and debug runs were noticeably slower than their later runs.
The Rust command used the following physical parameters; replace `OUT` with a
fresh directory for each timed run:

```sh
./md/target/release/md \
  --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 \
  --out OUT
```

## Force profiling

The saved Firefox Profiler call-tree screenshots give the following displayed
values:

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 98 | 4.1 |
| Cell list | 95 | 2.3 |

In [profile-naive.png](profile-naive.png),
`md::fluid::forces_and_potential` and its callees account for 3,948 of 4,013
displayed samples. In [profile-cells.png](profile-cells.png),
`md::fluid::forces_and_potential_cells` and its callees account for 1,421 of
1,491 displayed samples. The naive screenshot used the then-default requested
sampling rate of 1000 Hz; the retained cell-list screenshot is the median
cell-list run from the controlled 750 Hz comparison below. The screenshot
times therefore document the evidence files but are not a same-rate pair.

An initial cell-list profile took 5.9 s and did not improve on the 4.1 s naive
screenshot. Kernel sampling pressure subsequently lowered the available rate,
so three naive/cell-list pairs were recorded at an explicit 750 Hz with the
same release binary and parameters:

| Pair at 750 Hz | Naive profile (s) | Naive samples | Cells profile (s) | Cells samples |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.450 | 3,174 | 1.656 | 1,184 |
| 2 | 4.595 | 3,272 | 2.275 | 1,495 |
| 3 | 5.120 | 3,591 | 5.172 | 2,853 |
| Median time | 4.595 | — | 2.275 | — |

The cell-list median is lower, but the third cell-list run is slightly slower
than its paired naive run. All six paired trajectories are byte-identical.
The slow profiles also have lower sampling density; changing profiling load,
CPU speed, or another runtime condition may contribute, so these data do not
show a consistent per-run speedup.

The controlled profiles can be regenerated into a temporary directory by
running the command below once with `METHOD=naive` and once with
`METHOD=cells`:

```sh
PROFILE_DIR=$(mktemp -d)
METHOD=cells
samply record --rate 750 --save-only \
  --output "$PROFILE_DIR/$METHOD.json.gz" \
  ./md/target/release/md \
  --n 400 --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 200 --steps 1000 --sample-every 50 --seed 2026 \
  --force "$METHOD" --out "$PROFILE_DIR/$METHOD"
```

Load the resulting profile with `samply load` and select the full-range `md`
Call Tree to inspect the force share and process timeline.

## Scaling benchmark

The release benchmark was run on 2026-09-18 on the machine described above.
Each wall time surrounded one sequential subprocess; compilation was excluded.
For each particle count and repetition, naive and cell-list runs were adjacent,
with their order reversed in the middle repetition. Every matched pair produced
byte-identical trajectories and ten saved frames.

| N | Repetition | Naive (s) | Cells (s) |
| ---: | ---: | ---: | ---: |
| 100 | 1 | 0.160253 | 0.208762 |
| 100 | 2 | 0.173535 | 0.206605 |
| 100 | 3 | 0.165563 | 0.223114 |
| 400 | 1 | 3.866570 | 0.808738 |
| 400 | 2 | 2.472577 | 0.852067 |
| 400 | 3 | 2.524566 | 0.893196 |
| 1600 | 1 | 36.164243 | 3.153881 |
| 1600 | 2 | 35.581781 | 3.390734 |
| 1600 | 3 | 36.740791 | 3.314179 |

| N | Naive: median [range] (s) | Cells: median [range] (s) | Speedup: ratio of medians [paired range] |
| ---: | ---: | ---: | ---: |
| 100 | 0.165563 [0.160253–0.173535] | 0.208762 [0.206605–0.223114] | 0.793× [0.742–0.840×] |
| 400 | 2.524566 [2.472577–3.866570] | 0.852067 [0.808738–0.893196] | 2.963× [2.826–4.781×] |
| 1600 | 36.164243 [35.581781–36.740791] | 3.314179 [3.153881–3.390734] | 10.912× [10.494–11.467×] |

The command form below was repeated for `N=100,400,1600`, three repetitions,
and both force methods, using a fresh output path each time:

```sh
./md/target/release/md \
  --n "$N" --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 100 --steps 500 --sample-every 50 --seed 2026 \
  --force "$METHOD" --out "$OUT"
```

[scaling.png](scaling.png) plots median wall time per integration step with
min–max bars. The cell-list setup costs more at `N=100`, but the speedup rises
with system size because the naive method searches every particle pair. The
times include startup and trajectory output, and the first `N=400` naive run
is slower than the other two; the full range is retained. The original
one-off plotting helper was not retained, so the figure is documented by the
raw values and definitions above rather than by a repository script.

## Heating, cold, and hot runs

After the release build, the trajectories and videos can be regenerated from
`week2/` with NumPy, Matplotlib, and FFmpeg available:

```sh
RUN_DIR=$(mktemp -d)
./md/target/release/md \
  --n 400 --rho 0.8 --temperature 0.2 --ramp-to 1.2 --dt 0.01 \
  --eq-steps 2000 --steps 20000 --sample-every 100 --seed 2026 \
  --out heating > "$RUN_DIR/heating.tsv"
./md/target/release/md \
  --n 100 --rho 0.8 --temperature 0.2 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 \
  --out "$RUN_DIR/cold" > "$RUN_DIR/cold.tsv"
./md/target/release/md \
  --n 100 --rho 0.8 --temperature 1.0 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 \
  --out "$RUN_DIR/hot" > "$RUN_DIR/hot.tsv"
python3 scripts/video.py "$RUN_DIR/cold" cold.mp4
python3 scripts/video.py "$RUN_DIR/hot" hot.mp4
```

The committed ramp has 200 frames from production steps 100 through 20000.
Temperatures recomputed as `sum(v_x²+v_y²)/(2N−2)` rise monotonically from
0.205000000 to 1.1999999999; the maximum absolute difference from the linear
target is `5.45e-10`. Step zero is not saved, so the first visible target is
0.205 rather than 0.2.

Using minimum-image distances, radial bins about 0.1 wide, and annular-area
normalization, the RMS of `g(r)−1` over `2.5 ≤ r ≤ 5.0` falls from 0.629
in the first 20 ramp frames to 0.123 in the last 20. The largest peak over
that interval falls from 2.48 to 1.19. In the fixed-temperature runs, the
same contrast is 0.538 for cold and 0.140 for hot; first-shell peaks are 4.43
and 2.88. The cold run retains 99.8% of the six nearest-neighbour identities,
whereas the hot run retains 4.3%, consistent with vibration versus neighbour
exchange. Finite size and sampling keep the hot long-range `g(r)` from being
exactly one. The videos show lattice-like cold positions with distinct
long-range peaks, while the hot run retains a first shell with much weaker
long-range structure.

FFmpeg decoded 200 frames from each MP4 at 20 fps, lasting 10 seconds with no
reported dropped frames. All required evidence files are below 5 MB:

| Evidence file | Size (bytes) |
| --- | ---: |
| `heating/run.json` | 280 |
| `heating/traj.jsonl` | 4,715,545 |
| `cold.mp4` | 627,020 |
| `hot.mp4` | 691,595 |

The trajectory retains the specified nine significant digits. It can be
supplied to the course viewer through its GitHub raw URL. `fluid-viewer.png`
is a retained manual viewer export; no separate public-viewer load verification
is recorded.

## Evidence index

Commands run from `week2/` unless noted. Profiler screenshots and viewer
exports require manual inspection.

| Evidence | Generator or inspection step |
| --- | --- |
| `force-comparison.txt` | `cargo run --manifest-path md/Cargo.toml --example force_comparison > force-comparison.txt` |
| `field.png` | `python3 plot_field.py` |
| `dimer.png` | `cargo run --manifest-path md/Cargo.toml --example dimer` |
| `check.txt` | `make reproduce`, then `python3 scripts/check.py artifacts > check.txt` |
| `fluid.mp4` | `python3 scripts/video.py artifacts fluid.mp4` |
| `fluid-viewer.png` | Manual viewer export after loading the standard fluid run |
| `force-paths.txt` | `cargo test --manifest-path md/Cargo.toml --test force_paths -- --nocapture --test-threads=1` |
| `profile-naive.png`, `profile-cells.png` | Firefox Profiler Call Tree exports from the profiling procedure above |
| `scaling.png` | Plot of the raw benchmark values above; original plotting helper was not retained |
| `heating/run.json`, `heating/traj.jsonl` | Temperature-ramp command above |
| `cold.mp4`, `hot.mp4` | Fixed-temperature and video commands above |

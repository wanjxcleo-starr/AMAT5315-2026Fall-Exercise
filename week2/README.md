# Week 2 molecular dynamics

## Timing

Wall-clock seconds for three sequential runs of each program. Downloads, package
installation, and Rust compilation were completed before timing.

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy week2-sim.py | 23.73 | 23.44–39.78 |
| Rust debug | 16.66 | 16.42–27.91 |
| Rust release | 2.89 | 2.60–3.04 |

| Program | Run 1 (s) | Run 2 (s) | Run 3 (s) |
| --- | ---: | ---: | ---: |
| NumPy week2-sim.py | 39.78 | 23.44 | 23.73 |
| Rust debug | 27.91 | 16.66 | 16.42 |
| Rust release | 2.60 | 2.89 | 3.04 |

Measured on 2026-09-18 in WSL2 (Linux 6.18.33.2-microsoft-standard-WSL2),
Intel Core i7-8565U, 8 logical CPUs. Python 3.14.4 and NumPy 2.5.3 ran in
`/tmp/amat5315-week2-measure.PcoX5A/venv`; Rust used rustc 1.98.1. The
[course NumPy script](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week2-sim.py)
had SHA-256 `ec03acaf7e28fed74f4faa28a1b30e57924a0c6a1af4afd04a50f3f22ba9f7bc`.
At the time of these earlier timing runs, the installed `md` at
`/home/wan_jiaxing/.cargo/bin/md` had the same SHA-256
(`d6b84de6272f3ce46d9ca5b38cc5d4b078bd267af9c68ab3ba7767333d838d52`)
as `md/target/release/md` built for that measurement.

The setup ran from `week2/`:

```sh
curl -fL --output /tmp/amat5315-week2-measure.PcoX5A/week2-sim.py https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week2-sim.py
python3 -m venv /tmp/amat5315-week2-measure.PcoX5A/venv
/tmp/amat5315-week2-measure.PcoX5A/venv/bin/pip install numpy
cargo build --manifest-path md/Cargo.toml
cargo build --release --manifest-path md/Cargo.toml
```

For each program below, `i` took the values 1, 2, and 3 in order; the programs
ran in the table's order. Each run used its own output directory under `/tmp`;
the NumPy script writes to a
relative `artifacts/`, so its working directory was the corresponding
`numpy$i` directory. All nine runs produced 200 trajectory frames. The first
NumPy and debug runs were noticeably slower than their later runs.

```sh
BASE=/tmp/amat5315-week2-measure.PcoX5A
(cd "$BASE/numpy$i" && /usr/bin/time -f '%e' -o "$BASE/numpy$i.time" "$BASE/venv/bin/python" "$BASE/week2-sim.py")
/usr/bin/time -f '%e' -o "$BASE/debug$i.time" ./md/target/debug/md --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out "$BASE/debug$i" > "$BASE/debug$i.stdout"
/usr/bin/time -f '%e' -o "$BASE/release$i.time" md --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out "$BASE/release$i" > "$BASE/release$i.stdout"
```

## Profile

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 98 | 4.1 |
| Cell list | 95 | 2.3 |

The naive measurement used `samply 0.13.1` with the installed release `md`,
before any cell list optimization. In the full-range `md` Call Tree shown in
[profile-naive.png](profile-naive.png), `md::fluid::forces_and_potential` and
its callees account for 3,948 of 4,013 displayed samples (98.38%, shown as
98% by Firefox Profiler); the timeline shows 4.1 s. These displayed values
determine the table. A separate count of the unsymbolicated raw profile found
3,985 of 4,022 sample stacks with a program counter in the force function's
machine-code address range (99.08%). That address-range count uses a different
attribution method from the symbolicated Call Tree and is not the table value.
The untracked raw profile is
`/tmp/amat5315-week2-measure.PcoX5A/profile-naive.json.gz`; the run's trajectory
is in `/tmp/md-prof`. The command ran from `week2/` after the student
temporarily set `kernel.perf_event_paranoid=1`:

```sh
samply record --save-only --output /tmp/amat5315-week2-measure.PcoX5A/profile-naive.json.gz md --n 400 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 200 --steps 1000 --sample-every 50 --seed 2026 --out /tmp/md-prof
```

The first cell-list measurement requested the same `samply 0.13.1` default
sampling rate (1000 Hz), the same simulation parameters, and the installed
release `md` (SHA-256 `7f1558d300dba7ba09483e14195d78bcf5fc41365c0c55ff34c8ed55b6581ef2`,
identical to `md/target/release/md` built from this source, with debug symbols).
Without a `--force` flag, this build selects `cells`. The first full-range
`md` Call Tree showed `md::fluid::forces_and_potential_cells` and its callees
at 3,150 of 3,325 displayed samples (94.74% from those counts; 94% shown by
Firefox Profiler), and its timeline showed 5.9 s. That first elapsed time was
1.8 s above the naive screenshot's 4.1 s and did not meet the learning sheet's
elapsed-time comparison. The first screenshot was replaced by the paired-run
screenshot described below; its raw profile and trajectory remain outside Git at
`/tmp/amat5315-week2-cells-profile-zWIoQN/`.

```sh
samply record --save-only --output /tmp/amat5315-week2-cells-profile-zWIoQN/profile-cells.json.gz md --n 400 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 200 --steps 1000 --sample-every 50 --seed 2026 --out /tmp/amat5315-week2-cells-profile-zWIoQN/run > /tmp/amat5315-week2-cells-profile-zWIoQN/stdout.tsv
```

After that run, `kernel.perf_event_max_sample_rate` was 750, so a second
default-rate attempt was rejected before profiling. Without changing WSL
settings, three sequential naive/cells pairs were then recorded at an explicit
750 Hz with the **same installed release binary** and parameters. The times
below are the `md` process lifetime in each real samply profile:
`(processShutdownTime - processStartupTime) / 1000`, rounded to milliseconds.
The first profile yields 5.923 s by this method, consistent with its 5.9 s
Firefox display. Raw profiles and trajectories are outside Git in
`/tmp/amat5315-week2-paired-profiles-1ROipz/`.

The saved [profile-cells.png](profile-cells.png) shows the full-range `md` Call
Tree for pair 2's cells run, which is the median of the three cells runs.
`md::fluid::forces_and_potential_cells` and its callees account for 1,421 of
1,491 displayed samples (95.31% from those counts; 95% shown by Firefox
Profiler), and the timeline shows 2.3 s. The raw `md` process lifetime is
2.275 s. The Profile table uses the screenshot's displayed 95% and 2.3 s,
as it does for the naive screenshot.

| Pair at 750 Hz | Naive profile (s) | Naive samples | Cells profile (s) | Cells samples |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.450 | 3,174 | 1.656 | 1,184 |
| 2 | 4.595 | 3,272 | 2.275 | 1,495 |
| 3 | 5.120 | 3,591 | 5.172 | 2,853 |
| Median time | 4.595 | — | 2.275 | — |

The cells median is below the naive median, but one cells run took 5.172 s,
slightly longer than its paired naive run. The first 5.9 s result remains
documented above; the table shows the median repeat rather than the fastest
one. All six paired runs produced byte-identical trajectories. An unprofiled,
single-run diagnostic with the same physics parameters took 4.65 s (naive)
and 1.50 s (cells); these are different measurements and do not replace the
profile times. The slow cells profiles have fewer observed samples per second
(563 for the first profile and 552 for pair 3, versus 657–715 for the paired
profiles that ran faster). Summed `threadCPUDelta` in the slow profiles is
5.899 s and 5.153 s, close to their 5.923 s and 5.172 s process lifetimes,
so long off-CPU pauses are not supported by these profiles. The lower sample
density and longer on-CPU time are real. The WSL kernel log around the first
sampling session reports `perf: interrupt took too long` and lowers
`perf_event_max_sample_rate` through 1500 and 1250 to 750. This confirms
sampling pressure and changing effective conditions, but does not quantify
how much of the extra `md` CPU time came from sampling, CPU speed, or another
runtime condition. The current cells screenshot's 2.3 s is below the earlier
naive screenshot's 4.1 s, but those screenshots used different requested
sampling rates (750 versus 1000 Hz) and different release builds. The paired
profiles provide a same-binary, same-rate comparison; they do not show a
consistent speedup on every run.

One paired command, repeated with `--force naive` and `--force cells` and
separate output paths for pairs 1–3, was:

```sh
samply record --rate 750 --save-only --output /tmp/amat5315-week2-paired-profiles-1ROipz/cells-2.json.gz md --n 400 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 200 --steps 1000 --sample-every 50 --seed 2026 --force cells --out /tmp/amat5315-week2-paired-profiles-1ROipz/cells-2 > /tmp/amat5315-week2-paired-profiles-1ROipz/cells-2.stdout
```

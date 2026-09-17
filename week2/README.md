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
The installed `md` at `/home/wan_jiaxing/.cargo/bin/md` had the same SHA-256
(`d6b84de6272f3ce46d9ca5b38cc5d4b078bd267af9c68ab3ba7767333d838d52`)
as `md/target/release/md` built from the current source.

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
| Cell list | — | — |

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

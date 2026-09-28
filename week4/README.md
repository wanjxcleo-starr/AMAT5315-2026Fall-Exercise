# Week 4: spectral flow

This Rust crate implements the current learning sheet's periodic-line and two-dimensional flow experiments. Forward Euler, explicit-midpoint RK2, and classical RK4 share one `Integrator` interface. Parts 1–3, including their manual figure checks, are complete. In Part 4 the fourth-order checks and all three figure inspections pass, but none of the prescribed candidate steps meets the `5e-6` target and `chosen_dt` remains null. A separately authorized `dt=0.008` supplement passes; it does not make all of the learning sheet's prescribed acceptance conditions pass. The Extension and Challenge are not included.

`N` is a power of two and arrays use index `iy*N + ix`. The two-dimensional inverse FFT is normalized once by `1/N²`. First derivatives zero the even-grid Nyquist line to preserve real-field symmetry. The Poisson solve sets `ψ̂(0)=0`; velocity recovery separately sets `û(0)=v̂(0)=0`. The rectangular two-thirds mask is applied to the initial vorticity, every integrator stage, and the transformed nonlinear term. Each stage recovers its own velocity.

Energy and enstrophy are `E=mean((u²+v²)/2)` and `Z=mean(ω²/2)`. Random fields use resolution-independent seeded phases and are normalized to `E(0)=0.5`. `t_end/dt` must be an integer within floating-point tolerance; no step is silently changed. Snapshot arrays in `fields.jsonl` have six decimal places.

## Environment

From `week4/`, create a local Python environment and install the two analysis dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
cargo install --path . --quiet
```

The `.venv/`, Rust `target/`, full `artifacts/` recordings, and Python caches are ignored. Activate `.venv` before running the Python commands below.

## Automated checks

From `week4/`:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
python -m unittest discover -s scripts -p 'test_*.py'
```

The current run passes all 31 Rust tests and all 18 Python tests. These include the shared integrator stability polynomials, all three methods against an exact Fourier wave, the Nyquist rule, both line derivative implementations, the Part 1 study contract, the Part 2 derivative comparison, the prescribed Part 3 perturbation, the Part 4 error calculations, and all existing two-dimensional regression tests.

## Part 1: integrators on a line

With the environment active, run from `week4/`:

```sh
python -m unittest scripts/test_part1.py
python scripts/part1.py
```

The script obtains all numerical data from the Rust library through `line-study` and writes `evidence/line-stability.png` and `evidence/line-accuracy.png`. The measured RK4 growth boundary agrees with its stability polynomial; the `dt=0.045` pulse remains stable while `dt=0.056` develops the shortest-wave instability. The one-lap maximum errors are `1.800745e-05` for RK4 with Fourier derivatives, `3.098969e-01` for RK4 with centred differences, and `2.098856e-01` for Euler with Fourier derivatives. Fitted slopes are `1.032742` (Euler), `2.005486` (midpoint), `4.003969` (RK4), and `2.002533` (equal-weight RK4), all within 15% of their required values. Manual inspection confirmed the stability contrast, exact-solution comparison, and order plot.

## Part 2: flow discretization

The command uses the required explicit `--method rk4` argument:

```sh
mkdir -p evidence
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
      --out artifacts/taylor-green \
  | tee evidence/taylor-green.txt
```

This writes `run.json` and `fields.jsonl` under the ignored `artifacts/taylor-green/` directory. The existing recording ended with `E=0.167580011509` and `Z=0.335160023018`. Extracting the shared integrators did not change the RK4 stages, weights, accumulation order, or effective filtering, and all two-dimensional regression tests still pass, so the recording remains applicable to the current implementation and was not rerun.

Run the current Part 2 comparisons with NumPy and Matplotlib available:

```sh
python -m unittest scripts/test_part2.py
python scripts/part2.py
```

The four Fourier derivative errors for `g=sin(3x)cos(2y)` are below `1.6e-13`. Halving the centred-difference spacing gives error ratios from `3.90` to `3.97`. The script writes the current analytic field to ignored `artifacts/taylor-green/exact-t1.json`; the recorded final velocity has relative error `7.038587e-07`, below `1e-5`. It also writes `evidence/taylor-green.png` with the `t=0` and `t=1` vorticity and velocity on a shared colour scale. Manual inspection confirmed matching vortex positions and rotation directions, while the shared scale shows `max|ω|` falling from `2.000` to about `1.637`.

## Part 3: stability limit and sensitivity

First generate the full seeded random recording required by both Part 3 and Part 4:

```sh
mkdir -p artifacts/random evidence
field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
      --out artifacts/random \
  | tee artifacts/random.tsv evidence/decay.txt
python -m unittest scripts/test_part3.py
python scripts/part3.py
```

The script reuses the existing baseline random recording and runs only the missing stability and sensitivity cases. Because the fixed-step `fluid` interface rejects non-integral `t_end/dt`, the `dt=0.033` and `0.038` horizons are explicitly extended to the first complete step (`8.019` and `10.032`); neither `dt` is changed or shortened.

For Taylor–Green, the diffusive prediction is `0.03157596`; `dt=0.032` remains stable and `dt=0.033` becomes non-finite at `t=7.854`. The random generator gives `Umax=2.79409248` and the advective bound `0.01705224`. Both requested random steps `0.038` and `0.040` fail, so the prescribed downward scan gives `0.026` stable and `0.028` non-finite at `t=1.008`, respectively `1.52` and `1.64` times the bound. Euler at `dt=0.01` is non-finite at `t=0.86`. At `t=20`, the random perturbation distance has grown by a factor `126.64`, while Taylor–Green reaches equality at the six-decimal storage floor before small relative rounding effects reappear.

The script writes `evidence/blowup.png`, `evidence/sensitivity.png`, and `evidence/random.png`. All numerical acceptance checks pass. Manual inspection confirmed the blow-up, sensitivity, and random-flow evolution plots. Taylor–Green at `dt=0.032` shows no obvious instability only over the displayed interval; this is not evidence of long-time stability. The late Taylor–Green perturbation comparison is limited by the six-decimal snapshot precision.

## Part 4: order and step selection

With the environment active, run from `week4/`:

```sh
python -m unittest scripts/test_part4.py
python scripts/part4.py
python -m unittest scripts/test_part4_supplement.py
python scripts/part4_supplement.py
```

The main script writes `evidence/order.png`, `evidence/convergence.png`, and `evidence/convergence.json`, retaining the new final fields under ignored `artifacts/order/` and `artifacts/convergence/`. It reuses the matching `dt=0.01`, `t=2` frame from the full `artifacts/random/` recording; its initial vorticity agrees exactly with the newly generated runs.

**Fourth-order verification:** Taylor–Green relative velocity errors at `dt=0.4, 0.25, 0.2` are `5.986617e-4`, `7.879896e-5`, and `3.574224e-5`; the fitted slope is `4.104406`, within 15% of four. Random-flow relative vorticity errors against the `dt=0.0025` reference are `8.946387e-5`, `1.330460e-5`, and `5.393607e-6` at `dt=0.02, 0.0125, 0.01`; the slope is `4.052403`, within the required `[3.7, 4.3]`.

**Prescribed candidate result:** Richardson estimation from `dt=0.02` and `0.01` predicts errors `8.998212e-5`, `1.373018e-5`, and `5.623883e-6`. Therefore none of the three prescribed candidates is below `5e-6`; `dt=0.01` also has measured error `5.393607e-6`. The original result remains `chosen_dt: null`, and the main script exits non-zero instead of relaxing the threshold.

**Supplement after all prescribed candidates failed:** the separately authorized `dt=0.008` run keeps `N=128`, `nu=0.004`, seed 2026, band 2–6, the same initial field, `t_end=2`, and the existing `dt=0.0025` reference. Fourth-order scaling predicts `2.303542e-6`; the measured error is `2.184949e-6`. Both are strictly below `5e-6`. This is a check of one supplemental step, not a claim that `0.008` is the globally largest usable step. Its result is stored separately in `convergence.json`, and `evidence/convergence-supplemental.png` leaves the original candidate plot and null choice unchanged. Manual inspection confirmed `order.png`, `convergence.png`, and `convergence-supplemental.png`.

## Current-sheet evidence regeneration

Run the commands above in order. The evidence mapping is:

| Evidence | Producing command |
|---|---|
| `evidence/line-stability.png`, `evidence/line-accuracy.png` | `python scripts/part1.py` |
| `evidence/taylor-green.txt` | the Taylor–Green `field | fluid | tee` pipeline in Part 2 |
| `evidence/taylor-green.png` | `python scripts/part2.py` after that pipeline |
| `evidence/decay.txt` | the random `field | fluid | tee` pipeline in Part 3 |
| `evidence/blowup.png`, `evidence/sensitivity.png`, `evidence/random.png` | `python scripts/part3.py` after the random pipeline |
| `evidence/order.png`, `evidence/convergence.png`, `evidence/convergence.json` | `python scripts/part4.py` |
| `evidence/convergence-supplemental.png` and the separate supplemental section in `convergence.json` | `python scripts/part4_supplement.py` |

`part4.py` intentionally exits non-zero after writing its evidence because the prescribed candidates have no qualifying step. Run the supplemental command separately afterward; it reuses the retained reference and does not change the null prescribed choice.

## Additional viewer evidence

```sh
python scripts/viewer_copy.py
```

The script samples the existing `N=128` random recording at every second point in both directions; it does not run another simulation. The generated `fields.jsonl` has 101 frames with 4096 vorticity values per frame and is 3,884,872 bytes. Local viewer frames at `t=0,2,5,10` show small vortices evolving and merging into fewer, larger structures while enstrophy falls. The public viewer was checked in an incognito window: it loaded all 101 frames from GitHub and displayed intermediate-time vorticity fields. The five `viewer-*.png` files are manual exports: load `fields.jsonl` and export the four random times, and load `artifacts/taylor-green/fields.jsonl` and export `t=1` for the Taylor–Green image. These files are retained from the earlier sheet and do not by themselves complete a numbered Part in the current sheet.

# Week 2 Part 3 Dimer Implementation Plan

> **For agentic workers:** Execute the checked steps in order, inline. The student has already approved this design and requested implementation without another execution-choice pause.

**Goal:** Simulate the Lennard–Jones dimer with two integrators, verify the specified energy behavior, and provide an example that directly generates a two-panel PNG.

**Architecture:** A focused Rust dimer module owns state, pair force, energy, the shared integrator trait, and a common observation runner. A Rust example streams the three runs to a Python Matplotlib script that writes `week2/dimer.png` by default. The final image is generated, inspected, and committed by the student.

**Tech Stack:** Rust 2024 edition and standard library; Python 3 with Matplotlib for plotting. No new Rust dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-week2-dimer-design.md`

## Global constraints

- Work only in Week 2; do not stage Week 1 caches or Week 3 runs.
- Two atoms: mass 1; positions `(0,0)`, `(1.2,0)`; zero velocities; open boundaries; untruncated Lennard–Jones with `epsilon = sigma = 1`.
- Use one Rust `Integrator` trait for forward Euler and velocity-Verlet; `dt = 0.01`.
- Run Euler 500, Verlet 500, and a fresh Verlet 5000 steps. Record step 0 and every step.
- Record signed `delta_n = (E_n - E_0) / |E_0|`. Take absolute values only to calculate maximum Verlet error.
- Require `max |delta_n| < 1e-3` for Verlet 500 and `delta_500 > 0.5` for Euler 500. Inspect the full Verlet 5000 trace for bounded oscillation and no sustained drift; impose no `1e-3` bound on that long run.
- `cargo run --manifest-path md/Cargo.toml --example dimer` from `week2/` must directly generate `week2/dimer.png`. The left panel compares both signed 500-step errors with a legend and the title “Euler vs velocity-Verlet · 500 steps”; the right panel shows all 5000 Verlet steps with signed error multiplied by 1000.
- Do not run the default image command or commit `dimer.png`; the student will generate, inspect, and commit it. Do not push.

## File map

- `week2/md/src/dimer.rs`: state, pair forces, energy, integrators, observations, Rust tests.
- `week2/md/src/lib.rs`: expose the dimer module while retaining earlier Lennard–Jones API.
- `week2/md/examples/dimer.rs`: launch the three runs and pipe observations to the plotting script; optional output path supports a temporary smoke test.
- `week2/plot_dimer.py`: validate the three input series and draw the two-panel PNG.
- `week2/test_plot_dimer.py`: exercise the real renderer on controlled data, writing only to a temporary directory.
- `docs/superpowers/specs/2026-09-17-week2-dimer-design.md` and this plan:
  specification and work record.

## Task 1: Dimer physics and common integrator interface

**Interfaces:** `pub struct State { positions: [[f64; 2]; 2], velocities: [[f64; 2]; 2] }`, `pub fn initial_state() -> State`, `pub fn pair_forces(&State) -> [[f64; 2]; 2]`, `pub fn total_energy(&State) -> f64`, `pub trait Integrator { fn step(&self, state: &mut State, dt: f64); }`, `pub struct Euler`, `pub struct VelocityVerlet`.

- [ ] **Write failing Rust tests** in `src/dimer.rs` for the attractive force at separation 1.2, equal and opposite forces, Euler's first-step velocity `0.02211693342223078` for atom 0 with unchanged position, and Verlet's first-step atom-0 position `0.0001105846671111539`. The expected numbers are hand-calculated from the Lennard–Jones formula, independent of production helpers.
- [ ] **Verify red:** `cargo test --manifest-path week2/md/Cargo.toml --lib dimer` must fail because the dimer interface is absent.
- [ ] **Implement minimal physics:** compute `d = x1 - x0`, `r = hypot(dx,dy)`, and `f1 = lennard_jones_radial_force(r) * d/r`; set `f0 = -f1`. Compute kinetic plus one pair potential. Euler updates positions from old velocities and velocities from old forces. Verlet saves old forces, updates positions by `x + v*dt + 0.5*f*dt*dt`, recomputes forces, and updates velocities by `v + 0.5*(f_old + f_new)*dt`.
- [ ] **Verify green:** rerun the focused Rust tests, then `cargo test --manifest-path week2/md/Cargo.toml --lib`.
- [ ] **Review and commit** the tested physics and interface with a focused message.

## Task 2: Signed energy observations and numerical acceptance

**Interfaces:** `pub struct Observation { pub step: usize, pub time: f64, pub energy: f64, pub relative_error: f64 }`; `pub fn run<I: Integrator>(integrator: &I, steps: usize, dt: f64) -> Vec<Observation>`.

- [ ] **Write failing Rust tests** that require `run(&Euler, 500, 0.01)` and `run(&VelocityVerlet, 500, 0.01)` to return 501 observations with zero error at step 0 and signed errors thereafter; check `euler.last().unwrap().relative_error > 0.5` and `verlet.iter().skip(1).map(|o| o.relative_error.abs()).fold(0.0, f64::max) < 1e-3`. Check independent 5000-step Verlet data is finite, bounded, and oscillatory without forcing a `1e-3` threshold. Preserve the full signed series for visual inspection.
- [ ] **Verify red:** run the focused `cargo test --manifest-path week2/md/Cargo.toml --lib dimer` and confirm the missing runner or failed behavior causes the failure.
- [ ] **Implement minimal runner:** create a fresh initial state, calculate `E_0` once, push step 0, then call `step` and push each observation with `(E_n - E_0) / E_0.abs()`.
- [ ] **Verify green:** run the focused and complete Rust tests. If numerical acceptance fails, investigate the physics or integrator before altering a threshold.
- [ ] **Review and commit** the tested observation runner and acceptance tests.

## Task 3: Direct PNG generation from the dimer example

**Interfaces:** `cargo run --manifest-path md/Cargo.toml --example dimer` from `week2/` writes `week2/dimer.png`; `-- /tmp/name.png` overrides the output for smoke testing. The example writes CSV with `run,step,time,relative_error` to the plotting script's standard input. The script accepts the output path as its sole argument.

- [ ] **Write failing Python renderer tests** in `week2/test_plot_dimer.py`. Feed controlled CSV rows for `euler_500`, `verlet_500`, and `verlet_5000`; run the real script with a temporary output path and assert that a nonempty PNG signature is written. Add malformed/missing-series input that must fail without leaving a PNG. Inspect the real figure object to assert that the left panel has both signed series, the legend, and the exact comparison title, while the right panel retains the full signed series scaled by 1000.
- [ ] **Verify red:** run `python3 -m unittest week2/test_plot_dimer.py` in an environment with Matplotlib; failure must identify the missing script or output behavior.
- [ ] **Implement the renderer:** parse and validate the CSV; plot both signed 500-step errors in the left panel with a legend and the title “Euler vs velocity-Verlet · 500 steps”; plot every signed 5000-step Verlet error times 1000 in the right panel; label axes and annotate the 500-step Verlet maximum absolute error; save at the requested output path with the Agg backend.
- [ ] **Verify green:** rerun the Python tests. Keep their output in temporary directories.
- [ ] **Write a failing example integration check** that runs `cargo run --manifest-path md/Cargo.toml --example dimer -- /tmp/<temporary>/dimer.png` from `week2/` and checks the PNG exists; it must fail while the example is absent.
- [ ] **Implement the example:** call the shared runner for 500/500/5000 steps, pipe CSV rows to `python3 week2/plot_dimer.py <output>` using `CARGO_MANIFEST_DIR` to locate the script, check the child exit status, and print the two 500-step acceptance values. Do not silently swallow Python or file errors.
- [ ] **Verify green:** run the temporary-output example smoke test, `cargo test --manifest-path week2/md/Cargo.toml --all-targets`, and `cargo fmt --manifest-path week2/md/Cargo.toml --check`. Confirm `week2/dimer.png` remains absent.
- [ ] **Review and commit** the example, plotter, and tests. Report the measured numerical values and any uncertainty before committing. Leave the final PNG for the student.

## Final self-review

- [ ] Confirm each design requirement maps to a test or explicit inspection step; confirm no `TBD` or `TODO` remains.
- [ ] Review `git diff` and `git status`, verify only Week 2 files are staged, and do not push.

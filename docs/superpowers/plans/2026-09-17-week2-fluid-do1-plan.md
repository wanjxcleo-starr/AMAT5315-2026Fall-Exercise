# Week 2 Part 4 DO (1) Fluid Command Implementation Plan

> **For agentic workers:** Execute these TDD steps inline. The student approved the incremental design and requested Part 4 DO (1) only; the later reproduce, diagnostics, and video steps are outside this plan.

**Goal:** Make `week2/md/` a velocity-Verlet fluid command that records the specified 100-atom production trajectory.

**Architecture:** Add a fluid module for an owned many-atom state, periodic pair physics, seeded initialization, and equilibration/production stepping. Keep the existing untruncated Lennard–Jones functions and dimer module intact. Add a small recording module and a CLI binary; verify saved JSON independently from Python in a temporary output directory.

**Tech Stack:** Rust 2024 standard library, Cargo, Python 3 standard library for independent verification. No new crate dependencies.

**Spec:** `week2/md.design.toml`; source: Part 4 DO (1) and UNDERSTAND before the Part 5 heading in the Week 2 learning sheet.

## Global constraints

- Two-dimensional Lennard–Jones units: `epsilon = sigma = m = kB = 1`; minimum-image periodic boundaries; potential shifted to zero at `rc = 2.5` and zero for `r >= rc`; velocity-Verlet.
- For the contract run: `n = 100`, `rho = 0.8`, `temperature = 0.5`, `dt = 0.01`, `eq_steps = 2000`, `steps = 10000`, `sample_every = 50`, `seed = 2026`, `out = artifacts`.
- Require all nine CLI values, no physics defaults, no subcommand. Usage errors exit 2; runtime errors exit 1; diagnostics go to stderr; successful stdout is a tab-separated header and one line per saved frame.
- Keep full precision for integration. Save positions and velocities with 9 significant digits and recompute saved `E_pot` and `E_kin` from those serialized values. Save production steps divisible by `sample_every`, excluding zero.
- Write compact `<out>/run.json` and `<out>/traj.jsonl`; ignore generated `week2/artifacts/`. Preserve all Part 2 and Part 3 APIs, tests, and examples.
- Do not implement `Makefile`, `scripts/check.py`, `scripts/video.py`, or video output in this task. Do not push.

## File map

- `week2/md/src/fluid.rs`: lattice, box, minimum image, shifted pair law, seeded velocities, thermostat, Verlet, unit tests.
- `week2/md/src/record.rs`: nine-digit frame serialization, saved-energy recomputation, metadata and JSONL writers, unit tests.
- `week2/md/src/lib.rs`: expose the two new modules without changing Part 2 or dimer behavior.
- `week2/md/src/main.rs`: required-flag parser, validation, equilibration/production loop, outputs and exit codes.
- `week2/md/tests/md_command.rs`: command integration tests in temporary directories.
- `week2/.gitignore`: ignore generated `artifacts/` and Python cache for this week.
- `week2/md.design.toml`: existing green-panel specification to include with this deliverable.

### Task 1: Periodic lattice and shifted pair physics

**Interfaces:** `Box2 { lx, ly }`, `FluidState { pos: Vec<[f64;2]>, vel: Vec<[f64;2]>, box2: Box2 }`, `minimum_image(d, length)`, `wrapped(x, length)`, `shifted_pair_energy(r)`, `forces_and_potential(&FluidState)`.

- [ ] Write failing tests for a 10×10 staggered lattice at `rho=0.8`, periodic seam distance, equal-and-opposite force, and `U_cut(2.5)=0`. For example, a pair with horizontal difference `9.8` in a length-10 box has nearest-image difference `-0.2`.
- [ ] Run `cargo test --manifest-path week2/md/Cargo.toml --lib fluid` and record the expected missing-interface failure.
- [ ] Implement `a = sqrt(2 / (sqrt(3)*rho))`, `h = sqrt(3)*a/2`, `Lx = side*a`, `Ly = side*h`, with a half-row stagger on odd rows. For each unordered pair, use `d - L*round(d/L)`, and for `r<2.5` add `U(r)-U(2.5)` once and `F(r) d/r` with equal/opposite atom forces. Wrap with Euclidean remainder.
- [ ] Rerun focused and existing Part 2/3 Rust tests; fix physics failures before proceeding.

### Task 2: Seeded initialization, thermostat, and Verlet

**Interfaces:** `FluidState::new(n, rho, temperature, seed)`, `kinetic_energy(&FluidState)`, `thermostat_temperature(&FluidState)`, `rescale_to_temperature(&mut FluidState, target)`, `velocity_verlet_step(&mut FluidState, dt)`.

- [ ] Write failing tests that the same seed produces identical finite velocities; their mean is zero within floating precision; the initial rescale reaches `T_thermo = 2 E_kin/(2n-2)`; and one Verlet step keeps the pair force sum and total momentum near zero while positions remain wrapped.
- [ ] Run the focused tests to verify the missing behavior fails.
- [ ] Implement a fixed SplitMix64 uniform generator and Box–Muller Gaussian pairs of variance `temperature`; subtract both velocity means, then rescale by `sqrt(target/T_thermo)`. Apply the thermostat initially and after equilibration steps 50, 100, …; apply no thermostat during production. Use old and new pair forces in velocity-Verlet.
- [ ] Rerun focused and complete Rust tests, including existing dimer tests.

### Task 3: Nine-digit saved states and energies

**Interfaces:** `SavedFrame::from_state(step, dt, &FluidState)`, `write_run_json(writer, metadata)`, `write_frame_jsonl(writer, &SavedFrame)`.

- [ ] Write failing tests that a saved frame has compact valid JSON, wrapped positions and finite velocities, exactly 9 significant digits for nonzero position/velocity tokens, `t = step*dt`, and stored energies matching an independent recomputation from the parsed saved arrays.
- [ ] Run the focused tests to verify missing serialization fails.
- [ ] Format each saved position and velocity with nine significant digits, parse those decimal tokens back to `f64` for the saved-energy calculation, and write fixed-schema compact JSON. Keep the live integration state untouched. Write metadata with `n`, `rho`, `box`, `dt`, `temperature`, `eq_steps`, `steps`, `sample_every`, `seed`, and `integrator = "velocity-verlet"`.
- [ ] Rerun focused and complete Rust tests. Use Python `json` only as an independent consumer of the output, never as the source of stored energies.

### Task 4: CLI and contract recording

**Interfaces:** `md --n N --rho R --temperature T --dt DT --eq-steps Q --steps P --sample-every S --seed SEED --out DIR`.

- [ ] Write failing integration tests for a short `n=100` run: missing/unknown/duplicate flags produce exit 2 and stderr only; a bad output path produces exit 1; a 20-production-step run sampled every 5 steps writes four JSONL frames at steps 5, 10, 15, 20 and five TSV stdout lines including the header.
- [ ] Run `cargo test --manifest-path week2/md/Cargo.toml --test md_command` and confirm it fails because the current binary is only the greeting.
- [ ] Implement parsing and validation: all flags required; positive finite `rho`, `temperature`, `dt`; positive `steps` and `sample_every`; nonnegative `eq_steps`; even square lattice side and `2*rc < min(Lx,Ly)`. Create the output directory, run equilibration, then reset the production step count and save only divisible steps. Print only saved-frame TSV data to stdout.
- [ ] Rerun integration tests, all-target Cargo tests, `cargo fmt --check`, and `git diff --check`. Confirm Part 2/3 examples still build.

### Task 5: Full DO (1) verification and review

- [ ] Run the exact green-panel flag values from `week2/` in release mode with `--out artifacts`, without using `make`.
- [ ] Independently parse `run.json` and all 200 JSONL frames from Python in a temporary one-off verification command. Check required keys, step sequence, wrapped coordinates, 9-digit tokens, TSV/JSON agreement, and recomputed shifted potential and kinetic energy. Report measured recording counts, energy differences, and any numerical uncertainty; do not add a formal diagnostics script yet.
- [ ] Review only the DO (1) files, confirm artifacts and caches are excluded, display the exact staged list and `git diff --cached --check`, then commit the reviewed files. Do not push.

## Execution notes

- `cargo test --manifest-path week2/md/Cargo.toml --all-targets`: 16 library tests and 3 command tests passed; the existing dimer and Part 2 examples still build.
- `cargo fmt --manifest-path week2/md/Cargo.toml --check` and `git diff --check` passed.
- The exact reference run produced 200 frames at production steps 50 through 10000. An independent, temporary Python calculation from saved coordinates and velocities found maximum stored potential and kinetic energy differences of `1.14e-13` and `7.11e-14`; TSV values agreed exactly with JSON. The one-off secular drift was `0.000175295`, `T_speed` was `0.519443`, and `chi2/22` was `0.861455`.
- The learning sheet specifies a seed and Gaussian velocity distribution but no random-number algorithm. This implementation fixes SplitMix64 plus Box–Muller for repeatability. The official diagnostics script and its acceptance gate belong to a later DO step; these one-off values are not a substitute for that step.
- Review found an oversized `--n` overflow that returned 101 in a debug build. A failing command regression test exposed it, and checked multiplication now returns usage code 2.

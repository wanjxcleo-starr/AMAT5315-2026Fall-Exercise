# Week 2 Part 3: Lennard–Jones dimer design

## Physical model

Simulate two atoms of mass 1 in two dimensions. Their initial positions are
`(0, 0)` and `(1.2, 0)`, and both initial velocities are zero. Use open
boundaries and the existing unshifted, untruncated Lennard–Jones pair energy
and radial force with `epsilon = sigma = 1`. The time step is `0.01`.

For separation vector `d = position[1] - position[0]`, atom 1 receives
`F(r) d/r` and atom 0 receives its negative. The total energy is the sum of
both kinetic energies and the pair potential, counted once. The initial
energy `E_0` is nonzero.

## Integrators and observations

Both forward Euler and velocity-Verlet implement one Rust `Integrator` trait
and use the same force calculation. Forward Euler evaluates both updates
from the old state. Velocity-Verlet updates positions using old acceleration,
then reevaluates force before updating velocities. Each run starts from an
independent copy of the initial state.

Run Euler for 500 steps, Verlet for 500 steps, and Verlet independently for
5000 steps. Record step 0 and every completed step. Each observation includes
step, time, total energy, and the **signed** relative total energy error

`delta_n = (E_n - E_0) / |E_0|`.

Never take an absolute value when recording or plotting `delta_n`. Use
`max |delta_n|` only when reporting the maximum Verlet error.

## Acceptance and figure

- Over steps 1–500, Verlet's maximum absolute relative error is below `1e-3`.
- Euler's signed relative error at step 500 is above `0.5`.
- The full 5000-step Verlet error remains bounded and oscillatory, without
  sustained drift. This is checked from the signed time series and its plot;
  no `1e-3` hard threshold applies to the 5000-step run.

The Rust example `week2/md/examples/dimer.rs` runs all three simulations and
directly creates `week2/dimer.png` when invoked from `week2/` with
`cargo run --manifest-path md/Cargo.toml --example dimer`. It passes the
observations to a Week 2 Python plotting script, following the existing
`field_data`/`plot_field.py` pattern. The left panel shows the signed Euler
error through step 500. The right panel shows the **complete** 5000-step
signed Verlet error multiplied by 1000, with the 500-step maximum absolute
error annotated. The script requires Matplotlib. An optional output path
allows a temporary-image smoke test without creating the final image.

The final `dimer.png` will be generated, inspected, and committed by the
student. Codex may commit the implementation, tests, design, and plan but
must not generate or commit the final image or push commits.

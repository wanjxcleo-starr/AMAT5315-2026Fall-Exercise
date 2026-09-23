# Week 4 Part 1: spectral flow solver

This Rust crate implements the Part 1 solver on `[0,2π)²`. `field` writes Taylor–Green or seeded random velocity fields; `fluid` integrates their vorticity with Euler, explicit-midpoint RK2, or classical RK4. Parts 2 onward, the Extension, and the Challenge are not included.

`N` is a power of two and arrays use index `iy*N + ix`. The two-dimensional inverse FFT is normalized once by `1/N²`. First derivatives zero the even-grid Nyquist line to preserve real-field symmetry. The Poisson solve sets `ψ̂(0)=0`; velocity recovery separately sets `û(0)=v̂(0)=0`. The rectangular two-thirds mask is applied to the initial vorticity, every integrator stage, and the transformed nonlinear term. Each stage recovers its own velocity.

Energy and enstrophy are `E=mean((u²+v²)/2)` and `Z=mean(ω²/2)`. Random fields use resolution-independent seeded phases and are normalized to `E(0)=0.5`. `t_end/dt` must be an integer within floating-point tolerance; no step is silently changed. Snapshot arrays in `fields.jsonl` have six decimal places.

## Automated checks

From `week4/`:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

The current run passes all 17 tests, including the three Part 1 numerical tests. The release Taylor–Green run ended with `E=0.167580011509` and `Z=0.335160023018`; the exact energy is `0.25*exp(-0.4)=0.1675800115089...`. The internal, unrounded velocity passes the `1e-10` divergence bound.

## Part 1 Taylor–Green run

The resource contract adds the explicit `--method rk4` argument to the learning-sheet command:

```sh
cargo install --path . --quiet
mkdir -p evidence
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
      --out artifacts/taylor-green \
  | tee evidence/taylor-green.txt
```

This writes `run.json` and `fields.jsonl` under the ignored `artifacts/taylor-green/` directory. The course-viewer inspection and saved PNG remain a separate, user-run VERIFY step.

# AMAT5315 Modern Scientific Computing exercises

This repository records weekly coursework in scientific computing, with specifications, code, tests, and numerical evidence that can be checked from the repository root.

## Repository layout

- `week1/` contains a Python Monte Carlo estimator of π, its specification, and a pytest test.
- `week2/md/` contains a Rust crate with a greeting library function, a small executable, and a unit test.
- `week3/` contains a Rust Ising model simulation, its command design, and a plotting script and figure under `evidence/`.

## Environment and dependencies

`setup.txt` records Python 3.14.4 and Codex CLI 0.154.0. Week 1 uses the Python standard library for the estimator, `pytest` for its test, and `pypdf` for the tutor skill's PDF extraction. Create a virtual environment outside the repository and install both dependencies:

```sh
python3 -m venv ~/.venvs/amat5315
source ~/.venvs/amat5315/bin/activate
python3 -m pip install pytest pypdf
```

Weeks 2 and 3 require Rust and Cargo with support for the 2024 edition; their `Cargo.toml` files declare no third-party Rust dependencies. The local environment has Rust and Cargo 1.98.1. To regenerate the Week 3 evidence plot, also install Matplotlib with `python3 -m pip install matplotlib`; the plotting script reads data from `week3/runs/`.

## Reproduce the Week 1 result

From the repository root, with the virtual environment active, run the test:

```sh
python3 -m pytest week1/
```

Run the seeded estimate and print its absolute error against `math.pi`:

```sh
python3 - <<'PY'
import math
from week1.pi import estimate_pi

estimate = estimate_pi(1_000_000, seed=2026)
print(f"estimate={estimate:.6f}")
print(f"absolute error={abs(estimate - math.pi):.6f}")
PY
```

The verified estimate is **3.146604**, with absolute error **0.005011**. The Week 1 test passes.

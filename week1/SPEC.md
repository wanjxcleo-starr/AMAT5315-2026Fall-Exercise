# Week 1 Pi Estimator Specification

The function `estimate_pi(n, seed)` estimates pi by generating `n` uniformly distributed random points in the unit square from `(0, 0)` to `(1, 1)`, counting the points whose distance from the origin is at most 1, and returning four times the fraction of points inside this quarter circle. The `seed` parameter must make the calculation reproducible, so calls with the same `n` and `seed` produce the same result. Correctness is defined by the following assertion:

```python
abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2
```


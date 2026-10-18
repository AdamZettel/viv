---
title: Newton's method
order: 1
output: identical
---
- Python's `tol=1e-12, max_iter=50` defaults become named constants in Viv,
  which has no default arguments.
- The functions are passed as lambdas in both: `lambda x: x * x - 2.0` is
  `fn(x) -> x * x - 2.0`.
- `raise RuntimeError(...)` is `stop("...")`: it ends the run with
  `error: Newton did not converge` and the line, and there is no way to catch
  it. Both programs converge, so neither line runs.
- `cos` and `sin` are builtins in Viv; Python takes them from `math`. The values
  are printed to 12 decimals, which Viv's own libm and the platform's agree on.
- `assert abs(a - b) < tol` is `assert_near(a, b, tol)`, which prints the gap
  when it fails.

---
title: Gradients by hand and by grad
order: 1
output: identical
---
- The Python gradient is derived by hand: for the mean squared error,
  `2 X^T (X w - y) / n`. The Viv gradient is `grad(loss, w, X, y)`: the compiler
  writes the reverse pass of `loss` itself, and every engine runs that
  generated code. Nothing on the Viv side states the formula.
- Both check the gradient against central differences with `H = 1e-6`; the
  Python side does it in one comprehension over `np.eye(2)`, the Viv side in
  a loop that builds each unit vector (Viv has no `eye` yet).
- `grad(f, w, ...)` differentiates with respect to the first argument and
  passes the rest through unchanged, and the result has the shape of `w`.
- For one variable, Python writes the derivative `df` out (product rule).
  Viv's `deriv(fn(x) -> sin(x) * x * x, 1.3)` computes it in forward mode.
- `w0.copy()` is `copy(w0)`; `X.T @ r` would be `transpose(X) @ r`, but the Viv
  side never needs it.
- The descent prints the same loss to 8 decimals in both, although the
  gradients are computed by different formulas: they agree to rounding.

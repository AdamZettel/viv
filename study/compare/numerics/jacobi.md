---
title: Jacobi iteration
order: 6
output: identical
---
- A matrix literal is a list of rows in both; Viv needs no `np.array(...)`
  around it.
- `R @ x`, `(b - R @ x) / d` and `abs(x_new - x)` read the same: `@` is the
  matrix product and the rest is elementwise.
- `np.diag(A)` and `A - np.diag(d)` have no builtin twin; the Viv side copies
  `A`, reads the diagonal and zeroes it in one loop. `A[i, i]` indexes a
  matrix with one pair of brackets, as NumPy does.
- `np.max(...)` of an array is `max(...)` of an array.
- `np.allclose(x, ref, atol=1e-9)` is `assert_allclose(x, ref, 1e-9)`.
- `" ".join(f"{v:.8f}" for v in x)` is the same in both.

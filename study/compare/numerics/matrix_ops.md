---
title: Matrix operations
order: 9
output: identical
---
- `np.array([[1.0, 2.0], [3.0, 4.0]])` is the literal `[[1.0, 2.0], [3.0, 4.0]]`.
  `np.eye(2) * 2.0` has no builtin twin; the Viv side writes the matrix out.
- `A @ B` is the matrix product and `A * B` the elementwise product in both.
- `A.T` is `transpose(A)`; `A.shape` is `shape(A)`, which unpacks into
  `rows, cols`; `A.size` is `numel(A)`.
- `A.sum()` is `sum(A)`. There is no `axis=` argument, so row and column sums
  are loops.
- `A > 1.5` gives an array of 1.0 and 0.0 in Viv (booleans in NumPy); both
  multiply the same way.
- `sqrt` applies to every element of an array, as `np.sqrt` does. Viv's
  `round` takes no digit count, so rounding to 6 decimals is
  `round(C * 1e6) / 1e6`.
- `tolist()` on the Python side and `fmt(...)` on the Viv side print a matrix
  the same way, `[[1.0, 2.0], [3.0, 4.0]]`.

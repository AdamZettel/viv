---
title: LU factorization and solve
order: 8
output: identical
---
- Swapping two rows: NumPy's fancy index `LU[[k, p]] = LU[[p, k]]` is Viv's
  `LU[k, :], LU[p, :] = LU[p, :], LU[k, :]`. The right side is computed (and
  copied) before either row is written.
- `LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]` is the same statement in both: a
  compound assignment into a slice.
- `k + np.argmax(np.abs(LU[k:, k]))` is
  `max([i for i from k to n], key=fn(i) -> abs(LU[i, k]))` in Viv: the first
  row with the largest entry, as `argmax` picks.
- A dot product of two slices, `LU[i, :i] @ y[:i]`, is `sum(LU[i, :i] * y[:i])`.
- `b[perm]` (indexing with an array) is the comprehension `[b[j] for j in perm]`
  in Viv. `perm` holds doubles, and an index that is a double is read as an
  integer.
- `np.linalg.solve(A, b)` returns the solution; `lin_solve(A, b, x)` writes it
  into an array the caller made.
- `for i in reversed(range(n))` is `for i to n reversed`.

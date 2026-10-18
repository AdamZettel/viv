---
title: Power iteration
order: 7
output: identical
---
- `w @ w` (a dot product of two vectors) is `sum(w * w)` in Viv; `@` there is
  the matrix product, and a vector times a vector is elementwise.
- `np.ones(n)` is `zeros(n) + 1.0`.
- `np.linalg.eigh(A)` returns two arrays; Viv's `eig_sym(A, vals, vecs)` fills
  two arrays the caller made, with the eigenvalues ascending as `eigh` gives
  them. It uses cyclic Jacobi, not LAPACK.
- `vals[-1]`, `" ".join(f"{x:.6f}" for x in v)` and the if-expression in the
  last print are the same in both.
- The last bits of the eigenvalue depend on the order of the sums (NumPy calls
  BLAS), so both programs print ten decimals and whether the two methods agree
  to 1e-12, not the size of the difference.

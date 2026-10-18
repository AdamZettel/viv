# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ LU factorization with partial pivoting
def lu_factor(A):
    n = A.shape[0]
    LU = A.copy()
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(LU[k:, k])))
        if p != k:
            LU[[k, p]] = LU[[p, k]]
            perm[[k, p]] = perm[[p, k]]
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]
    return LU, perm

# @@ forward and back substitution
def lu_solve(LU, perm, b):
    n = len(b)
    y = b[perm].astype(float)
    for i in range(n):
        y[i] -= LU[i, :i] @ y[:i]
    x = y.copy()
    for i in reversed(range(n)):
        x[i] = (x[i] - LU[i, i + 1:] @ x[i + 1:]) / LU[i, i]
    return x

# @@ solve a system
A = np.array([[2.0, 1.0, 1.0],
              [4.0, -6.0, 0.0],
              [-2.0, 7.0, 2.0]])
b = np.array([5.0, -2.0, 9.0])
LU, perm = lu_factor(A)
x = lu_solve(LU, perm, b)
print("pivot order", perm.tolist())
print("x =", x.tolist())

# @@ against the library solver
x_lib = np.linalg.solve(A, b)
print("library x =", x_lib.tolist())

# @@ checks
assert np.allclose(A @ x, b, atol=1e-12)
assert np.allclose(x, x_lib, atol=1e-12)

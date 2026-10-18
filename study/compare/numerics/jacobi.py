# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ a diagonally dominant system
A = np.array([[10.0, -1.0, 2.0, 0.0],
              [-1.0, 11.0, -1.0, 3.0],
              [2.0, -1.0, 10.0, -1.0],
              [0.0, 3.0, -1.0, 8.0]])
b = np.array([6.0, 25.0, -11.0, 15.0])

# @@ Jacobi iteration: x_new = D^-1 (b - R x)
def jacobi(A, b, tol, max_iter):
    d = np.diag(A)
    R = A - np.diag(d)
    x = np.zeros(len(b))
    for k in range(max_iter):
        x_new = (b - R @ x) / d
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# @@ solve and check the residual
TOL = 1e-10
MAX_ITER = 200
x, iters = jacobi(A, b, TOL, MAX_ITER)
print("iterations:", iters)
print("x =", " ".join(f"{v:.8f}" for v in x))
print(f"largest residual {np.max(np.abs(A @ x - b)):.1e}")

# @@ checks
assert np.allclose(x, [1.0, 2.0, -1.0, 1.0], atol=1e-9)

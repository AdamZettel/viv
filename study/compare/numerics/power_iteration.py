# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ a symmetric matrix
A = np.array([[4.0, 1.0, 0.0],
              [1.0, 3.0, 1.0],
              [0.0, 1.0, 2.0]])

# @@ power iteration with the Rayleigh quotient
def power_iteration(A, iters):
    v = np.ones(A.shape[0])
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        v = w / np.sqrt(w @ w)
        lam = v @ A @ v
    return lam, v

ITERS = 60
lam, v = power_iteration(A, ITERS)
print(f"largest eigenvalue {lam:.10f}")
print("eigenvector", " ".join(f"{x:.6f}" for x in v))

# @@ against the library eigensolver
vals, vecs = np.linalg.eigh(A)
print("all eigenvalues", " ".join(f"{x:.10f}" for x in vals))
AGREE = 1e-12
print("agrees with eigh to 1e-12:", "yes" if abs(lam - vals[-1]) < AGREE else "no")

# @@ checks
assert np.allclose(A @ v, lam * v, atol=1e-8)

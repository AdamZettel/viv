# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ build matrices
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.eye(2) * 2.0
v = np.array([1.0, -1.0])
print("A =", A.tolist())
print("shape", A.shape, "size", A.size)

# @@ products
print("A @ B =", (A @ B).tolist())
print("A * B =", (A * B).tolist())
print("A @ v =", (A @ v).tolist())
print("A^T   =", A.T.tolist())

# @@ reductions along an axis
print("sum", A.sum(), "column sums", A.sum(axis=0).tolist(), "row sums", A.sum(axis=1).tolist())

# @@ elementwise functions and masks
C = np.sqrt(A) * (A > 1.5)
print("C =", np.round(C, 6).tolist())

# @@ determinant and inverse of a 2 by 2
det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
inv = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
print("det", det, "inverse", inv.tolist())

# @@ checks
assert np.allclose(A @ inv, np.eye(2))

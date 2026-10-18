# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-jacobi-eigenvalue-sweeps.viv.

The same cyclic Jacobi sweeps in Python, and numpy.linalg.eigh on the same
matrix. Run: python3 posts/refs/jacobi-eigenvalue-sweeps.py > posts/refs/jacobi-eigenvalue-sweeps.out
"""
import numpy as np

N = 8
A0 = np.array([[1.0 / (1 + i + j) + (1 + 0.5 * i if i == j else 0.0) for j in range(N)] for i in range(N)])


def offnorm(A):
    return np.sqrt(np.sum((A - np.diag(np.diag(A))) ** 2))


def rotate(A, V, p, q):
    theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])
    c, s = np.cos(theta), np.sin(theta)
    for k in range(N):
        akp, akq = A[k, p], A[k, q]
        A[k, p], A[k, q] = c * akp + s * akq, -s * akp + c * akq
    for k in range(N):
        apk, aqk = A[p, k], A[q, k]
        A[p, k], A[q, k] = c * apk + s * aqk, -s * apk + c * aqk
    for k in range(N):
        vkp, vkq = V[k, p], V[k, q]
        V[k, p], V[k, q] = c * vkp + s * vkq, -s * vkp + c * vkq


A = A0.copy()
V = np.eye(N)
off = [offnorm(A)]
for sweep in range(6):
    for p in range(N):
        for q in range(p + 1, N):
            rotate(A, V, p, q)
    off.append(offnorm(A))
print("off-norm after each sweep:")
for k, o in enumerate(off):
    print(f"  {k}  {o!r}")
ev = np.linalg.eigvalsh(A0)
print("numpy.linalg.eigh(A0) eigenvalues:")
for e in ev:
    print(f"  {float(e)!r}")
print("smallest gap:", repr(float(np.min(np.diff(ev)))))
print("max |sorted diag(A) - eigh|:", repr(float(np.max(np.abs(np.sort(np.diag(A)) - ev)))))
print("max |V diag V^T - A0|:", repr(float(np.max(np.abs(V @ np.diag(np.diag(A)) @ V.T - A0)))))

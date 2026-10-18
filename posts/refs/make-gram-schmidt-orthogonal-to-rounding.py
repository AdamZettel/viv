# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-make-gram-schmidt-orthogonal-to-rounding.viv (a problem).

The Lauchli vectors a_k = e_0 + eps e_{k+1}, k = 0..9, in 11 dimensions, with
eps = 1e-8. The exact orthonormal vectors are computed by Gram-Schmidt in
mpmath at 60 digits, where rounding does not matter; the double-precision
variants (classical, modified, classical applied twice) in numpy show how far
each is from orthogonal.

    python3 posts/refs/make-gram-schmidt-orthogonal-to-rounding.py > posts/refs/make-gram-schmidt-orthogonal-to-rounding.out
"""
import mpmath as mp
import numpy as np

N, D, EPS = 10, 11, 1e-8

A = np.zeros((N, D))
A[:, 0] = 1.0
for k in range(N):
    A[k, k + 1] = EPS
print(f"condition number of the vectors: {np.linalg.cond(A):.4e}")


def cgs(A, passes=1):
    Q = np.zeros_like(A)
    for k in range(len(A)):
        a = A[k].copy()
        for _ in range(passes):
            dots = Q[:k] @ a
            a = a - dots @ Q[:k]
        Q[k] = a / np.linalg.norm(a)
    return Q


def mgs(A):
    Q = np.zeros_like(A)
    for k in range(len(A)):
        a = A[k].copy()
        for j in range(k):
            a = a - (Q[j] @ a) * Q[j]
        Q[k] = a / np.linalg.norm(a)
    return Q


for name, Q in (("classical", cgs(A)), ("modified", mgs(A)), ("classical, twice", cgs(A, 2))):
    print(f"{name:17s} max |q_i . q_j - delta_ij| = {np.max(np.abs(Q @ Q.T - np.eye(N))):.4e}")

mp.mp.dps = 60
print(f"exact Gram-Schmidt in mpmath at {mp.mp.dps} digits")
eps = mp.mpf(EPS)  # the double nearest 1e-8, as the program has it
Am = [[mp.mpf(0)] * D for _ in range(N)]
for k in range(N):
    Am[k][0] = mp.mpf(1)
    Am[k][k + 1] = eps
Qm = []
for k in range(N):
    a = list(Am[k])
    for q in Qm:
        dot = mp.fsum(q[i] * a[i] for i in range(D))
        a = [a[i] - dot * q[i] for i in range(D)]
    nr = mp.sqrt(mp.fsum(x * x for x in a))
    Qm.append([x / nr for x in a])
for k in (0, 1, N - 1):
    print(f"exact q_{k} =", "[" + ", ".join(mp.nstr(x, 17) for x in Qm[k]) + "]")
Qd = cgs(A, 2)
print(f"classical twice vs exact: max entry difference {max(abs(float(Qm[k][i]) - Qd[k, i]) for k in range(N) for i in range(D)):.3e}")

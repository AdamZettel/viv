# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-lanczos-loses-orthogonality.viv.

The same matrix (a known diagonal hidden by three Householder reflections, from
the same linear congruential generator), the same starting vector and the same
two Lanczos runs, in numpy; numpy.linalg.eigvalsh gives the Ritz values and
checks the matrix's spectrum. Rounding differs from Viv's in the last bits, so
where orthogonality is lost the two agree in size and timing, not digit for digit.
Run: python3 posts/refs/lanczos-loses-orthogonality.py > posts/refs/lanczos-loses-orthogonality.out
"""
import numpy as np

N, STEPS, TOP, NEXT = 200, 60, 4.0, 3.0
seed = [12345.0]


def next_uniform():
    seed[0] = (1103515245.0 * seed[0] + 12345.0) % 2147483648.0
    return seed[0] / 1073741824.0 - 1.0


def random_unit(n):
    v = np.array([next_uniform() for _ in range(n)])
    return v / np.linalg.norm(v)


eigs = np.concatenate([[-TOP, -NEXT], np.linspace(-1.0, 1.0, N - 4), [NEXT, TOP]])
A = np.diag(eigs)
u = np.zeros(N)
u[-1] = 1.0
for r in range(3):
    v = random_unit(N)
    H = np.eye(N) - 2.0 * np.outer(v, v)
    A = H @ A @ H
    u = u - 2.0 * v * (v @ u)
q0 = random_unit(N)
print("eigvalsh(A) ends:", eigs[:2], "...", eigs[-2:], " max error vs the diagonal:",
      f"{np.max(np.abs(np.linalg.eigvalsh(A) - eigs)):.1e}")
print("|A u - 4 u| =", f"{np.linalg.norm(A @ u - TOP * u):.1e}")
print("A[0, 0], A[0, 1] =", repr(float(A[0, 0])), repr(float(A[0, 1])))


def lanczos(reorth):
    Q = np.zeros((STEPS, N))
    alpha, beta = np.zeros(STEPS), np.zeros(STEPS)
    Q[0] = q0
    along, overlap, ritz = [], [0.0], []
    for j in range(STEPS):
        along.append(abs(Q[j] @ u))
        if j > 0:
            overlap.append(np.max(np.abs(Q[:j] @ Q[j])))
        w = A @ Q[j]
        if j > 0:
            w -= beta[j - 1] * Q[j - 1]
        alpha[j] = Q[j] @ w
        w -= alpha[j] * Q[j]
        if reorth:
            for i in range(j + 1):
                w -= (Q[i] @ w) * Q[i]
        T = np.diag(alpha[:j + 1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1)
        ritz.append(np.linalg.eigvalsh(T))
        if j < STEPS - 1:
            beta[j] = np.linalg.norm(w)
            Q[j + 1] = w / beta[j]
    return np.array(along), np.array(overlap), ritz


for reorth in (False, True):
    along, overlap, ritz = lanczos(reorth)
    r = ritz[-1]
    print("full reorthogonalization" if reorth else "plain three-term recurrence")
    print(f"  Ritz ends after {STEPS} steps: {r[0]!r}  {r[-1]!r}; errors {abs(r[0] + TOP):.1e} {abs(r[-1] - TOP):.1e}")
    print(f"  three largest: {r[-1]!r} {r[-2]!r} {r[-3]!r}")
    print(f"  Ritz values within 1e-8 of 4: {int(np.sum(np.abs(r - TOP) < 1e-8))}")
    conv = next(j for j in range(STEPS) if abs(ritz[j][-1] - TOP) < 1e-14)
    print(f"  first step with the largest Ritz value within 1e-14 of 4: {conv}")
    print(f"  largest overlap of a new vector with an old one: {np.max(overlap):.3g}")
    d1 = int(np.argmin(along[:30]))
    r1 = d1 + int(np.argmax(along[d1:30]))
    d2 = r1 + int(np.argmin(along[r1:]))
    r2 = d2 + int(np.argmax(along[d2:]))
    print(f"  |q_j . u|: first minimum at step {d1} ({along[d1]:.2e}), back up at step {r1} ({along[r1]:.3g}),"
          f" again at step {r2} ({along[r2]:.3g}); max from step 25 on {np.max(along[25:]):.1e}")

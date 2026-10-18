# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-the-error-follows-the-condition-number.viv.

Part 1. H_n = L * hilbert(n), L = lcm(1..2n-1): integer entries L/(i+j+1),
exact in double, b = H_n * ones (integers, exact), so the stored system is the
exact one and its solution is exactly ones (checked below with fractions).
Solved by Gaussian elimination with partial pivoting, written out in Python
with the post's operation order, and by numpy (LAPACK). kappa_2 by numpy.linalg.cond
and, as the exact value, by mpmath eigenvalues at 80 digits.

Part 2. Least squares fit of a degree-d polynomial to m = 40 points t_i = i/39,
Vandermonde V[i][k] = t_i^k, data y_i = sum_k V[i][k] (so the coefficients are
all 1). Householder QR (the post's order of operations) and the normal
equations (V^T V c = V^T y by the same elimination), against the exact least
squares solution of the stored float data, computed in rationals with fractions.
kappa_2(V) by numpy and by mpmath's SVD.
Run: python3 posts/refs/the-error-follows-the-condition-number.py > posts/refs/the-error-follows-the-condition-number.out
"""
import math
from fractions import Fraction as F

import mpmath as mp
import numpy as np

EPS = 2.0 ** -52
mp.mp.dps = 80


def gepp(A, b):
    n = len(b)
    A = [r[:] for r in A]
    b = b[:]
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i
        A[k], A[p] = A[p], A[k]
        b[k], b[p] = b[p], b[k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s = s - A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


def exact_solve(A, b):
    n = len(b)
    A = [[F(v) for v in r] for r in A]
    b = [F(v) for v in b]
    for k in range(n):
        p = next(i for i in range(k, n) if A[i][k] != 0)
        A[k], A[p] = A[p], A[k]
        b[k], b[p] = b[p], b[k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    x = [F(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


print("Part 1: scaled Hilbert systems, exact solution all ones")
print(f"(kappa (mpmath): eigenvalues at {mp.mp.dps} digits)")
cond_off = {}
print(" n  L             kappa (mpmath)          kappa (numpy)   error GEPP              error/(kappa eps)  error numpy LAPACK  residual/(|H||x|)")
for n in range(2, 13):
    L = 1
    for d in range(1, 2 * n):
        L = L * d // math.gcd(L, d)
    H = [[float(L // (i + j + 1)) for j in range(n)] for i in range(n)]
    b = [0.0] * n
    for i in range(n):
        for j in range(n):
            b[i] = b[i] + H[i][j]
    assert all(b[i] == sum(L // (i + j + 1) for j in range(n)) for i in range(n))
    assert all(v == 1 for v in exact_solve(H, b))
    x = gepp(H, b)
    err = max(abs(v - 1.0) for v in x)
    xn = np.linalg.solve(np.array(H), np.array(b))
    errn = np.abs(xn - 1).max()
    ev = mp.eigsy(mp.matrix([[mp.mpf(v) for v in r] for r in H]))[0]
    kmp = float(max(ev) / min(ev))
    knp = np.linalg.cond(np.array(H))
    cond_off[n] = knp / kmp - 1.0
    Hn = np.array(H)
    res = np.abs(np.array(b) - Hn @ np.array(x)).max() / (np.abs(Hn).sum(1).max() * max(abs(v) for v in x))
    print(f"{n:2d}  {L:<12d}  {kmp!r:<22}  {knp:.6e}    {err!r:<22}  {err / (kmp * EPS):.4f}             {errn:.3e}           {res / EPS:.3f} eps")

print(f"numpy cond against mpmath: largest relative difference through n = 10 is {max(abs(cond_off[n]) for n in range(2, 11)):.1e}; at n = 12 it is off by {100 * abs(cond_off[12]):.2f}%")
print()
print("Part 2: least squares, m = 40, t_i = i/39, coefficients all 1")
print(" d   kappa(V) mpmath          kappa(V) numpy  QR error                NE error                QR/(kappa eps)  NE/(kappa^2 eps)  |exact LS - 1|")
M = 40
for d in range(3, 14):
    n = d + 1
    t = [i / (M - 1) for i in range(M)]
    V = [[1.0] * n for _ in range(M)]
    for i in range(M):
        for k in range(1, n):
            V[i][k] = V[i][k - 1] * t[i]
    y = [0.0] * M
    for i in range(M):
        for k in range(n):
            y[i] = y[i] + V[i][k]
    # Householder QR, the post's order of operations
    A = [r[:] for r in V]
    z = y[:]
    for k in range(n):
        nrm = 0.0
        for i in range(k, M):
            nrm = nrm + A[i][k] * A[i][k]
        nrm = math.sqrt(nrm)
        alpha = -nrm if A[k][k] >= 0 else nrm
        v = [A[i][k] for i in range(k, M)]
        v[0] = v[0] - alpha
        vv = 0.0
        for q in v:
            vv = vv + q * q
        for j in range(k, n):
            s = 0.0
            for i in range(k, M):
                s = s + v[i - k] * A[i][j]
            f = 2.0 * s / vv
            for i in range(k, M):
                A[i][j] = A[i][j] - f * v[i - k]
        s = 0.0
        for i in range(k, M):
            s = s + v[i - k] * z[i]
        f = 2.0 * s / vv
        for i in range(k, M):
            z[i] = z[i] - f * v[i - k]
    c = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = z[i]
        for j in range(i + 1, n):
            s = s - A[i][j] * c[j]
        c[i] = s / A[i][i]
    # normal equations
    G = [[0.0] * n for _ in range(n)]
    r = [0.0] * n
    for a in range(n):
        for bb in range(n):
            for i in range(M):
                G[a][bb] = G[a][bb] + V[i][a] * V[i][bb]
        for i in range(M):
            r[a] = r[a] + V[i][a] * y[i]
    cne = gepp(G, r)
    # exact least squares solution of the stored data, in rationals
    VF = [[F(v) for v in row] for row in V]
    yF = [F(v) for v in y]
    GF = [[sum(VF[i][a] * VF[i][bb] for i in range(M)) for bb in range(n)] for a in range(n)]
    rF = [sum(VF[i][a] * yF[i] for i in range(M)) for a in range(n)]
    ce = exact_solve(GF, rF)
    dev = max(abs(float(v - 1)) for v in ce)
    eqr = max(abs(v - 1.0) for v in c)
    ene = max(abs(v - 1.0) for v in cne)
    sv = mp.svd_r(mp.matrix([[mp.mpf(q) for q in row] for row in V]), compute_uv=False)
    kmp = float(max(sv) / min(sv))
    knp = np.linalg.cond(np.array(V))
    # numpy's LAPACK QR, for comparison
    Q, R = np.linalg.qr(np.array(V))
    cl = np.linalg.solve(R, Q.T @ np.array(y))
    print(f"{d:2d}  {kmp!r:<24} {knp:.6e}    {eqr!r:<22}  {ene!r:<22}  {eqr / (kmp * EPS):.4f}          {ene / (kmp * kmp * EPS):.4f}            {dev:.2e}"
          f"   (LAPACK QR error {np.abs(cl - 1).max():.2e})")

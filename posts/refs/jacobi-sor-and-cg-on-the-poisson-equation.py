# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-jacobi-sor-and-cg-on-the-poisson-equation.viv.

-lap u = 2[x(1-x) + y(1-y)] on the unit square, u = 0 on the edge, five-point
stencil on an n by n interior grid, h = 1/(n+1); exact u = x(1-x) y(1-y).
Jacobi, Gauss-Seidel, SOR at Young's omega and CG, each from zero until
|b - A u|_2 / |b|_2 < 1e-8. Iteration counts, rates and the theory's numbers.
Run: python3 posts/refs/jacobi-sor-and-cg-on-the-poisson-equation.py > posts/refs/jacobi-sor-and-cg-on-the-poisson-equation.out
"""
import math
import numpy as np

TOL = 1e-8


def setup(n):
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    b = h * h * 2.0 * (X * (1 - X) + Y * (1 - Y))
    exact = X * (1 - X) * Y * (1 - Y)
    return h, b, exact


def apply_A(u):
    p = np.pad(u, 1)
    return 4 * u - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def jacobi(b):
    u = np.zeros_like(b)
    hist = [1.0]
    while hist[-1] > TOL:
        p = np.pad(u, 1)
        u = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + b)
        hist.append(np.linalg.norm(b - apply_A(u)) / np.linalg.norm(b))
    return u, hist


def sor(b, omega):
    n = len(b)
    u = np.zeros((n + 2, n + 2))
    bb = np.pad(b, 1)
    hist = [1.0]
    while hist[-1] > TOL:
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                gs = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + bb[i, j])
                u[i, j] += omega * (gs - u[i, j])
        hist.append(np.linalg.norm(b - apply_A(u[1:-1, 1:-1])) / np.linalg.norm(b))
    return u[1:-1, 1:-1], hist


def cg(b):
    u = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.sum(r * r)
    hist = [1.0]
    while hist[-1] > TOL:
        Ap = apply_A(p)
        a = rs / np.sum(p * Ap)
        u = u + a * p
        r = r - a * Ap
        rn = np.sum(r * r)
        hist.append(math.sqrt(rn / np.sum(b * b)))
        p = r + (rn / rs) * p
        rs = rn
    return u, hist


def kappa(h):
    # the extreme eigenvalues of the matrix itself, from numpy, against cot^2(pi h / 2)
    return 1.0 / math.tan(math.pi * h / 2) ** 2


counts = {}
for n in (15, 23, 31):
    h, b, exact = setup(n)
    omega = 2.0 / (1.0 + math.sin(math.pi * h))
    uj, hj = jacobi(b)
    us, hs = sor(b, omega)
    uc, hc = cg(b)
    counts[n] = (len(hj) - 1, len(hs) - 1, len(hc) - 1)
    print(f"n = {n}, h = 1/{n + 1}:")
    print(f"  Jacobi {len(hj) - 1} sweeps (ln tol / ln cos(pi h) = {math.log(TOL) / math.log(math.cos(math.pi * h)):.2f});"
          f" last ratio {hj[-1] / hj[-2]!r}, cos(pi h) = {math.cos(math.pi * h)!r}; max error {np.max(np.abs(uj - exact)):.2e}")
    print(f"  SOR omega = {omega!r}: {len(hs) - 1} sweeps (ln tol / ln(omega-1) = {math.log(TOL) / math.log(omega - 1):.2f});"
          f" max error {np.max(np.abs(us - exact)):.2e}")
    print(f"  CG {len(hc) - 1} iterations; kappa = cot^2(pi h/2) = {kappa(h)!r};"
          f" bound 0.5 sqrt(kappa) ln(2/tol) = {0.5 * math.sqrt(kappa(h)) * math.log(2 / TOL):.3f}; max error {np.max(np.abs(uc - exact)):.2e}")
    if n == 15:
        ug, hg = sor(b, 1.0)
        c = math.cos(math.pi * h)
        print(f"  Gauss-Seidel {len(hg) - 1} sweeps; last ratio {hg[-1] / hg[-2]!r}, cos^2(pi h) = {c * c!r}")
        print(f"  SOR residual / (omega-1)^k: k=30 {hs[30] / (omega - 1) ** 30:.2f}, k=60 {hs[60] / (omega - 1) ** 60:.2f}")
        # kappa checked against the matrix: dense A for n = 15, numpy eigvalsh
        N = n * n
        A = np.zeros((N, N))
        for i in range(n):
            for j in range(n):
                e = np.zeros((n, n))
                e[i, j] = 1.0
                A[:, i * n + j] = apply_A(e).ravel()
        ev = np.linalg.eigvalsh(A)
        print(f"  numpy: lambda_max / lambda_min of A = {ev[-1] / ev[0]!r}")
j, s, c = counts[15]
j2, s2, c2 = counts[31]
print(f"h 1/16 -> 1/32: Jacobi x{j2 / j:.3f}, SOR x{s2 / s:.3f}, CG x{c2 / c:.3f}")

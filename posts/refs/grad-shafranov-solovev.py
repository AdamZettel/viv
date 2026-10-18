# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-grad-shafranov-solovev.viv.

The post solves the Grad-Shafranov equation

    Delta* psi = psi_RR - psi_R / R + psi_ZZ = (1 + 1/kappa^2) R^2

by five-point finite differences and SOR on the box 0.6 <= R <= 1.4,
|Z| <= 0.7, with psi fixed on the boundary to Solov'ev's closed form

    psi(R, Z) = (R^2 - 1)^2 / 8 + R^2 Z^2 / (2 kappa^2),   kappa = 1.7.

This script checks the closed form symbolically (sympy), builds the same
five-point system as a sparse matrix and solves it directly with scipy's
spsolve (SuperLU), so the SOR iteration is not involved, and prints the
largest error against the closed form on each grid and the observed orders.
It also prints the flux surface through R = 1.25 on the midplane: its inner
crossing and the center of the two.

    python3 posts/refs/grad-shafranov-solovev.py > posts/refs/grad-shafranov-solovev.out
"""
import numpy as np
import scipy
import sympy as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

KAPPA = 1.7
R_, Z_ = sp.symbols("R Z", positive=True)
k = sp.Rational(17, 10)
psi_s = (R_**2 - 1)**2 / 8 + R_**2 * Z_**2 / (2 * k**2)
dstar = sp.diff(psi_s, R_, 2) - sp.diff(psi_s, R_) / R_ + sp.diff(psi_s, Z_, 2)
print("Delta* psi - (1 + 1/kappa^2) R^2 =", sp.simplify(dstar - (1 + 1 / k**2) * R_**2))


def psi_ex(R, Z):
    return (R * R - 1.0)**2 / 8.0 + R * R * Z * Z / (2.0 * KAPPA**2)


def solve(nr, nz):
    R = np.linspace(0.6, 1.4, nr)
    Z = np.linspace(-0.7, 0.7, nz)
    hr = R[1] - R[0]
    hz = Z[1] - Z[0]
    n = (nr - 2) * (nz - 2)
    A = lil_matrix((n, n))
    b = np.zeros(n)

    def idx(i, j):
        return (i - 1) * (nz - 2) + (j - 1)
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            r = R[i]
            row = idx(i, j)
            b[row] = (1.0 + 1.0 / KAPPA**2) * r * r
            coef = {(i + 1, j): 1 / hr**2 - 1 / (2 * r * hr), (i - 1, j): 1 / hr**2 + 1 / (2 * r * hr),
                    (i, j + 1): 1 / hz**2, (i, j - 1): 1 / hz**2}
            A[row, row] = -2 / hr**2 - 2 / hz**2
            for (p, q), c in coef.items():
                if p in (0, nr - 1) or q in (0, nz - 1):
                    b[row] -= c * psi_ex(R[p], Z[q])
                else:
                    A[row, idx(p, q)] = c
    x = spsolve(A.tocsr(), b)
    P = psi_ex(R[:, None], Z[None, :]).copy()
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            P[i, j] = x[idx(i, j)]
    err = np.max(np.abs(P - psi_ex(R[:, None], Z[None, :])))
    return R, Z, P, err


print(f"scipy {scipy.__version__}, sympy {sp.__version__}")
errs = []
for nr, nz in [(9, 15), (17, 29), (33, 57), (65, 113)]:
    R, Z, P, e = solve(nr, nz)
    errs.append(e)
    print(f"grid {nr} x {nz}, h = {R[1]-R[0]:.4f}: max |psi - psi_exact| = {e:.10e}")
for a, b in zip(errs, errs[1:]):
    print(f"observed order log2(e_h / e_h/2) = {np.log2(a / b):.6f}")
psi_b = psi_ex(1.25, 0.0)
r_in = np.sqrt(1.0 - np.sqrt(8.0 * psi_b))
for nr, nz in [(33, 57)]:
    R, Z, P, e = solve(nr, nz)
    j0 = (nz - 1) // 2
    psi_b = psi_ex(1.25, 0.0)
    for i in range(1, nr):
        a, b = P[i - 1, j0], P[i, j0]
        if a >= psi_b > b:
            print(f"33 x 57 grid, linear interpolation on the midplane: inner crossing R = {R[i-1] + (a - psi_b) / (a - b) * (R[1]-R[0]):.10f}")
print(f"flux surface through R = 1.25: psi = {psi_b:.10f}, inner crossing R = {r_in:.10f}, center {(1.25 + r_in)/2:.10f}")

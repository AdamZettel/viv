# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-gauss-legendre-quadrature-and-where-tanh-sinh-wins.viv.

Gauss-Legendre nodes and weights from numpy.polynomial.legendre.leggauss
(numpy computes them as the eigenvalues of the Jacobi matrix, a different
method from the post's Newton iteration on the Legendre recurrence), exact
integrals in closed form, and the tanh-sinh rule written out in numpy.
mpmath.quad(method="tanh-sinh") in 50-digit arithmetic confirms the exact
values the errors are measured against.
Run: python3 posts/refs/gauss-legendre-quadrature-and-where-tanh-sinh-wins.py > posts/refs/gauss-legendre-quadrature-and-where-tanh-sinh-wins.out
"""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import mpmath

mpmath.mp.dps = 50
print("exact values, checked with mpmath tanh-sinh at 50 digits:")
print("  int e^x          =", mpmath.nstr(mpmath.quad(mpmath.exp, [-1, 1]), 20), " e - 1/e =", repr(math.e - 1 / math.e))
print("  int 1/sqrt(1-x^2) =", mpmath.nstr(mpmath.quad(lambda x: 1 / mpmath.sqrt(1 - x * x), [-1, 1]), 20), " pi =", repr(math.pi))

x5, w5 = leggauss(5)
print("leggauss(5) nodes:  ", [repr(float(v)) for v in x5])
print("leggauss(5) weights:", [repr(float(v)) for v in w5])

print("integral of (1+x)^k over [-1, 1] = 2^(k+1)/(k+1); relative error of the n-point rule:")
for n in (3, 5):
    x, w = leggauss(n)
    errs = []
    for k in range(14):
        exact = 2.0 ** (k + 1) / (k + 1)
        errs.append(abs(np.sum(w * (1 + x) ** k) - exact) / exact)
    print(f"  n = {n}:", " ".join(f"{e:.2e}" for e in errs))
x, w = leggauss(5)
q10 = float(np.sum(w * (1 + x) ** 10))
closed = 2.0 ** 11 * math.factorial(5) ** 4 / (11 * math.factorial(10) ** 2)
print(f"degree 10, n = 5: exact - rule = {2.0 ** 11 / 11 - q10!r}; closed form 2^11 (5!)^4 / (11 (10!)^2) = {closed!r}")

E = math.e - 1 / math.e
print("e^x, n-point Gauss-Legendre, error:")
for n in range(1, 13):
    x, w = leggauss(n)
    print(f"  n = {n:2d}: {abs(np.sum(w * np.exp(x)) - E):.3e}")
print("1/sqrt(1 - x^2), n-point Gauss-Legendre, error and error * n:")
for n in (4, 8, 16, 32, 64, 128):
    x, w = leggauss(n)
    e = abs(np.sum(w / np.sqrt(1 - x * x)) - math.pi)
    print(f"  n = {n:3d}: {e:.6e}   {e * n:.4f}")

TMAX = 4.5
print(f"tanh-sinh, t in [-{TMAX}, {TMAX}], step h: points, error on e^x, error on 1/sqrt(1-x^2):")
for level in range(7):
    h = 0.5 ** level
    K = int(math.floor(TMAX / h))
    t = np.arange(-K, K + 1) * h
    u = 0.5 * math.pi * np.sinh(t)
    sech2 = 1.0 / np.cosh(u) ** 2
    wt = h * 0.5 * math.pi * np.cosh(t) * sech2
    e_exp = abs(np.sum(wt * np.exp(np.tanh(u))) - E)
    terms = wt / np.sqrt(sech2)
    e_sing = abs(np.sum(terms) - math.pi)
    print(f"  h = 1/{2 ** level:<3d} {2 * K + 1:4d} points  {e_exp:.3e}  {e_sing:.3e}"
          f"   last node t = {K * h}, its term on 1/sqrt(1-x^2) {terms[-1]:.3e}")

h = 0.25
K = int(math.floor(TMAX / h))
t = np.arange(-K, K + 1) * h
u = 0.5 * math.pi * np.sinh(t)
x = np.tanh(u)
with np.errstate(divide="ignore"):
    naive = np.sum(h * 0.5 * np.pi * np.cosh(t) / np.cosh(u) ** 2 / np.sqrt(1.0 - x * x))
print(f"h = 1/4: nodes where 1 - x*x rounds to 0: {int(np.sum(1.0 - x * x == 0.0))} of {2 * K + 1};"
      f" the sum with 1 - x*x computed from x: {naive}")

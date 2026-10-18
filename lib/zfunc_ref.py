# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference values for lib/zfunc.viv, from scipy.special.wofz.

scipy's wofz is the Faddeeva package of S. G. Johnson (MIT), which combines
Algorithm 916 (Zaghloul and Ali, ACM TOMS 38, 2011) with continued fractions
for large |z| and Taylor expansions near the axes: a different algorithm from
the Weideman rational expansion lib/zfunc.viv uses, so agreement is a check.

Run: python3 lib/zfunc_ref.py   (scipy 1.12, numpy 1.26)
"""
import numpy as np
from scipy.special import wofz, dawsn

POINTS = [
    0.0 + 0.0j,       # w(0) = 1
    1.0 + 0.0j,       # real axis: w(x) = exp(-x^2) + (2i/sqrt(pi)) Dawson(x)
    0.5 + 0.3j,
    2.0 + 1.0j,
    -3.0 + 0.5j,      # w(-conj z) = conj w(z)
    5.5 + 2.5j,
    0.0 + 3.0j,       # imaginary axis: w(iy) = exp(y^2) erfc(y), real
    2.0 - 0.2j,       # lower half plane: w(z) = 2 exp(-z^2) - w(-z)
    -1.2 - 0.7j,
    0.3 - 1.0j,
]
for z in POINTS:
    w = wofz(z)
    print(f"z = {z.real:5.2f} {z.imag:+5.2f}i   w = {w.real!r} {w.imag!r}")

print("check: w(1) - (e^-1 + 2i/sqrt(pi) D(1)) =", abs(wofz(1.0) - (np.exp(-1.0) + 2j / np.sqrt(np.pi) * dawsn(1.0))))

# the plasma dispersion function Z = i sqrt(pi) w and its derivative Z' = -2 (1 + z Z)
for z in [1.0 + 0.0j, 2.0 - 0.2j]:
    Z = 1j * np.sqrt(np.pi) * wofz(z)
    print(f"Z({z}) = {Z.real!r} {Z.imag!r};  Z' = {(-2 * (1 + z * Z)).real!r} {(-2 * (1 + z * Z)).imag!r}")

# the largest relative error of the Weideman expansion with N = 40 terms on the
# grid the post's figure uses, as computed here in numpy (for the post's claim)
def weideman(z, N=40):
    M = 2 * N
    L = np.sqrt(N / np.sqrt(2))
    k = np.arange(-M + 1, M)
    th = k * np.pi / M
    t = L * np.tan(th / 2)
    f = np.exp(-t * t) * (L * L + t * t)
    a = np.array([np.sum(f * np.cos(n * th)) / (2 * M) for n in range(1, N + 1)])
    zu = np.where(z.imag >= 0, z, -z)
    Z = (L + 1j * zu) / (L - 1j * zu)
    p = np.polyval(a[::-1], Z)
    wu = 2 * p / (L - 1j * zu) ** 2 + (1 / np.sqrt(np.pi)) / (L - 1j * zu)
    return np.where(z.imag >= 0, wu, 2 * np.exp(-z * z) - wu)

X, Y = np.meshgrid(np.linspace(-6, 6, 241), np.linspace(-1, 4, 101))
Zg = X + 1j * Y
for N in [32, 40]:
    rel = np.abs(weideman(Zg, N) - wofz(Zg)) / np.abs(wofz(Zg))
    print(f"numpy Weideman N={N}, max relative error on [-6,6] x [-1,4]:", rel.max())

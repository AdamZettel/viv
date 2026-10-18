# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference values for posts/2026-09-25-landau-damping-from-the-dispersion-relation.viv.

The Langmuir-wave dispersion relation for a Maxwellian plasma, in units
omega_pe = 1, lambda_D = 1, v_th = 1 (so k means k lambda_D):

    eps(omega) = 1 + (1 + zeta Z(zeta)) / k^2 = 0,   zeta = omega / (sqrt(2) k),

with Z(zeta) = i sqrt(pi) wofz(zeta) from scipy.special. The root is found by
scipy.optimize.newton with the derivative from Z' = -2 (1 + zeta Z), started at
the Bohm-Gross frequency, and polished with mpmath at 30 digits as a check on
the double-precision root.

Run: python3 posts/refs/landau_root.py   (scipy 1.12, numpy 1.26, mpmath 1.3)
"""
import numpy as np
from scipy.special import wofz
from scipy.optimize import newton
import mpmath as mp

RT2 = np.sqrt(2.0)


def Z(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def eps(w, k):
    z = w / (RT2 * k)
    return 1.0 + (1.0 + z * Z(z)) / (k * k)


def deps(w, k):
    z = w / (RT2 * k)
    dZ = -2.0 * (1.0 + z * Z(z))
    return (Z(z) + z * dZ) / (k * k) / (RT2 * k)


def root(k, guess):
    return newton(lambda w: eps(w, k), guess, fprime=lambda w: deps(w, k), tol=1e-13, maxiter=100)


def mp_root(k, guess):
    mp.mp.dps = 30
    k = mp.mpf(k)
    def e(w):
        z = w / (mp.sqrt(2) * k)
        Zm = 1j * mp.sqrt(mp.pi) * mp.exp(-z * z) * mp.erfc(-1j * z)
        return 1 + (1 + z * Zm) / (k * k)
    return mp.findroot(e, mp.mpc(guess))


def weak(k):
    return -np.sqrt(np.pi / 8.0) / k**3 * np.exp(-1.0 / (2 * k * k) - 1.5)


w05 = root(0.5, complex(np.sqrt(1 + 3 * 0.25), -0.01))
print("k = 0.5: omega =", repr(w05.real), repr(w05.imag), " |eps| =", abs(eps(w05, 0.5)))
print("  mpmath 30 digits:", mp_root(0.5, w05))
print("  phase velocity omega_r / k =", w05.real / 0.5)
print("  weak-damping formula at 0.5:", weak(0.5), " ratio", weak(0.5) / w05.imag)
for k in [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
    g = root(k, complex(np.sqrt(1 + 3 * k * k), 0.0))
    m = mp_root(k, g)
    print(f"k = {k}: mpmath omega = {mp.nstr(m.real, 17)} {mp.nstr(m.imag, 17)}")
    print(f"        scipy - mpmath = {abs(complex(m) - g):.1e}   Bohm-Gross {np.sqrt(1 + 3 * k * k)!r}   weak {weak(k)!r}  ratio {weak(k) / g.imag!r}")

# the next root down at k = 0.5, from the guess 1.8 - 1.1i read off the post's heatmap
w2 = root(0.5, complex(1.8, -1.1))
print("k = 0.5, second root: scipy", repr(w2.real), repr(w2.imag), " |eps| =", abs(eps(w2, 0.5)))
print("  mpmath 30 digits:", mp_root(0.5, w2))

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference values for posts/2026-09-25-hasegawa-wakatani-one-rung-at-a-time.viv.

The linearized Hasegawa-Wakatani equations for one Fourier mode
exp(i(kx x + ky y) + lambda t), with zeta = -k^2 phi,

    d/dt [zeta, n] = M [zeta, n],
    M = [[-C/k^2,                  -C],
         [-C/k^2 + i kappa ky/k^2, -C]],

are solved here with numpy.linalg.eigvals (growth rate = max Re lambda,
frequency = -Im of that eigenvalue), and the zonal (ky = 0) relaxation with
scipy.linalg.expm. The post computes the same things with the quadratic
formula and a Runge-Kutta loop.

Run: python3 posts/refs/hasegawa_wakatani.py   (numpy 1.26, scipy 1.12)
"""
import numpy as np
from scipy.linalg import expm

KAPPA = 1.0


def M(kx, ky, C, tilde=1.0):
    k2 = kx * kx + ky * ky
    c = C * (tilde if ky == 0 else 1.0)
    return np.array([[-c / k2, -c], [-c / k2 + 1j * KAPPA * ky / k2, -c]])


def mode(kx, ky, C):
    lam = np.linalg.eigvals(M(kx, ky, C))
    l = lam[np.argmax(lam.real)]
    return l.real, -l.imag


def hm(kx, ky):
    return KAPPA * ky / (1 + kx * kx + ky * ky)


print("HM omega(0, 1)      =", repr(hm(0, 1)))
print("HM omega(0.3, 0.7)  =", repr(hm(0.3, 0.7)))
print("HM omega(0, 2)      =", repr(hm(0, 2)))
g, w = mode(0.0, 1.0, 1.0)
print("C = 1, (0, 1): gamma =", repr(g), " omega =", repr(w))
g, w = mode(0.3, 0.7, 0.5)
print("C = 0.5, (0.3, 0.7): gamma =", repr(g), " omega =", repr(w))
for C in [10.0, 100.0, 1000.0, 1e4]:
    g, w = mode(0.0, 1.0, C)
    print(f"C = {C:8g}: C*gamma = {C * g!r}   C^2 (1 - omega/omega_HM) = {C * C * (1 - w / hm(0, 1))!r}")
print("large-C predictions at (0,1): C gamma -> w^2 k^2/(1+k^2) =", 0.25 * 1 / 2,
      "; C^2 (1 - w/w_HM) -> 2 w^2 k^4/(1+k^2)^2 =", 2 * 0.25 / 4)
# growth-rate scan in ky for the figure's claim: the fastest-growing ky at C = 1
kys = np.linspace(0.05, 3.0, 5901)
gs = np.array([mode(0.0, k, 1.0)[0] for k in kys])
print("C = 1: max gamma over ky in [0.05, 3] =", gs.max(), "at ky =", kys[gs.argmax()])

kz, T = 0.5, 2.0
for tilde, name in [(1.0, "classical"), (0.0, "modified")]:
    z = (expm(M(kz, 0.0, 1.0, tilde) * T) @ np.array([1.0, 0.0])).real
    print(f"{name}: zeta(T)/zeta(0) at kx = {kz}, T = {T}: {z[0]!r}")
print("residual k^2/(1+k^2) =", kz**2 / (1 + kz**2))
zinf = (expm(M(kz, 0.0, 1.0) * 60.0) @ np.array([1.0, 0.0])).real
print("classical at T = 60:", repr(zinf[0]))

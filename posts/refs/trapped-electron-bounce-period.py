# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-trapped-electron-bounce-period.viv.

The post integrates the trapped electron's equation of motion in the wave
frame, x'' = -phi0 sin x (units k = m = e = 1), with velocity Verlet, and
computes the complete elliptic integral K(m) by the arithmetic-geometric mean.
This script gets the same numbers without either:

  - K(m) from scipy.special.ellipk (parameter m = k^2), which uses the Cephes
    polynomial approximations, not an AGM;
  - the bounce period by integrating the pendulum with scipy's solve_ivp
    (DOP853, rtol = atol = 1e-12) and locating the first return to v = 0 with
    an event, doubled;
  - the trapping boundary from energy conservation: an electron at the trough
    (x = 0) with speed v0 has energy v0^2/2 - phi0, and it passes over the
    crest (x = pi, potential +phi0) when that energy exceeds phi0, so the
    boundary is v0 = 2 sqrt(phi0), found here by brentq on the sign of the
    energy margin of a solve_ivp run.

    python3 posts/refs/trapped-electron-bounce-period.py > posts/refs/trapped-electron-bounce-period.out
"""
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import ellipk


def period_ivp(theta0, phi0):
    def rhs(t, y):
        return [y[1], -phi0 * np.sin(y[0])]

    def turn(t, y):
        return y[1]
    turn.direction = 1.0  # v rises through 0 at the far turning point
    s = solve_ivp(rhs, [0, 200], [theta0, 0.0], method="DOP853", rtol=1e-12, atol=1e-12,
                  events=turn)
    return 2.0 * s.t_events[0][0]


print(f"scipy {scipy.__version__}")
th0 = 0.1
m = np.sin(th0 / 2) ** 2
T_ellip = 4.0 * ellipk(m)
print(f"theta0 = 0.1: 4 K(sin^2(theta0/2)) = {T_ellip:.12f}")
print(f"theta0 = 0.1: solve_ivp period     = {period_ivp(th0, 1.0):.12f}")
print(f"2 pi (1 + theta0^2/16)             = {2*np.pi*(1+th0**2/16):.12f}")
print(f"2 pi (1 + theta0^2/16 + 11 theta0^4/3072) = {2*np.pi*(1+th0**2/16+11*th0**4/3072):.12f}")
print(f"phi0 = 4, theta0 = 0.1: solve_ivp period = {period_ivp(th0, 4.0):.12f} (half of phi0 = 1)")
for a in [0.5, 1.0, 2.0, 2.5, 3.0, 3.1]:
    print(f"theta0 = {a}: 4 K = {4*ellipk(np.sin(a/2)**2):.10f}   solve_ivp = {period_ivp(a, 1.0):.10f}")
print(f"K(0.5) = {ellipk(0.5):.15f}")

from scipy.optimize import brentq


def excess(v0):
    # largest x reached in 40 time units, minus pi: positive once the electron passes the crest
    s = solve_ivp(lambda t, y: [y[1], -np.sin(y[0])], [0, 40], [0.0, v0], method="DOP853",
                  rtol=1e-12, atol=1e-12, dense_output=True)
    return np.max(s.y[0]) - np.pi


vb = brentq(excess, 1.9, 2.1, xtol=1e-6)
print(f"trapping boundary speed, phi0 = 1: {vb:.6f} (2 sqrt(phi0) = 2)")

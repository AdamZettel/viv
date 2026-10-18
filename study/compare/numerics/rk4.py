# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ the ODE: a harmonic oscillator as a first-order system
OMEGA = 2.0

def rhs(t, y):
    return np.array([y[1], -OMEGA * OMEGA * y[0]])

# @@ one classical Runge-Kutta step
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# @@ integrate to t = 1 and compare with cos(2t)
T_END = 1.0
for steps in [10, 20, 40]:
    h = T_END / steps
    y = np.array([1.0, 0.0])
    t = 0.0
    for _ in range(steps):
        y = rk4_step(rhs, t, y, h)
        t += h
    err = abs(y[0] - np.cos(OMEGA * T_END))
    print(f"{steps:3d} steps: x(1) = {y[0]:.10f}, error {err:.2e}")

# @@ checks
assert err < 1e-6

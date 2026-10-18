# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-chuas-circuit-double-scroll.viv.

The largest Lyapunov exponent of Chua's circuit (alpha = 9, beta = 100/7,
m0 = -8/7, m1 = -5/7), computed a different way from the post: scipy's
DOP853 at rtol = atol = 1e-11 integrates the flow together with its
variational equation (the Jacobian applied to a tangent vector), and the
tangent vector is renormalized once per time unit.

    python3 posts/refs/chuas-circuit-double-scroll.py > posts/refs/chuas-circuit-double-scroll.out

It prints the exponent from one long run, and the spread of the estimate
over windows as long as the post's (600 time units), which sets the post's
tolerance. Output recorded 2026-09-25 (scipy 1.12.0) is in
posts/refs/chuas-circuit-double-scroll.out.
"""
import numpy as np
from scipy.integrate import solve_ivp

ALPHA = 9.0
BETA = 100.0 / 7.0
M0 = -8.0 / 7.0
M1 = -5.0 / 7.0


def diode(x):
    return M1 * x + 0.5 * (M0 - M1) * (abs(x + 1) - abs(x - 1))


def slope(x):
    return M0 if abs(x) < 1 else M1


def rhs(t, u):
    x, y, z, a, b, c = u
    g = slope(x)
    return [ALPHA * (y - x - diode(x)), x - y + z, -BETA * y,
            ALPHA * (b - a - g * a), a - b + c, -BETA * b]


print("diode(0.5) =", diode(0.5), " diode(2.0) =", diode(2.0))

TOL = dict(method="DOP853", rtol=1e-11, atol=1e-11)
print(f"integrator: {TOL['method']} at rtol = {TOL['rtol']:g}, atol = {TOL['atol']:g}")
# settle onto the attractor from the post's start (0.7, 0, 0)
state = solve_ivp(lambda t, u: rhs(t, list(u) + [0, 0, 0])[:3], (0, 100), [0.7, 0, 0], **TOL).y[:, -1]

WINDOW = 600      # the post's measuring time: 120000 steps of 0.005
NWIN = 40
# the x samples every 0.005 time units, the post's step, for counting hops
T_SAMPLE = np.linspace(0.0, 1.0, 201)
logs = []
hops = []
last_sign = np.sign(state[0])
v = np.array([1.0, 0.0, 0.0])
for w in range(NWIN * WINDOW):
    sol = solve_ivp(rhs, (0, 1), list(state) + list(v), t_eval=T_SAMPLE, **TOL)
    signs = np.sign(sol.y[0, 1:])
    hops.append(int(np.sum(signs != np.concatenate([[last_sign], signs[:-1]]))))
    last_sign = signs[-1]
    state = sol.y[:3, -1]
    v = sol.y[3:, -1]
    n = np.linalg.norm(v)
    logs.append(np.log(n))
    v = v / n
logs = np.array(logs)
lam = logs.mean()
windows = logs.reshape(NWIN, WINDOW).mean(axis=1)
print(f"largest Lyapunov exponent over {NWIN * WINDOW} time units: {lam:.4f}")
print(f"its standard error (from the {NWIN} windows): {windows.std(ddof=1) / np.sqrt(NWIN):.4f}")
print(f"spread of one {WINDOW}-time-unit estimate (std over windows): {windows.std(ddof=1):.4f}")
print(f"windows: min {windows.min():.4f}, max {windows.max():.4f}")

# hops between the lobes (sign changes of x): the rate, and the spread of
# the count over windows of 210 time units, the post's filmed stretch
FILMED = 210
hops = np.array(hops)
print(f"hops per time unit: {hops.mean():.4f}")
counts = hops[: (len(hops) // FILMED) * FILMED].reshape(-1, FILMED).sum(axis=1)
print(f"hops in {FILMED} time units: mean {counts.mean():.2f}, std {counts.std(ddof=1):.2f}, "
      f"min {counts.min()}, max {counts.max()} ({len(counts)} windows)")

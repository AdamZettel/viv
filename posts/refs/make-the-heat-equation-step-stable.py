# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-make-the-heat-equation-step-stable.viv (a problem).

u_t = u_xx on 19 interior points with u = 0 at both ends, r = dt / dx^2.
Backward Euler solves (I + r L) u_new = u, L the second-difference matrix
[-1, 2, -1]; here with numpy.linalg.solve on the full matrix. The decay of
each sine mode is also given in closed form: mode m is multiplied by
1 / (1 + 4 r sin^2(m pi / (2 (n + 1)))) per step.

    python3 posts/refs/make-the-heat-equation-step-stable.py > posts/refs/make-the-heat-equation-step-stable.out
"""
import numpy as np

n = 19
NSTEP = 16
L = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
i = np.arange(1, n + 1)


def run(u, r, implicit=True):
    for _ in range(NSTEP):
        u = np.linalg.solve(np.eye(n) + r * L, u) if implicit else u - r * (L @ u)
    return u


checker = np.array([(-1.0) ** k for k in range(n)])
for r in (0.4, 0.6):
    print(f"FTCS, r = {r}: max |u| after {NSTEP} steps from the checkerboard = {np.max(np.abs(run(checker, r, False))):.6g}")
for r in (0.6, 5.0):
    u = run(checker, r)
    print(f"backward Euler, r = {r}: checkerboard after {NSTEP} steps: u[9] = {u[9]!r}, max |u| = {np.max(np.abs(u))!r}")
for m in (1, n):
    r = 0.6
    u0 = np.sin(m * np.pi * i / (n + 1))
    u = run(u0, r)
    g = 1.0 / (1.0 + 4 * r * np.sin(m * np.pi / (2 * (n + 1))) ** 2)
    print(f"mode m = {m}, r = {r}: numpy u[k]/u0[k] = {u[4] / u0[4]!r}, closed form g^{NSTEP} = {g ** NSTEP!r}")

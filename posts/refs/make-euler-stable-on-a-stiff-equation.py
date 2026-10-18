# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-make-euler-stable-on-a-stiff-equation.viv (a problem).

The tests hold the behavior of backward (implicit) Euler, computed here with
numpy, not with the program being solved: y_{n+1} = y_n / (1 + 50 h) for the
scalar equation, and y_{n+1} = (I - h A)^{-1} y_n with numpy.linalg.solve for
the 2 by 2 system. The exact solutions are printed beside them.

    python3 posts/refs/make-euler-stable-on-a-stiff-equation.py > posts/refs/make-euler-stable-on-a-stiff-equation.out
"""
import numpy as np

LAM = 50.0
for h, n in ((0.05, 1), (0.05, 12), (1.0, 1)):
    y = 1.0
    for _ in range(n):
        y = y / (1.0 + LAM * h)
    yf = 1.0
    for _ in range(n):
        yf = yf * (1.0 - LAM * h)
    print(f"y' = -50 y, h = {h}, {n} steps: backward Euler {y!r}, forward Euler {yf!r}, exact {np.exp(-LAM * h * n)!r}")

A = np.array([[-500.5, 499.5], [499.5, -500.5]])
print("eigenvalues of A:", np.linalg.eigvalsh(A))
h = 0.1
y = np.array([2.0, 0.0])
yf = y.copy()
for _ in range(10):
    y = np.linalg.solve(np.eye(2) - h * A, y)
    yf = yf + h * A @ yf
print(f"system, h = {h}, 10 steps: backward Euler [{y[0]!r}, {y[1]!r}]")
print(f"  forward Euler {yf.tolist()}")
print(f"  exact at t = 1: [{np.exp(-1) + np.exp(-1000)!r}, {np.exp(-1) - np.exp(-1000)!r}]")
print(f"  (1 / 1.1)^10 = {(1 / 1.1) ** 10!r}: the slow mode's backward-Euler factor")

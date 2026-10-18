# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math

# @@ the problem: y' = -2 t y, y(0) = 1, exact y = exp(-t^2)
def f(t, y):
    return -2.0 * t * y

def exact(t):
    return math.exp(-t * t)

# @@ forward Euler
def euler(f, y0, t_end, n):
    h = t_end / n
    t, y = 0.0, y0
    for _ in range(n):
        y += h * f(t, y)
        t += h
    return y

# @@ the error halves when the step halves
T_END = 2.0
N0 = 10
REFINEMENTS = 5
errors = []
for k in range(REFINEMENTS):
    n = N0 * 2 ** k
    y = euler(f, 1.0, T_END, n)
    errors.append(abs(y - exact(T_END)))
    print(f"{n:4d} steps  y(2) = {y:.9f}  error = {errors[-1]:.3e}")
for k in range(1, REFINEMENTS):
    print(f"halving the step divides the error by {errors[k - 1] / errors[k]:.3f}")

# @@ checks
assert abs(errors[-2] / errors[-1] - 2.0) < 0.1

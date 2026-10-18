# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math

# @@ composite Simpson's rule
def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 == 1 else 2) * f(a + i * h)
    return s * h / 3

# @@ the error falls as h^4
exact = 1.0 - math.cos(1.0)
prev = None
for n in [4, 8, 16, 32]:
    err = abs(simpson(math.sin, 0.0, 1.0, n) - exact)
    ratio = f"{prev / err:6.2f}" if prev else "     -"
    print(f"n = {n:2d}  error = {err:.3e}  ratio = {ratio}")
    prev = err

# @@ a Gaussian integral
g = simpson(lambda x: math.exp(-x * x), -6.0, 6.0, 200)
print(f"integral of exp(-x^2) = {g:.12f}, sqrt(pi) = {math.sqrt(math.pi):.12f}")

# @@ checks
assert abs(g - math.sqrt(math.pi)) < 1e-10

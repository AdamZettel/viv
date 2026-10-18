# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports and constants
import math

# @@ Newton's method with a given derivative
def newton(f, df, x, tol=1e-12, max_iter=50):
    for k in range(max_iter):
        step = f(x) / df(x)
        x -= step
        print(f"  step {k + 1}: x = {x:.12f}")
        if abs(step) < tol:
            return x
    raise RuntimeError("Newton did not converge")

# @@ the square root of 2
root2 = newton(lambda x: x * x - 2.0, lambda x: 2.0 * x, 1.0)
print(f"sqrt(2) ~ {root2:.12f}")

# @@ where cos(x) = x
dottie = newton(lambda x: math.cos(x) - x, lambda x: -math.sin(x) - 1.0, 1.0)
print(f"cos(x) = x at {dottie:.12f}")

# @@ checks
assert abs(root2 - math.sqrt(2.0)) < 1e-12
assert abs(math.cos(dottie) - dottie) < 1e-12

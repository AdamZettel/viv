# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Every NumPy and math function in py2viv's mapping (SUBSET.md), called once and printed, so
diff.py checks each against NumPy in three engines. The calling-convention adapters (sort,
fft, solve, cholesky, inv, eigh, eigvalsh) are in adapters.py.

Viv's libm is its own (lang/viv/src/vmath), not the platform's, and its sum runs left to
right where NumPy's is pairwise, so agreement is to rounding.
# py2viv: rtol=1e-15 atol=0
"""
import math
import numpy as np

x = np.linspace(-1.5, 2.0, 8)
p = np.linspace(0.1, 3.0, 8)
M = np.array([[1.0, 2.0, -0.5], [0.25, -3.0, 4.0], [2.0, 0.5, 1.5]])
v = np.array([3.0, -1.0, 2.5])
w = np.array([0.5, 2.0, -1.0])

print("zeros", np.zeros(3), np.zeros((2, 2)))
print("ones full", np.ones(3), np.full(3, 2.5), np.full((2, 2), -1.0))
print("arange", np.arange(5), np.arange(1.0, 2.0, 0.25), np.arange(10, 0, -3))
print("linspace", x)
print("eye", np.eye(3), np.eye(3, k=1), np.identity(2))
print("like", np.zeros_like(M), np.ones_like(v), np.full_like(v, 4.0))
print("diag", np.diag(v), np.diag(M))
print("trace outer", np.trace(M), np.outer(v, w))
print("cumsum prod", np.cumsum(x), np.prod(v))
print("argmax argmin", np.argmax(x), np.argmin(M))
print("sum mean std var", np.sum(M), np.mean(x), np.std(x), np.var(x))
print("max min", np.max(M), np.min(x), np.amax(v), np.amin(v), max(v), min(w))
print("dot matmul", np.dot(v, w), np.dot(M, v), M @ M, v @ M, np.matmul(M, M))
print("transpose", np.transpose(M), M.T)
print("elementwise", np.sqrt(p), np.exp(x), np.log(p), np.sin(x), np.cos(x), np.tan(x))
print("more", np.tanh(x), np.sinh(x), np.cosh(x), np.arctan(x), np.abs(x), np.floor(x), np.ceil(x))
print("inverse trig", np.arcsin(x / 2.0), np.arccos(x / 2.0))
print("maximum minimum", np.maximum(x, 0.0), np.minimum(x, w[0]), np.clip(x, -1.0, 1.0))
print("sign where", np.sign(x), np.where(x > 0.0, x, -x), np.square(x))
print("all any", np.all(p > 0.0), np.any(x > 1.9), np.all(x > 0.0))
print("power", np.power(p, 1.5), p ** 2.0, 2.0 ** 0.5)
print("norm", np.linalg.norm(v), np.linalg.norm(M))
print("roll diff cross", np.roll(x, 2), np.roll(x, -1), np.diff(x), np.cross(v, w))
print("isclose allclose", np.isclose(0.1 + 0.2, 0.3), np.allclose(v, v + 1e-12), np.isnan(x[0]))
print("math", math.sqrt(2.0), math.exp(1.5), math.log(10.0), math.log(8.0, 2.0), math.log10(1000.0), math.log2(0.125))
print("math trig", math.sin(0.5), math.cos(0.5), math.tan(0.5), math.asin(0.5), math.acos(0.5), math.atan(0.5), math.atan2(1.0, -2.0))
print("math hyp", math.sinh(0.5), math.cosh(0.5), math.tanh(0.5), math.erf(0.5), math.gamma(4.5))
print("math misc", math.floor(-2.5), math.ceil(-2.5), math.fabs(-3.0), math.hypot(3.0, 4.0), math.fmod(-7.0, 3.0), math.pi, math.e)
print("python", abs(-2.5), round(2.5), round(-3.5), int(-7.9), float(3), pow(2.0, 10), -7 % 3, -7 // 2, 7.5 % -2.0)
print("sorted copy", sorted(np.array([3.0, 1.0, 2.0])), np.copy(v), v.copy())
print("methods", M.sum(), M.max(), M.min(), M.mean(), x.std(), v.dot(w), M.shape[1], M.size, M.ndim)
print("isfinite isinf", math.isfinite(1.0), math.isinf(math.inf), np.isinf(x[0]))
print("vector % //", (x + 0.5) % 1.0 - 0.5, x // 0.75)
print("nested comprehension", np.array([[1.0 / (1 + i + j) + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]))

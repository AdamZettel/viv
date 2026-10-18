# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-bfgs-against-gradient-descent-on-rosenbrock.viv.

The same two algorithms in plain Python floats, operation for operation:
backtracking line search (t = 1, halved until the Armijo condition with
c = 1e-4 holds, or t <= 1e-14), gradient descent until |x - (1,1)| < 1e-9,
BFGS from H = I until |grad f| < 1e-12, skipping the update when y.s <= 1e-14.
Then numpy.linalg.eigvalsh on the Hessian at (1, 1), and scipy's own BFGS
(which uses a Wolfe line search) for comparison.
Run: python3 posts/refs/bfgs-against-gradient-descent-on-rosenbrock.py > posts/refs/bfgs-against-gradient-descent-on-rosenbrock.out
"""
import math
import numpy as np
import scipy.optimize


def rosen(p):
    a = 1.0 - p[0]
    b = p[1] - p[0] * p[0]
    return a * a + 100.0 * b * b


def grad(p):
    b = p[1] - p[0] * p[0]
    return [-2.0 * (1.0 - p[0]) - 400.0 * p[0] * b, 200.0 * b]


def line_search(x, d, g):
    t, fx = 1.0, rosen(x)
    slope = g[0] * d[0] + g[1] * d[1]
    while rosen([x[0] + t * d[0], x[1] + t * d[1]]) > fx + 1e-4 * t * slope and t > 1e-14:
        t = 0.5 * t
    return t


def dist(p):
    return math.sqrt((p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 1.0) * (p[1] - 1.0))


def matmul(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]


START = [-1.2, 1.0]

# gradient descent
x = list(START)
gd_err = []
gd_x = []
while True:
    gd_x.append(x)
    gd_err.append(dist(x))
    if gd_err[-1] < 1e-9:
        break
    g = grad(x)
    d = [-g[0], -g[1]]
    t = line_search(x, d, g)
    x = [x[0] + t * d[0], x[1] + t * d[1]]
gd_iters = len(gd_err) - 1
steps = [math.hypot(gd_x[k + 1][0] - gd_x[k][0], gd_x[k + 1][1] - gd_x[k][1]) for k in range(gd_iters)]
j = max(range(gd_iters), key=lambda k: steps[k])
print(f"gradient descent's longest step: iteration {j}, length {steps[j]!r},"
      f" from {gd_x[j]!r} to {gd_x[j + 1]!r}; f there {rosen(gd_x[j])!r} -> {rosen(gd_x[j + 1])!r};"
      f" f at x = 0 on the segment {rosen([0.0, gd_x[j][1] + (gd_x[j + 1][1] - gd_x[j][1]) * (-gd_x[j][0]) / (gd_x[j + 1][0] - gd_x[j][0])])!r};"
      f" second longest {sorted(steps)[-2]!r}; its step length t = {steps[j] / math.hypot(*grad(gd_x[j]))!r}")
first = lambda errs, tol: next(k for k, e in enumerate(errs) if e < tol)
print(f"gradient descent: {gd_iters} iterations to 1e-9; below 1e-3 at {first(gd_err, 1e-3)}, below 1e-6 at {first(gd_err, 1e-6)}")
k6 = first(gd_err, 1e-6)
rate = (gd_err[gd_iters] / gd_err[k6]) ** (1.0 / (gd_iters - k6))
print(f"  rate per step from 1e-6 to 1e-9: {rate!r}")

# BFGS
x = list(START)
H = [[1.0, 0.0], [0.0, 1.0]]
g = grad(x)
bf_err = []
bf_t = []
while True:
    bf_err.append(dist(x))
    if math.sqrt(g[0] * g[0] + g[1] * g[1]) < 1e-12:
        break
    d = [-(H[0][0] * g[0] + H[0][1] * g[1]), -(H[1][0] * g[0] + H[1][1] * g[1])]
    t = line_search(x, d, g)
    bf_t.append(t)
    s = [t * d[0], t * d[1]]
    xn = [x[0] + s[0], x[1] + s[1]]
    gn = grad(xn)
    y = [gn[0] - g[0], gn[1] - g[1]]
    sy = s[0] * y[0] + s[1] * y[1]
    if sy > 1e-14:
        rho = 1.0 / sy
        L = [[1.0 - rho * s[0] * y[0], -rho * s[0] * y[1]], [-rho * s[1] * y[0], 1.0 - rho * s[1] * y[1]]]
        LT = [[L[0][0], L[1][0]], [L[0][1], L[1][1]]]
        H = matmul(matmul(L, H), LT)
        for i in range(2):
            for j in range(2):
                H[i][j] += rho * s[i] * s[j]
    x, g = xn, gn
bf_iters = len(bf_err) - 1
print(f"BFGS: |grad f| < 1e-12 after {bf_iters} iterations; within 1e-9 of (1,1) at {first(bf_err, 1e-9)}")
for k in range(bf_iters - 7, bf_iters + 1):
    step = f"   step length there: {bf_t[k]!r}" if k < bf_iters else ""
    print(f"  error after iteration {k}: {bf_err[k]!r}   ratio to the one before: {bf_err[k] / bf_err[k - 1]:.3g}{step}")

# the Hessian at (1, 1)
Hess = np.array([[802.0, -400.0], [-400.0, 200.0]])
ev = np.linalg.eigvalsh(Hess)
kappa = ev[1] / ev[0]
print(f"Hessian eigenvalues {ev[0]!r} {ev[1]!r}; kappa {kappa!r}; (kappa-1)/(kappa+1) {(kappa - 1) / (kappa + 1)!r}")
print(f"  log(rate) / log((kappa-1)/(kappa+1)) = {math.log(rate) / math.log((kappa - 1) / (kappa + 1)):.4f}")

r = scipy.optimize.minimize(rosen, START, jac=grad, method="BFGS", options={"gtol": 1e-12})
print(f"scipy.optimize.minimize BFGS (Wolfe line search): {r.nit} iterations, {r.nfev} function evaluations,"
      f" |x - (1,1)| = {dist(r.x):.1e}")

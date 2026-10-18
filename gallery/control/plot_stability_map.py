# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A stability map for the cart-pole's feedback gains
==================================================

Scale the two halves of the cart-pole's LQR gain, the part that acts on the
cart and the part that acts on the pole, and map which combinations keep the
upright balanced. For every stable gain, solve the Lyapunov equation
A X + X A^T = -I with `vp.control.lyapunov` (Bartels-Stewart, on Viv's
Schur method, an algorithm after SLICOT). X is the steady-state covariance of the
state when unit white noise drives every component, so its trace measures
how far disturbances push each stable loop.
"""

# %%
# ## The family of gains
#
# The model and the gain are the ones in the cart-pole LQR example:
# `vp.control.cartpole_model` linearizes the nonlinear cart-pole about the
# upright and `vp.control.lqr` gives K. The gain here is
# K(a, b) = (a K_x, a K_xdot, b K_theta, b K_thetadot) for a and b from 0 to 3.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

A, B = vp.control.cartpole_model(h=1e-5)
K = vp.control.lqr(
    A=A, B=B, Q=np.diag([1.0, 1.0, 20.0, 1.0]), R=np.array([[0.05]])
).K.ravel()
print("K =", np.round(K, 4))

a_s = np.linspace(0.0, 3.0, 61)
b_s = np.linspace(0.0, 3.0, 61)
growth = np.zeros((len(b_s), len(a_s)))
cost = np.full((len(b_s), len(a_s)), np.nan)
for i, b in enumerate(b_s):
    for j, a in enumerate(a_s):
        Kab = K * np.array([a, a, b, b])
        Acl = A - B @ Kab[None, :]
        growth[i, j] = np.linalg.eigvals(Acl).real.max()
        if growth[i, j] < 0:
            cost[i, j] = np.trace(vp.control.lyapunov(A=Acl, Q=-np.eye(4)).X)
stable = growth < 0
print("stable gains: %d of %d" % (stable.sum(), stable.size))
i1, j1 = list(b_s).index(1.0), list(a_s).index(1.0)
print(
    "at the LQR gain: slowest decay rate %.4f 1/s, tr X = %.4f"
    % (-growth[i1, j1], cost[i1, j1])
)

# %%
# ## Where the loop is stable
#
# The left map is the largest real part of the closed-loop eigenvalues:
# negative (blue) is stable. The stable region lies above a line: the pole
# gain has to grow with the cart gain, and a cart gain too large for the pole
# gain tips the pole over (red). At a = 0 nothing pulls the cart back, and
# one eigenvalue stays at zero. The second map is tr X where the loop is
# stable (blank elsewhere), on a log scale: small cart gains respond
# slowly and pay for it in tr X.

fig = figure(
    "Largest real part of the closed-loop eigenvalues",
    "a (cart gain scale)",
    "b (pole gain scale)",
    window=(0.0, 3.0, 0.0, 3.0),
)
fig.heatmap(np.clip(growth, -3.0, 3.0), "max Re lambda")
fig.origin("lower")
fig.colormap("diverging")
fig.label("lqr", "LQR", 1.0, 1.0)

fig = figure(
    "$\\log_{10} \\mathrm{tr}\\,X$, the disturbance response of the stable gains",
    "a (cart gain scale)",
    "b (pole gain scale)",
    window=(0.0, 3.0, 0.0, 3.0),
)
fig.heatmap(np.log10(cost), "log10 tr X")
fig.origin("lower")
fig.label("lqr", "LQR", 1.0, 1.0)
best = np.unravel_index(np.nanargmin(cost), cost.shape)
print(
    "smallest tr X on the grid: %.4f at a = %.2f, b = %.2f"
    % (cost[best], a_s[best[1]], b_s[best[0]])
)

# %%
# ## A slice through the map
#
# Along b = 1: the decay rate of the slowest mode and tr X against a. The
# LQR gain gives the fastest decay on this line. It does not give the
# smallest tr X on the map: LQR minimizes its own weighted cost, not the
# response to equal white noise on every state.

fig = figure("Along b = 1", "a (cart gain scale)", "")
fig.plot(a_s, -growth[i1], "decay rate of the slowest mode (1/s)")
fig.plot(a_s, cost[i1], "tr X")
fig.yright("tr X", "tr X")
fig.style("decay rate of the slowest mode (1/s)", "s1 nomarker")
fig.style("tr X", "s2 nomarker")
fig.vline("LQR", 1.0)
fig.hline("stability boundary", 0.0)

# %%
# ## How it's checked
#
# Every X on the map satisfies its equation: the largest relative residual
# |A X + X A^T + I| / (2 |A| |X| + 1) over the stable gains, and the largest
# difference from SciPy's `solve_continuous_lyapunov` at the LQR gain.

from scipy.linalg import solve_continuous_lyapunov

worst = 0.0
for i, b in enumerate(b_s[::5]):
    for j, a in enumerate(a_s[::5]):
        if stable[5 * i, 5 * j]:
            Acl = A - B @ (K * np.array([a, a, b, b]))[None, :]
            X = vp.control.lyapunov(A=Acl, Q=-np.eye(4)).X
            res = np.abs(Acl @ X + X @ Acl.T + np.eye(4)).max() / (
                2 * np.abs(Acl).max() * np.abs(X).max() + 1
            )
            worst = max(worst, res)
print("worst relative residual on every fifth grid point: %.1e" % worst)
Acl = A - B @ K[None, :]
X = vp.control.lyapunov(A=Acl, Q=-np.eye(4)).X
print(
    "max |X - X_scipy| / max |X_scipy| at the LQR gain: %.1e"
    % (
        np.abs(X - solve_continuous_lyapunov(Acl, -np.eye(4))).max()
        / np.abs(X).max()
    )
)

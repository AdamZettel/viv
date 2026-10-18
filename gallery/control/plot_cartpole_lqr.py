# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Balancing a cart-pole with LQR
==============================

Linearize a cart-pole about the upright, compute the optimal state feedback
from the continuous algebraic Riccati equation, and run the nonlinear system
under it from 25 degrees off vertical. The Riccati solver is Viv's Schur
method, an algorithm after SLICOT (`lib/are.viv`, `lib/schur.viv`); its residual is
checked here and its solution compared with SciPy's.
"""

# %%
# ## The model
#
# The state is (x, xdot, theta, thetadot): cart position and velocity, pole
# angle from vertical and its rate. `vp.control.cartpole_model` linearizes the
# nonlinear dynamics of `lib/cartpole.viv` (a 1 kg cart, a 0.1 kg pole of
# half-length 0.5 m) by central differences, so A and B describe the same
# system the simulation runs. The open loop has one positive eigenvalue: left
# alone, the pole falls.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

A, B = vp.control.cartpole_model(h=1e-5)
print("A =\n", np.round(A, 4))
print("B =", np.round(B.ravel(), 4))
print(
    "open-loop eigenvalues:", np.round(np.sort(np.linalg.eigvals(A).real), 4)
)

# %%
# ## The gain
#
# Q weights the state errors (the angle 20 times the others) and R the force.
# `vp.control.lqr` returns K = R^-1 B^T P, where P solves
# A^T P + P A - P B R^-1 B^T P + Q = 0, and the control is u = -K s.

Q = np.diag([1.0, 1.0, 20.0, 1.0])
R = np.array([[0.05]])
lqr = vp.control.lqr(A=A, B=B, Q=Q, R=R)
K, P = lqr.K, lqr.P
print("K =", np.round(K.ravel(), 4))
print(
    "closed-loop eigenvalues:",
    np.round(np.sort_complex(np.linalg.eigvals(A - B @ K)), 4),
)

# %%
# ## Checking the Riccati solution
#
# The relative residual is the equation's left side divided by the size of its
# terms; a backward-stable solver leaves a small multiple of the unit roundoff
# (1.1e-16). SciPy's `solve_continuous_are` is an independent implementation
# of the same method, so its P should agree to the problem's conditioning.

from scipy.linalg import solve_continuous_are

Rinv = np.linalg.inv(R)
res = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
scale = (
    2 * np.linalg.norm(A, 1) * np.linalg.norm(P, 1)
    + np.linalg.norm(P, 1) ** 2 * np.linalg.norm(B @ Rinv @ B.T, 1)
    + np.linalg.norm(Q, 1)
)
P_ref = solve_continuous_are(A, B, Q, R)
print("relative CARE residual: %.1e" % (np.linalg.norm(res, 1) / scale))
print(
    "max |P - P_scipy| / max |P_scipy|: %.1e"
    % (np.abs(P - P_ref).max() / np.abs(P_ref).max())
)
print("check():", lqr.check().passed)

# %%
# ## The closed loop
#
# `vp.control.simulate` runs the nonlinear cart-pole, not the linear model,
# under u = -K s for 6 seconds in steps of 10 ms. A second gain, with the
# force 20 times more expensive (R = 1), shows the trade the weights make: a
# gentler push and a slower recovery.

s0 = np.array([0.0, 0.0, np.radians(25.0), 0.0])
cheap = vp.control.simulate(K=K.ravel(), s0=s0, dt=0.01, steps=600).trajectory
K_dear = vp.control.lqr(A=A, B=B, Q=Q, R=np.array([[1.0]])).K
dear = vp.control.simulate(
    K=K_dear.ravel(), s0=s0, dt=0.01, steps=600
).trajectory

for name, tr in (("R = 0.05", cheap), ("R = 1", dear)):
    print(
        "%-8s  final angle %.1e rad   peak force %5.1f N   cart travels to x "
        "= %+.2f m"
        % (
            name,
            abs(tr[-1, 3]),
            np.abs(tr[:, 5]).max(),
            tr[np.argmax(np.abs(tr[:, 1])), 1],
        )
    )

fig = figure(
    "Pole angle under LQR, from 25 degrees", "t (s)", "$\\theta$ (degrees)"
)
fig.plot(cheap[::5, 0], np.degrees(cheap[::5, 3]), "$R = 0.05$")
fig.plot(dear[::5, 0], np.degrees(dear[::5, 3]), "$R = 1$")
fig.style("$R = 0.05$", "s1 nomarker")
fig.style("$R = 1$", "s2 dash nomarker")
fig.hline("upright", 0.0)

fig = figure("Force on the cart", "t (s)", "u (N)")
fig.plot(cheap[::5, 0], cheap[::5, 5], "$R = 0.05$")
fig.plot(dear[::5, 0], dear[::5, 5], "$R = 1$")
fig.style("$R = 0.05$", "s1 nomarker")
fig.style("$R = 1$", "s2 dash nomarker")

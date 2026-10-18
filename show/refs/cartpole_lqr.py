# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The reference gains for show/cartpole_lqr.viv.

The cart-pole of lib/cartpole.viv, linearized at upright by the same central
differences, and the Riccati equation solved by SciPy's
scipy.linalg.solve_continuous_are. Prints K = R^-1 B^T P to paste into
SCIPY_K.

    python3 show/refs/cartpole_lqr.py
"""
import math

import numpy as np
import scipy
from scipy.linalg import solve_continuous_are

CART_M, POLE_M, POLE_L, GRAVITY = 1.0, 0.1, 0.5, 9.81


def rate(s, force):
    th, thd = s[2], s[3]
    total = CART_M + POLE_M
    sn, cs = math.sin(th), math.cos(th)
    tmp = (force + POLE_M * POLE_L * thd * thd * sn) / total
    thdd = (GRAVITY * sn - cs * tmp) / (POLE_L * (4.0 / 3.0 - POLE_M * cs * cs / total))
    xdd = tmp - POLE_M * POLE_L * thdd * cs / total
    return np.array([s[1], xdd, s[3], thdd])


eps = 1.0e-5
A = np.zeros((4, 4))
for j in range(4):
    zp = np.zeros(4)
    zm = np.zeros(4)
    zp[j], zm[j] = eps, -eps
    A[:, j] = (rate(zp, 0.0) - rate(zm, 0.0)) / (2 * eps)
B = ((rate(np.zeros(4), eps) - rate(np.zeros(4), -eps)) / (2 * eps)).reshape(4, 1)
Q = np.eye(4)
Q[2, 2] = 20.0
R = np.array([[0.05]])
P = solve_continuous_are(A, B, Q, R)
K = (B.T @ P / R[0, 0])[0]
print("scipy", scipy.__version__)
print("SCIPY_K = [" + ", ".join(repr(float(k)) for k in K) + "]")

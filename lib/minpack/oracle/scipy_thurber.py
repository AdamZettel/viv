# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""scipy.optimize.leastsq on NIST's Thurber data, for the post's tests.

    python3 lib/minpack/oracle/scipy_thurber.py

Same model, residual sign, settings and starting points as oracle/thurber.f
and the post (posts/2026-09-25-fitting-electron-mobility-with-minpack.viv).
Prints info, nfev, njev, the final norm and x for each start, each number as
the shortest decimal that reads back as the same double.
"""
import os
import numpy as np
from scipy.optimize import leastsq

here = os.path.dirname(os.path.abspath(__file__))
lines = open(os.path.join(here, '..', 'data', 'Thurber.dat')).read().splitlines()
d = np.array([[float(v) for v in l.split()] for l in lines[60:97]])
y, x = d[:, 0], d[:, 1]
x2 = x * x
x3 = x2 * x


def f(b):
    num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3
    den = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3
    return num / den - y


def jac(b):
    num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3
    den = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3
    t = num / (den * den)
    return np.column_stack([1.0 / den, x / den, x2 / den, x3 / den, -(x * t), -(x2 * t), -(x3 * t)])


tol = np.sqrt(2.22044604926e-16)
for s in ([1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03], [1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05]):
    b, cov, info, msg, ier = leastsq(f, np.array(s), Dfun=jac, full_output=True, ftol=tol, xtol=tol,
                                     gtol=0.0, maxfev=800, factor=100.0)
    print('info', ier, 'nfev', info['nfev'], 'njev', info['njev'], 'fnorm', repr(float(np.linalg.norm(f(b)))))
    print('x', ', '.join(repr(float(v)) for v in b))

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-make-numerov-find-the-decaying-solution.viv (a problem).

y'' = y on [0, m h]. The decaying solution is e^{-x}. Numerov's recurrence,
c y_{i+1} = 2 d y_i - c y_{i-1} with c = 1 - h^2/12 and d = 1 + 5 h^2/12,
is run here in numpy both ways: outward from y_0 = 1, y_1 = e^{-h}, and
inward from y_m = e^{-mh}, y_{m-1} = e^{-(m-1)h}, then divided by y_0 so that
y(0) = 1. The errors are relative to e^{-x}.

    python3 posts/refs/make-numerov-find-the-decaying-solution.py > posts/refs/make-numerov-find-the-decaying-solution.out
"""
import numpy as np


def coeffs(h):
    return 1 - h * h / 12, 1 + 5 * h * h / 12


def outward(m, h):
    c, d = coeffs(h)
    y = np.zeros(m + 1)
    y[0], y[1] = 1.0, np.exp(-h)
    for i in range(m - 1):
        y[i + 2] = (2 * d * y[i + 1] - c * y[i]) / c
    return y


def inward(m, h):
    c, d = coeffs(h)
    y = np.zeros(m + 1)
    y[m], y[m - 1] = np.exp(-h * m), np.exp(-h * (m - 1))
    for j in range(m - 2, -1, -1):
        y[j] = (2 * d * y[j + 1] - c * y[j + 2]) / c
    return y / y[0]


def err(y, h):
    x = h * np.arange(len(y))
    return np.max(np.abs(y / np.exp(-x) - 1))


for m, h in ((60, 0.15), (120, 0.075), (120, 0.15)):
    yo = outward(m, h)
    yi = inward(m, h)
    print(f"m = {m}, h = {h} (x to {m * h:g}): outward max relative error {err(yo, h):.6e}, "
          f"inward {err(yi, h):.6e}; inward y[m] = {yi[-1]!r}")
e1 = err(inward(60, 0.15), 0.15)
e2 = err(inward(120, 0.075), 0.075)
print(f"halving h divides the inward error by {e1 / e2:.4f} (fourth order: 16)")

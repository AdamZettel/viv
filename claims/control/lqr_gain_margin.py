# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of lqr_gain_margin.viv: the model from
vp.control.cartpole_model, the gain from vp.control.lqr, the eigenvalues
by NumPy (LAPACK's dgeev) instead of lib/schur.

###
python: ~/.venvs/power/bin/python
###
"""
import numpy as np
import vipyr as vp
from claimlib import show, test

m = vp.control.cartpole_model(h=1e-6)
A, B = np.asarray(m.A), np.asarray(m.B).reshape(4, 1)
r = vp.control.lqr(A=A, B=B, Q=np.eye(4), R=np.eye(1))
K = np.asarray(r.K).reshape(1, 4)


def slowest(g):
    return float(np.linalg.eigvals(A - g * B @ K).real.max())


scales = [0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
lo, hi = 0.0, 0.5
while hi - lo > 1e-6:
    mid = 0.5 * (lo + hi)
    lo, hi = (lo, mid) if slowest(mid) < 0 else (mid, hi)
show("open_loop", float(np.linalg.eigvals(A).real.max()))
show("slowest_nominal", slowest(1.0))
show("worst_slowest", max(slowest(g) for g in scales))
show("lowest_stable_scale", hi)


@test("the open loop is unstable")
def _():
    assert np.linalg.eigvals(A).real.max() > 0


@test("stable at every scaling from 1/2 to 1000")
def _():
    assert max(slowest(g) for g in scales) < 0


@test("the lowest stabilizing scaling is below 1/2, as the guarantee allows")
def _():
    assert hi < 0.5, hi

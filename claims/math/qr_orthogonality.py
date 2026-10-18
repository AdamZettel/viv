# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of qr_orthogonality.viv: the same four factorizations
through vp.load of claims/math/qrlib.viv, and LAPACK's Householder QR
(numpy.linalg.qr) on the same matrices as an independent check.

###
python: ~/.venvs/power/bin/python
###
"""
import numpy as np
import vipyr as vp
from claimlib import show, test

N_SIZE, N_KAPPA, SEED, CGS_FIT = 40, 12, 7, 6
U = 2.0**-53
qr = vp.load("claims/math/qrlib.viv")
kappas, lh, lg, lm, lc = qr.orthogonality_losses(N_SIZE, N_KAPPA, SEED)
lapack = np.array([
    np.linalg.norm(q.T @ q - np.eye(N_SIZE))
    for q in (np.linalg.qr(qr.seeded_condition(N_SIZE, kap, SEED + k))[0] for k, kap in enumerate(kappas))
])


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


nu = N_SIZE * U
show("worst_householder_in_nu", lh.max() / nu)
show("worst_givens_in_nu", lg.max() / nu)
show("slope_householder", slope(kappas, lh))
show("slope_givens", slope(kappas, lg))
show("slope_mgs", slope(kappas, lm))
show("slope_cgs", slope(kappas[:CGS_FIT], lc[:CGS_FIT]))
show("worst_lapack_in_nu", lapack.max() / nu)
show("slope_lapack", slope(kappas, lapack))


@test("Householder (Viv's and LAPACK's) and Givens QR stay below 10 n u for every kappa")
def _():
    assert max(lh.max(), lg.max(), lapack.max()) < 10 * nu


@test("their loss does not grow with kappa: slope below 0.1")
def _():
    assert max(abs(slope(kappas, lh)), abs(slope(kappas, lg)), abs(slope(kappas, lapack))) < 0.1


@test("modified Gram-Schmidt grows like u kappa: slope between 0.8 and 1.2")
def _():
    assert 0.8 < slope(kappas, lm) < 1.2


@test("classical Gram-Schmidt grows like u kappa^2: slope between 1.6 and 2.4 up to kappa = 1e6")
def _():
    assert 1.6 < slope(kappas[:CGS_FIT], lc[:CGS_FIT]) < 2.4


@test("classical Gram-Schmidt has no orthogonality left from kappa = 1e8 on: loss above 1")
def _():
    assert kappas[np.argmax(lc > 1.0)] == 1e8 and (lc[kappas >= 1e8] > 1.0).all()

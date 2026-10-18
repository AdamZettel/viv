# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of dc_error_with_loading.viv: the same scaled cases
solved by vp.power.runpf, NR and DC.

###
python: ~/.venvs/power/bin/python
###
"""
import numpy as np
import vipyr as vp
from claimlib import show, test
from vipyr import _power

PD, QD, PG = 2, 3, 1
base = _power.load_case("case118")[0]
loads = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
err, ok = [], 0
for lam in loads:
    c = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    c["bus"][:, PD] *= lam
    c["bus"][:, QD] *= lam
    c["gen"][:, PG] *= lam
    ac = vp.power.runpf(c, method="nr", tol=1e-8, max_it=30)
    dc = vp.power.runpf(c, method="dc", tol=1e-8, max_it=30)
    ok += bool(ac.success)
    err.append(np.abs(np.asarray(dc.pf) - np.asarray(ac.pf)).max() / np.abs(np.asarray(ac.pf)).max())
show("converged", ok)
show("dc_error_half", err[0])
show("dc_error_base", err[1])
show("dc_error_triple", err[-1])


@test("the AC flow converges at all six loadings")
def _():
    assert ok == 6


@test("the DC error grows at every step of loading")
def _():
    assert all(b > a for a, b in zip(err, err[1:])), err


@test("it is between 10% and 17% at base load and above 40% at three times base load")
def _():
    assert 0.10 < err[1] < 0.17 and err[-1] > 0.40, err

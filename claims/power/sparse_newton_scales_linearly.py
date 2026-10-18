# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of sparse_newton_scales_linearly.viv: the Newton solves
through vp.power.runpf.

###
python: ~/.venvs/power/bin/python
###
"""
import time

import numpy as np
from claimlib import show, test
import vipyr as vp
from vipyr import _power

CASES = ["case9", "case14", "case30", "case118", "case300", "case2383wp"]
REPS = 5


def best(fn, reps):
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        t = min(t, time.perf_counter() - t0)
    return out, t


rows = []
for name in CASES:
    c = _power.load_case(name)[0]
    nb = c["bus"].shape[0]
    r, t_nr = best(lambda: vp.power.runpf(c, method="nr"), REPS)
    rows.append((name, nb, r.iterations, r.success, t_nr))
    show(f"{name}.iterations", r.iterations)


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


big = [r for r in rows if r[1] >= 118]
s_sparse = slope([r[1] for r in big], [r[4] for r in big])
show("time_slope_sparse", s_sparse)
show("max_iterations", max(r[2] for r in rows))
show("min_buses", min(r[1] for r in rows))
show("max_buses", max(r[1] for r in rows))


@test("Newton converges on every case in at most 6 iterations")
def _():
    assert all(r[3] for r in rows), [r[0] for r in rows if not r[3]]
    assert max(r[2] for r in rows) <= 6, [(r[0], r[2]) for r in rows]


@test("the sparse Newton solve's time grows with a log-log slope between 1.0 and 1.6")
def _():
    assert 1.0 <= s_sparse <= 1.6, s_sparse

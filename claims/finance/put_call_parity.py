# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of put_call_parity.viv: the same grid through
vp.finance.bs_price and vp.finance.crr_price.

###
python: ~/.venvs/power/bin/python
###
"""
import math

import vipyr as vp
from claimlib import show, test

K, R, Q = 100.0, 0.03, 0.0
worst, gap, prem, n = 0.0, 0.0, math.inf, 0
for a in range(10):
    S = 60.0 + 10.0 * a
    for b in range(6):
        T = 0.1 + 0.58 * b
        for c in range(50):
            sig = 0.1 + 0.4 * c / 49.0
            call = vp.finance.bs_price(S=S, K=K, T=T, r=R, q=Q, sigma=sig, is_call=True).price
            put = vp.finance.bs_price(S=S, K=K, T=T, r=R, q=Q, sigma=sig, is_call=False).price
            worst = max(worst, abs(call - put - (S * math.exp(-Q * T) - K * math.exp(-R * T))) / S)
            if n % 30 == 0:
                tree = lambda c, am: vp.finance.crr_price(S=S, K=K, T=T, r=R, q=Q, sigma=sig, N=400, is_call=c, american=am).price
                gap = max(gap, abs(tree(True, True) - tree(True, False)))
                prem = min(prem, tree(False, True) - tree(False, False))
            n += 1
show("n_quotes", n)
show("worst_parity", worst)
show("worst_call_gap", gap)
show("min_put_premium", prem)


@test("3,000 quotes satisfy put-call parity to 1e-15 of the spot")
def _():
    assert n == 3000 and worst < 1e-15, worst


@test("the American call equals the European call exactly")
def _():
    assert gap == 0.0, gap


@test("the American put is never worth less than the European put")
def _():
    assert prem >= 0.0, prem

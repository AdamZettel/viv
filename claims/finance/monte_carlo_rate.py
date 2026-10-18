# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of monte_carlo_rate.viv: the reference price from
vp.finance.bs_price, the Monte Carlo runs in NumPy (its own generator, so
its errors are its own; only the price and the slope's band are shared).

###
python: ~/.venvs/power/bin/python
###
"""
import numpy as np
import vipyr as vp
from claimlib import show, test

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0
price = float(vp.finance.bs_price(S=S0, K=K, T=T, r=R, q=0.0, sigma=SIG, is_call=True).price)
show("bs_price", price)
rng = np.random.default_rng(99)
paths = [1000, 10000, 100000, 1000000]
rms = []
for n in paths:
    est = []
    for _ in range(16):
        st = S0 * np.exp((R - 0.5 * SIG**2) * T + SIG * np.sqrt(T) * rng.standard_normal(n))
        est.append(np.exp(-R * T) * np.maximum(st - K, 0.0).mean())
    rms.append(np.sqrt(np.mean((np.array(est) - price) ** 2)))
slope = float(np.polyfit(np.log(paths), np.log(rms), 1)[0])
show("numpy_mc_slope", slope)


@test("vp.finance.bs_price gives 10.4506")
def _():
    assert abs(price - 10.450583572185565) < 1e-12, price


@test("the RMS error falls with a log-log slope between -0.6 and -0.4")
def _():
    assert -0.6 < slope < -0.4, slope

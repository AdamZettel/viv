# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Implied volatility, binomial trees and Greeks of European options
=================================================================

Recover the implied volatility of 3,000 option quotes and plot the smile,
watch a Cox-Ross-Rubinstein tree converge to the Black-Scholes price, and
compare Greeks from Viv's reverse-mode `grad` with the closed forms. The
pricing code is `examples/options/*.viv`, loaded as Python modules with
`vp.load`.
"""

# %%
# Load the Black-Scholes and Greeks programs. `vp.load` compiles each `.viv`
# file to C once and caches the library; every annotated `def` becomes a
# Python function on NumPy arrays.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

bs = vp.load("examples/options/bs.viv")
greeks = vp.load("examples/options/greeks.viv")

# %%
# ## Implied volatility of 3,000 quotes
#
# The quotes are synthetic: prices made from a known smile,
# sigma(k) = 0.20 - 0.10 k + 0.25 k^2 in the log-moneyness k = ln(K/F),
# at three maturities. Inverting every price recovers that smile, so the
# round-trip error says how accurate `implied_vols` is.

S, r, q = 100.0, 0.03, 0.01
maturities = [0.25, 1.0, 2.0]
n_per = 1000
K = np.concatenate([np.linspace(60.0, 160.0, n_per) for _ in maturities])
T = np.repeat(maturities, n_per)
F = S * np.exp((r - q) * T)
k = np.log(K / F)
true_vol = 0.20 - 0.10 * k + 0.25 * k**2
# out of the money: calls above the forward, puts below
is_call = (K >= F).astype(float)

price = bs.bs_prices(np.full_like(K, S), K, T, r, q, true_vol, is_call)
iv = bs.implied_vols(price, np.full_like(K, S), K, T, r, q, is_call)
rel = np.abs(iv - true_vol) / true_vol
print("quotes:", len(price))
print("worst relative error of the recovered volatility: %.1e" % rel.max())

fig = figure(
    "Implied volatility smile, recovered from 3,000 prices",
    "strike K",
    "implied volatility",
)
for t in maturities:
    m = T == t
    fig.plot(K[m][::40], iv[m][::40], "T = %g recovered" % t)
    fig.plot(K[m][::10], true_vol[m][::10], "T = %g input" % t)
for i, t in enumerate(maturities):
    fig.style("T = %g recovered" % t, "s%d noline circle" % (i + 1))
    fig.style("T = %g input" % t, "s%d dash thin nomarker" % (i + 1))

# %%
# ## Binomial tree convergence to Black-Scholes
#
# `vp.finance.crr_price` prices a European call on an N-step tree. The error
# against the closed form falls like 1/N and oscillates, because the strike
# sits at a different place between the tree's nodes as N changes.

option = dict(S=S, K=100.0, T=1.0, r=r, q=q, sigma=0.2, is_call=True)
exact = vp.finance.bs_price(**option).price
Ns = [10, 20, 40, 80, 160, 320, 640, 1280]
err = []
for N in Ns:
    p = vp.finance.crr_price(**option, N=N, american=False).price
    err.append(abs(p - exact))
    print("N = %4d   tree %.8f   |error| %.2e" % (N, p, abs(p - exact)))
slope = np.polyfit(np.log(Ns), np.log(err), 1)[0]
print("fitted slope of log|error| against log N: %.2f" % slope)

fig = figure(
    "Cox-Ross-Rubinstein error against tree depth",
    "steps N",
    "|tree - Black-Scholes|",
)
fig.plot(Ns, err, "CRR, European call")
fig.plot(Ns, [err[0] * Ns[0] / n for n in Ns], "1/N")
fig.style("1/N", "s2 dash thin nomarker")
fig.xscale("log")
fig.yscale("log")

# %%
# ## Greeks by automatic differentiation
#
# `bs_greeks_arr` takes delta and vega from Viv's reverse-mode `grad` of the
# pricing function, and gamma from the identity gamma = vega / (S^2 sigma T).
# The closed forms below are written in NumPy, independently.

from math import erf, sqrt, pi


def closed_form(S, K, T, r, q, s):
    d1 = (np.log(S / K) + (r - q + 0.5 * s * s) * T) / (s * np.sqrt(T))
    Nd1 = 0.5 * (1.0 + np.vectorize(erf)(d1 / sqrt(2.0)))
    pdf = np.exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
    delta = np.exp(-q * T) * Nd1
    gamma = np.exp(-q * T) * pdf / (S * s * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf * np.sqrt(T)
    return delta, gamma, vega


Kg = np.linspace(70.0, 130.0, 61)
Sg = np.full_like(Kg, S)
Tg = np.full_like(Kg, 0.5)
sg = np.full_like(Kg, 0.22)
d_ad, g_ad, v_ad = greeks.bs_greeks_arr(Sg, Kg, Tg, r, q, sg, np.ones_like(Kg))
d_cf, g_cf, v_cf = closed_form(Sg, Kg, Tg, r, q, sg)
for name, a, b in (
    ("delta", d_ad, d_cf),
    ("gamma", g_ad, g_cf),
    ("vega", v_ad, v_cf),
):
    print(
        "%-5s  worst relative difference from the closed form: %.1e"
        % (name, np.max(np.abs(a - b) / np.abs(b)))
    )

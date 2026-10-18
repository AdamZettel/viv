# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Early exercise of an American put, priced two ways
==================================================

Price an American put, which may be exercised at any time before expiry, with
two independent methods: Crank-Nicolson finite differences on the
Black-Scholes PDE with the early-exercise constraint
(`vp.finance.cn_american_put`), and a Cox-Ross-Rubinstein binomial tree
(`vp.finance.crr_price`). Plot the price against the stock price, the value
of the right to exercise early, and the tree's convergence to the PDE.
"""

# %%
# ## Hull's example
#
# A put struck at 50 on a stock at 50, five months to expiry, 10% rates and
# 40% volatility (Hull, Options, Futures and Other Derivatives, example
# 21.1). The PDE runs on 3,200 stock-price steps up to S = 200 and 1,600
# time steps, the first 4 of them fully implicit to damp the payoff's kink;
# the tree takes 20,000 steps, averaged with 20,001 (the price oscillates
# between even and odd step counts).

import numpy as np
import vipyr as vp
from vipyr.figure import figure

par = dict(K=50.0, T=5.0 / 12.0, r=0.1, q=0.0, sigma=0.4)
cn = vp.finance.cn_american_put(
    S0=50.0, Smax=200.0, M=3200, Nt=1600, implicit=4, **par
).price
tree = [
    vp.finance.crr_price(
        S=50.0, N=N, is_call=False, american=True, **par
    ).price
    for N in (20000, 20001)
]
euro = vp.finance.bs_price(S=50.0, is_call=False, **par).price
print("American put, Crank-Nicolson:   %.5f" % cn)
print(
    "American put, CRR N = 20000/20001: %.5f %.5f (mean %.5f)"
    % (tree[0], tree[1], np.mean(tree))
)
print("European put, Black-Scholes:     %.5f" % euro)
print(
    "Hull's five-step tree: %.2f"
    % vp.finance.crr_price(
        S=50.0, N=5, is_call=False, american=True, **par
    ).price
)

# %%
# ## Price against the stock price
#
# Deep in the money the American put is worth exactly K - S: exercising at
# once is optimal, so its price meets the payoff line, while the European
# put falls below it: its holder must wait for K, and waiting costs
# interest. The gap between the two, the early-exercise premium, grows the
# deeper the put is in the money. This curve uses a coarser
# grid, 400 by 200, good to about 1e-3.

S = np.arange(25.0, 90.0, 2.5)
am = np.array(
    [
        vp.finance.cn_american_put(
            S0=s, Smax=200.0, M=400, Nt=200, implicit=4, **par
        ).price
        for s in S
    ]
)
eu = np.array(
    [vp.finance.bs_price(S=s, is_call=False, **par).price for s in S]
)
fig = figure("Put value, K = 50, five months", "stock price S", "value")
fig.plot(S, am, "American (Crank-Nicolson)")
fig.plot(S, eu, "European (Black-Scholes)")
fig.plot(S, np.maximum(50.0 - S, 0.0), "payoff K - S")
fig.style("American (Crank-Nicolson)", "s1 nomarker")
fig.style("European (Black-Scholes)", "s2 nomarker")
fig.style("payoff K - S", "s6 dash thin nomarker")
fig = figure(
    "Early-exercise premium, American minus European",
    "stock price S",
    "premium",
)
fig.plot(S, am - eu, "premium")
i = int(np.argmax(am - eu))
print(
    "largest premium %.4f at S = %.1f; at S = 25 the American put is %.5f, K "
    "- S = %.1f" % ((am - eu)[i], S[i], am[0], 50 - S[0])
)

# %%
# ## Each method converges to the other
#
# The tree's price (the mean of N and N + 1 steps) against the finest PDE
# value, and the PDE's price on grids of M stock steps (and M / 2 time
# steps) against the finest tree value. The tree's error halves with each
# doubling of N, like 1/N. The PDE's falls by about three per doubling once
# the grid is fine enough, until it meets the reference tree's own error.

Ns = [50, 100, 200, 400, 800, 1600, 3200]
tr = [
    np.mean(
        [
            vp.finance.crr_price(
                S=50.0, N=N + k, is_call=False, american=True, **par
            ).price
            for k in (0, 1)
        ]
    )
    for N in Ns
]
pde = [
    vp.finance.cn_american_put(
        S0=50.0, Smax=200.0, M=N, Nt=N // 2, implicit=4, **par
    ).price
    for N in Ns
]
fig = figure(
    "Tree and PDE against each other's finest value",
    "steps N (tree) or M (PDE)",
    "|error|",
)
fig.plot(Ns, [abs(p - cn) for p in tr], "CRR tree, mean of N and N + 1")
fig.plot(
    Ns, [abs(p - np.mean(tree)) for p in pde], "Crank-Nicolson, M x M/2 grid"
)
fig.xscale("log")
fig.yscale("log")
for N, a, b in zip(Ns, tr, pde):
    print("N = M = %4d   tree %.5f   PDE %.5f" % (N, a, b))

# %%
# ## How it's checked
#
# The two methods share no code: one marches a grid backward in time and
# projects onto the payoff at each step, the other works back through a
# recombining tree. Their difference here, and `check()`, which holds
# `crr_price` to Hull's printed American put values for N = 5 to 500.

print(
    "|CN(3200 x 1600) - mean of CRR 20000/20001| = %.1e"
    % abs(cn - np.mean(tree))
)
print(
    "check():",
    vp.finance.crr_price(S=50.0, N=5, is_call=False, american=True, **par)
    .check()
    .passed,
)

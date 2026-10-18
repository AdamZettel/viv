# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The Heston model's volatility smile
===================================

Price European options under the Heston stochastic-volatility model with
`vp.finance.heston_cos`, Fang and Oosterlee's COS method (a Fourier-cosine
expansion of the density), convert the prices to Black-Scholes implied
volatilities, and plot the smile at three maturities and for three
correlations between the stock and its variance. The COS price converges to
Fang and Oosterlee's published value as the number of terms grows; the
convergence plot shows how fast.
"""

# %%
# ## Parameters
#
# Fang and Oosterlee's test case (SIAM J. Sci. Comput. 31, 826 (2008)): the
# variance starts at v0 = 0.0175 and reverts at rate kappa = 1.5768 to theta
# = 0.0398, its own volatility is eta = 0.5751, and its shocks are correlated
# with the stock's by rho = -0.5711. S = 100, no rates or dividends. The
# expansion uses N = 1024 cosine terms on an interval of L = 32 standard
# deviations of the log price, the settings of the program's own test.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

par = dict(
    S=100.0,
    r=0.0,
    q=0.0,
    v0=0.0175,
    kappa=1.5768,
    theta=0.0398,
    eta=0.5751,
    rho=-0.5711,
)
bs = vp.load("examples/options/bs.viv")


def heston(K, T, N=1024, L=32.0, **kw):
    p = dict(par, **kw)
    calls = np.ones_like(K)
    return vp.finance.heston_cos(K=K, T=T, N=N, L=L, is_call=calls, **p).prices


def implied(price, K, T):
    S = np.full_like(K, par["S"])
    T = np.full_like(K, T)
    calls = np.ones_like(K)
    r = q = 0.0
    return bs.implied_vols(price, S, K, T, r, q, calls)


p = heston(np.array([100.0]), 1.0)[0]
print(
    "at the money, T = 1: %.9f   Fang and Oosterlee: 5.785155450   |diff| %.1e"
    % (p, abs(p - 5.785155450))
)

# %%
# ## The smile at three maturities
#
# Negative correlation makes the implied volatility fall with the strike (a
# skew): when the stock drops, variance tends to rise, which fattens the
# left tail of the return distribution. The skew flattens with maturity as
# the variance reverts to theta = 0.0398 (a volatility of 0.20).

K = np.linspace(70.0, 140.0, 57)
fig = figure(
    "Implied volatility under Heston", "strike K", "implied volatility"
)
for T in (0.25, 1.0, 3.0):
    iv = implied(heston(K, T), K, T)
    fig.plot(K, iv, "T = %g" % T)
    fig.style("T = %g" % T, "nomarker")
    print(
        "T = %-4g  implied vol at K = 80, 100, 120: %.4f %.4f %.4f"
        % (T, *np.interp([80.0, 100.0, 120.0], K, iv))
    )
fig.hline("sqrt(theta)", np.sqrt(par["theta"]))

# %%
# ## The correlation sets the skew
#
# At T = 1, three values of rho. With rho = 0 the smile is symmetric in
# log-strike; the sign of rho decides which wing is higher.

fig = figure(
    "Implied volatility at T = 1 for three correlations",
    "strike K",
    "implied volatility",
)
for rho in (-0.7, 0.0, 0.5):
    iv = implied(heston(K, 1.0, rho=rho), K, 1.0)
    fig.plot(K, iv, "$\\rho$ = %g" % rho)
    fig.style("$\\rho$ = %g" % rho, "nomarker")
    print(
        "rho = %+.1f   iv(80) - iv(120) = %+.4f"
        % (rho, np.interp(80.0, K, iv) - np.interp(120.0, K, iv))
    )

# %%
# ## Convergence in the number of terms
#
# Two parameters set the COS method's accuracy: the number of terms N and
# the width L of the interval the density is expanded on. Against the
# published value at the money, T = 1: with L = 12 the error stops falling
# at 4e-5, the density mass the interval cuts off; with L = 32 it falls
# until it meets the published value's own last digit (1.6e-8 from an
# accurate quadrature). A wider interval needs more terms to resolve the
# same density.

Ns = [32, 64, 128, 192, 256, 384, 512, 768, 1024]
fig = figure(
    "COS price error against the number of terms", "N", "|price - 5.785155450|"
)
for L in (12.0, 32.0):
    errs = [
        abs(heston(np.array([100.0]), 1.0, N=N, L=L)[0] - 5.785155450)
        for N in Ns
    ]
    print(
        "L = %2.0f   " % L
        + "  ".join("N=%d: %.1e" % (N, e) for N, e in zip(Ns, errs))
    )
    fig.plot(Ns, errs, "L = %g" % L)
fig.xscale("log")
fig.yscale("log")

# %%
# ## How it's checked
#
# `heston_cos` matches Fang and Oosterlee's value to 1.6e-8 (they print
# nine decimals) and, on 21 options, SciPy's quadrature of the Gil-Pelaez
# integrals to 2e-13. `check()` reruns those cases.

print(
    "check():",
    vp.finance.heston_cos(
        K=np.array([100.0]),
        T=1.0,
        N=1024,
        L=32.0,
        is_call=np.array([1.0]),
        **par
    )
    .check()
    .passed,
)

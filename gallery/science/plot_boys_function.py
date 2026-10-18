# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
r"""
The Boys function, the integral behind Gaussian basis sets
==========================================================

Evaluate the Boys functions $F_n(T) = \int_0^1 t^{2n} e^{-Tt^2}\,dt$ with
`vp.special.boys`, which returns orders 0 through n at
once, and measure their relative accuracy against mpmath at 50 digits over
$T$ from 0 to 200. Every electron-repulsion and nuclear-attraction integral
over Gaussian orbitals reduces to these functions, so a Hartree-Fock code
calls them millions of times. This one is `lib/hf/boys.viv`, the routine
Viv's Hartree-Fock library computes its integrals with.
"""

# %%
# ## The functions
#
# $F_n(0) = 1/(2n + 1)$, and for large $T$,
#
# $$F_n(T) \approx \frac{(2n - 1)!!}{2^{n+1}} \sqrt{\frac{\pi}{T^{2n+1}}}.$$
#
# Higher orders fall off faster.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

Ts = np.concatenate([np.linspace(0.0, 2.0, 21), np.geomspace(2.2, 200.0, 60)])
F = np.array([vp.special.boys(n_max=16, T=float(T)).F for T in Ts])
print(
    "F_n(0):",
    np.round(F[0, :5], 6),
    "  1/(2n+1):",
    np.round(1 / (2 * np.arange(5) + 1.0), 6),
)
fig = figure("Boys functions $F_n(T)$", "$T$", "$F_n(T)$")
for n in (0, 2, 4, 8, 16):
    fig.plot(Ts[1:], F[1:, n], "n = %d" % n)
    fig.style("n = %d" % n, "nomarker")
fig.xscale("log")
fig.yscale("log")

# %%
# ## Accuracy against mpmath
#
# The reference is $F_n(T) = \Gamma(n + \tfrac12)\, P(n + \tfrac12, T) / (2T^{n + 1/2})$,
# with $P$ the regularized lower incomplete gamma function, at 50 digits. The
# program's tolerance is 5e-15 relative, worst over every order.

import mpmath

mpmath.mp.dps = 50


def boys_mp(n, T):
    if T == 0:
        return mpmath.mpf(1) / (2 * n + 1)
    a = mpmath.mpf(n) + mpmath.mpf(1) / 2
    return mpmath.gammainc(a, 0, T) / (2 * mpmath.mpf(T) ** a)


orders = (0, 4, 8, 16)
rel = {
    n: [
        float(
            abs(F[i, n] - boys_mp(n, mpmath.mpf(float(T))))
            / boys_mp(n, mpmath.mpf(float(T)))
        )
        for i, T in enumerate(Ts)
    ]
    for n in orders
}
fig = figure("Relative error against mpmath, 50 digits", "$T$", "relative error")
for n in orders:
    fig.plot(Ts[1:], np.maximum(rel[n][1:], 1e-18), "n = %d" % n)
    fig.style("n = %d" % n, "noline circle")
fig.hline("5e-15", 5e-15)
fig.xscale("log")
fig.yscale("log")
for n in orders:
    print(
        "n = %2d   worst relative error over %d values of T: %.1e"
        % (n, len(Ts), max(rel[n]))
    )

# %%
# ## How it's checked
#
# `check()` reruns the recorded cases against the same mpmath formula; the
# tolerance comes from `lib/hf/boys`'s own measurement, 1.7e-15 on its table.

print("check():", vp.special.boys(n_max=16, T=2.0).check().passed)

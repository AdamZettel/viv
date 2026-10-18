# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The Lennard-Jones fluid against its equation of state
=====================================================

Simulate 256 Lennard-Jones atoms by molecular dynamics at constant
temperature with `vp.liquids.md_nvt`, the Viv port of Allen and Tildesley's
md_nvt_lj, at several state points, and compare the energy and pressure
with the equation of state Thol et al. fitted to simulations of the same
model (`vp.liquids.eos`). Along the way, g(r) shows the fluid's structure
change from a gas to a dense liquid. Every program here gives the bits of
upstream's Fortran from the same start (`ports/liquids/COMPARISON.md`).
"""

# %%
# ## Equilibrate, then measure
#
# A run starts from an fcc lattice, which melts within the first few time
# units. As with upstream, where one run's `cnf.out` becomes the next run's
# `cnf.inp`, the first run here only equilibrates and the second, started
# from its final positions and velocities, is the one that is averaged. The
# potential is cut and shifted at r_c = 2.5 sigma, the model Thol et al.'s
# 2015 fit describes. Errors are the standard errors of 5 block averages.

import numpy as np
import vipyr as vp
from vipyr.figure import figure


def simulate(density, temperature, sample=0):
    start = vp.liquids.md_nvt(
        n=256, density=density, temperature=temperature, nblock=2, nstep=1000
    )
    return vp.liquids.md_nvt(
        n=256,
        density=density,
        temperature=temperature,
        nblock=5,
        nstep=2000,
        seed=2,
        r_start=start.r_final,
        v_start=start.v_final,
        box_start=start.box,
        sample=sample,
        gr_dr=0.02,
    )


states = [(0.1, 2.0), (0.5, 1.5), (0.75, 1.0), (0.85, 0.9)]
runs = {s: simulate(*s, sample=20) for s in states}
print(
    " rho    T     E/N (MD)            E/N (EOS)   P (MD)            P (EOS)"
)
for (rho, T), r in runs.items():
    e = vp.liquids.eos(temperature=T, density=rho)
    print(
        "%.2f  %.1f  %8.4f +- %.4f  %8.4f   %7.4f +- %.4f  %7.4f"
        % (rho, T, r.e_cs, r.e_cs_err, e.e_cs, r.p_cs, r.p_cs_err, e.p_cs)
    )

# %%
# ## The pair distribution function
#
# g(r) is the density of atoms at distance r from an atom, over the bulk
# density. In the gas it is the Boltzmann factor exp(-v(r) / T) with a
# little structure; in the dense liquid it has a sharp first peak near
# 2^(1/6) sigma, the potential's minimum, and shells beyond it that decay
# within a few diameters.

fig = figure("$g(r)$ of the Lennard-Jones fluid", "$r/\\sigma$", "$g(r)$")
for (rho, T), r in runs.items():
    fig.plot(r.gr_r, r.gr, "$\\rho$ = %.2f, $T$ = %.1f" % (rho, T))
fig.hline("1", 1.0)
for (rho, T), r in runs.items():
    k = int(np.argmax(r.gr))
    print(
        "rho = %.2f, T = %.1f: first peak g = %.2f at r = %.2f"
        % (rho, T, r.gr[k], r.gr_r[k])
    )

# %%
# ## The pressure along a supercritical isotherm
#
# At T = 1.5, above the cut-and-shifted fluid's critical temperature (about
# 1.09), the fluid is one phase at every density, so the equation of state
# holds all the way from the dilute gas to the dense fluid. The points are
# the simulations; the line is the fit.

isotherm = [0.1, 0.3, 0.5, 0.7, 0.85]
mds = [simulate(rho, 1.5) for rho in isotherm]
dens = np.linspace(0.02, 0.9, 45)
fig = figure("Pressure at $T = 1.5$", "$\\rho\\sigma^3$", "$P\\sigma^3/\\epsilon$")
fig.plot(
    dens,
    [vp.liquids.eos(temperature=1.5, density=d).p_cs for d in dens],
    "Thol et al. (2015)",
)
fig.scatter(isotherm, [r.p_cs for r in mds], "md_nvt, N = 256")
print("rho    P (MD)             P (EOS)   (MD - EOS) / error")
for rho, r in zip(isotherm, mds):
    p = vp.liquids.eos(temperature=1.5, density=rho).p_cs
    print(
        "%.2f  %7.4f +- %.4f  %7.4f   %+.1f"
        % (rho, r.p_cs, r.p_cs_err, p, (r.p_cs - p) / r.p_cs_err)
    )

# %%
# ## How it's checked
#
# `ports/liquids/test_liquids.py` holds every program to upstream's Fortran
# bit for bit (built with the same random numbers and libm), then checks
# longer runs: md_nvt and md_nve against this equation of state within 4
# standard errors, mc_nvt and mc_npt against the results upstream's GUIDE.md
# tabulates, and the dt^2 law of the energy error (the next example). The
# equation of state itself is upstream's eos_lj, and it matches upstream
# bit for bit at four state points.

r = runs[(0.75, 1.0)]
print("check() of the rho = 0.75 run:", r.check().passed)

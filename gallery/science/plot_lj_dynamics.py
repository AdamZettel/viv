# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Energy conservation and diffusion in a Lennard-Jones liquid
===========================================================

Run constant-energy molecular dynamics of 256 Lennard-Jones atoms with
`vp.liquids.md_nve`, the Viv port of Allen and Tildesley's md_nve_lj, and
measure two things: how the fluctuation of the total energy shrinks with
the time step, which is velocity Verlet's error made visible, and how the
atoms diffuse, from the velocity autocorrelation function and from the
mean-square displacement.
"""

# %%
# ## A liquid to start from
#
# md_nvt melts an fcc lattice at density 0.75 and T = 1, in 4 blocks of 1000
# steps; its final positions and velocities start every run below.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

eq = vp.liquids.md_nvt(
    n=256, density=0.75, temperature=1.0, nblock=4, nstep=1000
)
start = dict(r_start=eq.r_final, v_start=eq.v_final, box_start=eq.box)
last = eq.blocks[-1]
print(
    "last block of the equilibration: T = %.4f, E/N = %.4f"
    % (last[4], last[0])
)

# %%
# ## The energy error goes as dt^2
#
# Velocity Verlet is second order: the total energy is not conserved
# exactly but fluctuates about its starting value by an amount proportional
# to dt^2, without drifting. The mean-square deviation of E/N over 4 time
# units should fall by 2^4 = 16 each time dt is halved. Upstream's GUIDE.md
# reports the same quantity (from longer runs) for comparison.

dts = [0.008, 0.004, 0.002, 0.001]
guide = {
    0.008: 1.8430e-7,
    0.004: 1.3121e-8,
    0.002: 1.2621e-9,
    0.001: 1.8530e-10,
}
msd = []
for dt in dts:
    r = vp.liquids.md_nve(nblock=1, nstep=int(round(4.0 / dt)), dt=dt, **start)
    msd.append(r.conserved_msd)
    print(
        "dt = %.3f   MSD of E/N = %.3e   GUIDE.md (md_nve_lj) %.3e"
        % (dt, r.conserved_msd, guide[dt])
    )
slope = np.polyfit(np.log(dts), np.log(msd), 1)[0]
print("slope of log MSD against log dt: %.2f (velocity Verlet: 4)" % slope)

fig = figure("Energy fluctuation against time step", "$\\Delta t$", "MSD of $E/N$")
fig.plot(dts, msd, "md_nve (Viv)")
fig.plot(dts, [guide[d] for d in dts], "GUIDE.md")
fig.plot(dts, [msd[0] * (d / dts[0]) ** 4 for d in dts], "$\\Delta t^4$")
fig.style("$\\Delta t^4$", "s6 dash nomarker")
fig.xscale("log")
fig.yscale("log")

# %%
# ## Two traces of E/N
#
# The energy per atom along the first 2 time units at two step sizes,
# measured from its mean: bounded oscillation at both, four times smaller
# at half the step.

fig = figure("$E/N$ about its mean", "$t$", "$E/N - \\langle E/N \\rangle$")
for dt in (0.008, 0.004):
    r = vp.liquids.md_nve(nblock=1, nstep=int(round(2.0 / dt)), dt=dt, **start)
    t = dt * np.arange(1, len(r.energy_series) + 1)
    fig.plot(t, r.energy_series - r.energy_series.mean(), "dt = %g" % dt)

# %%
# ## Diffusion two ways
#
# The same run gives the velocity autocorrelation function and the
# mean-square displacement, each averaged over atoms and over time origins
# every 10 samples (a sample every 5 steps). The diffusion coefficient is a
# third of the VACF's integral (Green-Kubo) and a sixth of the MSD's slope
# at long times (Einstein); the two are independent estimates from the
# same trajectory.

r = vp.liquids.md_nve(
    nblock=5,
    nstep=2000,
    dt=0.005,
    sample=5,
    nt=200,
    origin_interval=10,
    **start
)
print("T = %.4f +- %.4f" % (r.t_kin, r.t_kin_err))
print("D from the VACF: %.4f" % r.d_vacf)
print("D from the MSD:  %.4f" % r.d_msd)

fig = figure("Velocity autocorrelation", "$t$", "$\\langle v(0) \\cdot v(t) \\rangle / \\langle v^2 \\rangle$")
fig.plot(r.corr_time, r.vacf / r.vacf[0], "md_nve")
fig.hline("0", 0.0)
fig = figure("Mean-square displacement", "$t$", "$\\langle |r(t) - r(0)|^2 \\rangle$")
fig.plot(r.corr_time, r.msd, "md_nve")
fig.plot(r.corr_time, 6.0 * r.d_msd * r.corr_time, "6 D t")
fig.style("6 D t", "s6 dash nomarker")

# %%
# ## Speed
#
# Wall times on one machine, measured by `ports/liquids/oracle/bench.py`
# and read from `ports/liquids/timings.json`: upstream's Fortran (gfortran
# -O2), upstream's Python (NumPy), and the Viv port through vipyr (its exact
# build), the same runs from the same starting configuration.

timings = json.load(open("ports/liquids/timings.json"))
print("measured %s on %s" % (timings["date"], timings["machine"]))
print("%-42s %9s %9s %9s" % ("run", "Fortran", "Python", "Viv"))
for row in timings["runs"]:
    py = "%8.2fs" % row["python_s"] if "python_s" in row else "        -"
    print(
        "%-42s %8.2fs %s %8.2fs"
        % (row["run"], row["fortran_s"], py, row["viv_s"])
    )

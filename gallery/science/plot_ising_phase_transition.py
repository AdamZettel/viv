# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The Ising model through its phase transition
============================================

Run Monte Carlo on the two-dimensional Ising model at 13 temperatures and
three lattice sizes with `vp.statmech.ising`, and watch the magnetization
vanish, the susceptibility peak and the Binder cumulants cross near Onsager's
critical temperature T_c = 2/ln(1 + sqrt 2) = 2.269. The program is
`stat/ising/ising_mc.viv`; at every run it also reports the exact answers it
can compute for that lattice, which is how the numbers here are checked.
"""

# %%
# ## Spins below, at and above T_c
#
# One 64 x 64 lattice at three temperatures, after 300 Wolff cluster updates
# from an ordered start. `spins` is the final configuration, +1 or -1 per
# site. Below T_c one sign fills the lattice; at T_c there are clusters of
# every size; above it the spins decorrelate within a few sites.

import numpy as np
import vipyr as vp
from vipyr.figure import figure, grid

Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
g = grid("64 x 64 lattice after 300 Wolff updates", 1, 3)
for T in (2.0, round(Tc, 3), 3.0):
    r = vp.statmech.ising(
        size=64,
        temperature=T,
        algorithm="wolff",
        sweeps=300,
        thermalization=100,
        seed=3,
    )
    print("T = %.3f   mean |m| = %.3f +- %.3f" % (T, r.mag_abs, r.mag_abs_err))
    g.figure("$T$ = %.3f" % T).heatmap(r.spins)

# %%
# ## Magnetization, susceptibility and the Binder cumulant
#
# For each size L the scan runs 2000 Wolff updates after 500 for
# thermalization, with errors from 20 bins. The susceptibility here is
# chi = L^2 (<m^2> - <|m|>^2) / T. Its peak grows with L and moves toward T_c.
# The Binder cumulant U = 1 - <m^4> / (3 <m^2>^2) goes from 2/3 in the
# ordered phase to 0 in the disordered one, and its curves for different L
# cross at T_c, because U depends on L only through L / xi.

temps = np.round(np.linspace(1.8, 3.0, 13), 3)
sizes = [8, 16, 32]
scan = {}
for L in sizes:
    scan[L] = [
        vp.statmech.ising(
            size=L,
            temperature=T,
            algorithm="wolff",
            sweeps=2000,
            thermalization=500,
            seed=11,
        )
        for T in temps
    ]

print(" T     " + "  ".join("|m| L=%-3d   U L=%-3d " % (L, L) for L in sizes))
for i, T in enumerate(temps):
    print(
        "%.2f  " % T
        + "  ".join(
            "%.3f      %.3f     " % (scan[L][i].mag_abs, scan[L][i].binder)
            for L in sizes
        )
    )

fig = figure("Magnetization per spin", "$T$", "$\\langle |m| \\rangle$")
for k, L in enumerate(sizes):
    fig.plot(temps, [r.mag_abs for r in scan[L]], "L = %d" % L)
# Yang's spontaneous magnetization, (1 - sinh(2/T)^-4)^(1/8) below T_c and 0 above
Tf = np.linspace(1.8, 3.0, 121)
yang = np.where(Tf < Tc, np.abs(1.0 - np.sinh(2.0 / Tf) ** -4.0) ** 0.125, 0.0)
print(
    "Yang at T = 2.0: %.6f, the program's mag_infinite: %.6f"
    % (yang[20], scan[32][2].mag_infinite)
)
fig.plot(Tf, yang, "Yang, infinite lattice")
fig.style("Yang, infinite lattice", "s6 dash nomarker")
fig.vline("T_c", Tc)

fig = figure("Susceptibility", "$T$", "$\\chi$")
for L in sizes:
    fig.plot(temps, [r.chi for r in scan[L]], "L = %d" % L)
fig.vline("T_c", Tc)

fig = figure("Binder cumulant", "$T$", "U")
for L in sizes:
    fig.plot(temps, [r.binder for r in scan[L]], "L = %d" % L)
fig.vline("T_c", Tc)
fig.hline("2/3", 2.0 / 3.0)

# %%
# ## How it's checked
#
# The program knows the exact energy of the finite periodic lattice from
# Kaufman's partition function, and Yang's magnetization for the infinite
# one (plotted above). The energy per spin of every run in the L = 16 scan,
# against the exact value, in units of its own error bar:

z = [(r.energy - r.exact_energy) / r.energy_err for r in scan[16]]
for T, zi, r in zip(temps, z, scan[16]):
    print(
        "T = %.2f   E = %.5f +- %.5f   exact %.5f   (%+.1f sigma)"
        % (T, r.energy, r.energy_err, r.exact_energy, zi)
    )
print(
    "largest deviation: %.1f sigma over %d runs"
    % (max(abs(x) for x in z), len(z))
)
print(
    "check() of the L = 32 run at T = 2.3:",
    scan[32][list(temps).index(2.3)].check().passed,
)

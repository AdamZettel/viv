# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Oxygen-16 from a Skyrme energy functional
=========================================

Solve the Hartree-Fock equations of oxygen-16 with the Skyrme force SLy5 using
`vp.nuclear.skyrme_hf` (`lib/skyrme/skyrmehf.viv`, a spherical solver on a
radial mesh), and plot the proton and neutron densities, the single-particle
levels, and how the energy converges as the mesh is refined. The binding
energy and charge radius are compared with experiment. The same function
takes other spherical closed-shell nuclei (`nucleus="40Ca"`) and the forces
SLy4, SLy5, SIII and SkM* (`force=`).
"""

# %%
# ## The ground state
#
# The default call is SLy5 (Chabanat et al., Nucl. Phys. A 635, 231 (1998))
# on a mesh of 0.1 fm out to 15 fm. Energies in MeV, radii in fm.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

r = vp.nuclear.skyrme_hf()
print("converged in %d iterations: %s" % (r.iterations, r.converged))
print("E / A = %.4f MeV   E = %.3f MeV" % (r.e_per_a, r.energy))
print(
    "rms radii: protons %.4f fm, neutrons %.4f fm, charge %.4f fm"
    % (r.r_p, r.r_n, r.r_ch)
)
# r.labels names each entry of r.levels: "n 1s1/2", ..., "p 1p1/2"
labels = r.labels.split(",")
levels = [
    [e for l, e in zip(labels, r.levels) if l.startswith(k)]
    for k in ("n ", "p ")
]
names = [l[2:] for l in labels if l.startswith("n ")]
for q, who in enumerate(("neutron", "proton")):
    print(
        "%-8s levels: " % who
        + "   ".join("%s %.3f" % (n, e) for n, e in zip(names, levels[q]))
    )

# %%
# ## Densities
#
# Protons and neutrons fill the same three orbits, so the two densities
# nearly coincide; the protons' Coulomb repulsion pushes them slightly
# outward. The protons' kinetic-energy density tau (dotted) is largest at
# the center, where the 1p orbitals vanish but their gradients do not. Together the
# two densities are about 0.15 fm^-3 at the center, close to nuclear
# saturation density (0.16 fm^-3), and fall to zero over about 2 fm.

mesh = np.asarray(r.mesh)
fig = figure("Oxygen-16 densities, SLy5", "$r$ (fm)", "fm$^{-3}$")
fig.plot(mesh[:70], r.rho[0][:70], "neutrons")
fig.plot(mesh[:70], r.rho[1][:70], "protons")
fig.plot(mesh[:70], r.tau[1][:70], "protons: $\\tau$")
fig.style("neutrons", "s1 nomarker")
fig.style("protons", "s2 dash nomarker")
fig.style("protons: $\\tau$", "s3 dot nomarker")
print(
    "densities at the center: neutrons %.4f, protons %.4f fm^-3"
    % (r.rho[0][0], r.rho[1][0])
)
print(
    "neutrons counted on the mesh: %.6f"
    % (4 * np.pi * np.sum(mesh**2 * r.rho[0]) * (mesh[1] - mesh[0]))
)

# %%
# ## Single-particle levels
#
# The spin-orbit field puts 1p3/2 below 1p1/2 for both kinds of nucleon.
# Each proton level sits 3.3 to 3.8 MeV above the matching neutron level: the
# Coulomb repulsion of the other seven protons.

fig = figure(
    "Single-particle levels, SLy5",
    "",
    "e (MeV)",
    window=(0.0, 3.0, -40.0, 0.0),
)
for q, who in enumerate(("neutrons", "protons")):
    x0 = 0.3 + 1.4 * q
    xs, ys = [], []
    for e in levels[q]:
        xs += [x0, x0 + 1.0, float("nan")]
        ys += [e, e, float("nan")]
    fig.plot(xs, ys, who)
    fig.style(who, "s%d thick nomarker" % (q + 1))
    for n, e in zip(names, levels[q]):
        fig.label("%s %s" % (who, n), n, x0 + 0.3, e + 1.2)
fig.xtick("neutrons", 0.8)
fig.xtick("protons", 2.2)

# %%
# ## Mesh convergence
#
# The same calculation on meshes from 0.2 fm to 0.05 fm. The energy per
# nucleon converges like h^2, so two meshes give an extrapolation to h = 0
# (Richardson, assuming h^2): E(0) = E(h/2) + (E(h/2) - E(h)) / 3. Two
# measurements of the order are printed: between successive halvings
# (0.2, 0.1 and 0.05 fm), and the slope of a fit of the distance from the
# extrapolated limit against h.

spacings = [0.2, 0.15, 0.1, 0.075, 0.05]
conv = [vp.nuclear.skyrme_hf(spacing=h) for h in spacings]
e_h = {h: c.e_per_a for h, c in zip(spacings, conv)}
extrap = e_h[0.05] + (e_h[0.05] - e_h[0.1]) / 3.0
for h, c in zip(spacings, conv):
    print(
        "h = %.3f fm   E/A = %.6f MeV   r_ch = %.4f fm   |E/A - E/A(0)| = "
        "%.1e" % (h, c.e_per_a, c.r_ch, abs(c.e_per_a - extrap))
    )
halving = np.log2((e_h[0.2] - e_h[0.1]) / (e_h[0.1] - e_h[0.05]))
slope = np.polyfit(
    np.log(spacings),
    np.log([abs(c.e_per_a - extrap) for c in conv]),
    1,
)[0]
print("order between halvings 0.2, 0.1, 0.05 fm: %.3f" % halving)
print(
    "fitted order against the limit: %.3f   E/A extrapolated from 0.1 and 0.05 fm: %.4f MeV"
    % (slope, extrap)
)
assert 1.9 < halving < 2.1, "the mesh error is no longer second order: %.3f" % halving
fig = figure(
    "$E/A$ against mesh spacing", "$h$ (fm)", "$|E/A(h) - E/A(0)|$ (MeV)"
)
fig.plot(spacings, [abs(c.e_per_a - extrap) for c in conv], "SLy5")
fig.xscale("log")
fig.yscale("log")

# %%
# ## Against experiment
#
# The binding energy per nucleon of oxygen-16 is 7.976 MeV (AME2020: Wang et
# al., Chin. Phys. C 45, 030003 (2021)) and its charge radius 2.699 fm
# (Angeli and Marinova, At. Data Nucl. Data Tables 99, 69 (2013)). SLy5 was
# fitted to the binding energies and radii of doubly magic nuclei, oxygen-16
# among them; extrapolated to h = 0 it overbinds by 0.6%, and its charge
# radius is about 4% too large.

ba_exp, rch_exp = 7.976, 2.699
print(
    "B/A: SLy5 %.3f MeV (h -> 0), experiment %.3f MeV (%+.1f%%)"
    % (-extrap, ba_exp, 100 * (-extrap / ba_exp - 1))
)
print(
    "r_ch: SLy5 %.3f fm (h = 0.05 fm), experiment %.3f fm (%+.1f%%)"
    % (conv[-1].r_ch, rch_exp, 100 * (conv[-1].r_ch / rch_exp - 1))
)

# %%
# ## How it's checked
#
# `tools/export/test_skyrmehf.py` compares this program with a NumPy twin on
# both of its meshes, and with the skyrme_rpa code of Colò, Cao, Van Giai and
# Capelli (Comput. Phys. Commun. 184, 142 (2013)) for the extrapolated energy
# per nucleon, the charge radius and the orbital energies. `check()` reruns
# the program's own tests and compares the result's bits with the recorded
# ones.

print("check():", r.check().passed)

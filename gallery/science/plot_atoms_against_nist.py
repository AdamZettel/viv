# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Atoms from hydrogen to uranium, against NIST
============================================

Solve the Kohn-Sham equations of 23 atoms in the local density approximation
with `vp.dftatom.atom`, the Viv port of Certik, Pask and Vackar's dftatom, and
compare every orbital energy and total energy with NIST's atomic reference
data (Standard Reference Database 141). Then plot radial densities of the
noble gases, and redo gold and uranium with the Dirac equation to see what
relativity does to their orbitals.
"""

# %%
# ## Every atom NIST lists
#
# NIST tabulates LDA results for H to Ar, Fe, Kr, Xe, Au and U, nonrelativistic
# (LDA) and relativistic (RLDA). The radial mesh and the convergence settings
# are upstream dftatom's for each atom, as recorded in
# `ports/dftatom/oracle/upstream_reference.json`. Energies in hartree.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

nist = json.load(open("ports/dftatom/oracle/nist_reference.json"))["atoms"]
cases = json.load(open("ports/dftatom/oracle/upstream_reference.json"))[
    "cases"
]


def params(kind, z):
    p = dict(cases["%s_%s" % (kind, z)]["params"])
    p["Z"], p["N"] = p.pop("z"), p.pop("n")
    return p


def eig_diffs(r, ref):
    mine = {(o.n, o.l, o.kappa): o.energy for o in r.orbitals}
    return [abs(mine[(n, l, k)] - e) for n, l, k, e in ref["orbitals"]]


lda = {}
print(
    "atom   Z   E_tot (vipyr)          E_tot (NIST)     |dE|     worst |d eps|"
)
for z in nist:
    r = vp.dftatom.atom(**params("lda", z))
    lda[int(z)] = r
    ref = nist[z]["lda"]
    print(
        "%-3s  %3s  %20.8f  %16.6f  %7.1e  %7.1e"
        % (
            nist[z]["symbol"],
            z,
            r.etot,
            ref["etot"],
            abs(r.etot - ref["etot"]),
            max(eig_diffs(r, ref)),
        )
    )

Zs = sorted(lda)
fig = figure("Differences from NIST, LDA", "$Z$", "hartree")
fig.plot(
    Zs,
    [abs(lda[z].etot - nist[str(z)]["lda"]["etot"]) for z in Zs],
    "|E_tot - NIST|",
)
fig.plot(
    Zs,
    [max(eig_diffs(lda[z], nist[str(z)]["lda"])) for z in Zs],
    "worst |orbital energy - NIST|",
)
fig.hline("NIST's last printed digit", 1e-6)
fig.style("|E_tot - NIST|", "s1 noline circle")
fig.style("worst |orbital energy - NIST|", "s2 noline square")
fig.yscale("log")

# %%
# ## Orbital energies across the table
#
# The binding energy -eps of each subshell against Z, on a log scale: the 1s
# level goes from half a hartree in hydrogen to about 3,700 hartree in
# uranium, 13% below the Z^2 / 2 of a bare nucleus: the other electrons
# screen part of the charge.

fig = figure("Kohn-Sham orbital energies, LDA", "$Z$", "$-\\varepsilon$ (hartree)")
names = "spdf"
for n, l in [(1, 0), (2, 1), (3, 0), (3, 2), (4, 1), (5, 1)]:
    xs = [z for z in Zs if any(o.n == n and o.l == l for o in lda[z].orbitals)]
    ys = [
        -next(o.energy for o in lda[z].orbitals if o.n == n and o.l == l)
        for z in xs
    ]
    fig.plot(xs, ys, "%d%s" % (n, names[l]))
    fig.style("%d%s" % (n, names[l]), "thin")
fig.plot(Zs, [z * z / 2.0 for z in Zs], "$Z^2/2$")
fig.style("$Z^2/2$", "s6 dash nomarker thin")
fig.yscale("log")
print(
    "uranium 1s: -eps = %.1f hartree, Z^2 / 2 = %.0f"
    % (-lda[92].orbitals[0].energy, 92**2 / 2)
)

# %%
# ## Radial densities of the noble gases
#
# 4 pi r^2 rho(r) is the number of electrons per unit radius. Its peaks are
# the shells. On a log axis in r the density to plot is r times that,
# 4 pi r^3 rho, the electrons per unit of ln r, so that equal areas under the
# curves are equal numbers of electrons. Helium has one peak, neon two, argon
# three and krypton four, the innermost one a shoulder at the left. Each new shell sits farther out, and the inner ones
# move closer to the nucleus as Z grows.

fig = figure(
    "Radial electron density per unit $\\ln r$, $4\\pi r^3\\rho$",
    "$r$ (bohr)",
    "electrons",
)
for z, name in [(2, "He"), (10, "Ne"), (18, "Ar"), (36, "Kr")]:
    r = lda[z]
    m = (r.mesh > 1e-3) & (r.mesh < 6.0)
    x, rho = r.mesh[m], r.density[m]
    keep = np.unique(
        np.searchsorted(x, np.geomspace(1e-3, 6.0, 300)).clip(0, len(x) - 1)
    )
    fig.plot(x[keep], 4 * np.pi * x[keep] ** 3 * rho[keep], name)
    fig.style(name, "nomarker")
    print(
        "%-2s  electrons within 6 bohr: %.4f of %d"
        % (name, np.trapz(4 * np.pi * x**2 * rho, x), z)
    )
fig.xscale("log")

# %%
# ## Gold and uranium with the Dirac equation
#
# The relativistic runs solve the Dirac equation for each orbital, so a
# level with l > 0 splits in two (j = l - 1/2 and l + 1/2, told apart by
# Dirac's kappa). The bars are each orbital's energy change in percent:
# negative is more strongly bound. Every s and p1/2 orbital binds more
# strongly, the 6s valence orbital most of all (37%). Every d5/2 level, and
# the 4f and 5d shells, bind less, because the contracted s and p shells
# screen the nucleus better.

rel = {z: vp.dftatom.atom(**params("rlda", z)) for z in ("79", "92")}
for z, r in rel.items():
    ref = nist[z]["rlda"]
    print(
        "%s RLDA: E_tot %.6f, NIST %.6f, |dE| %.1e, worst |d eps| %.1e; "
        "relativity lowers E_tot by %.1f hartree"
        % (
            nist[z]["symbol"],
            r.etot,
            ref["etot"],
            abs(r.etot - ref["etot"]),
            max(eig_diffs(r, ref)),
            lda[int(z)].etot - r.etot,
        )
    )

au, au_nr = rel["79"], lda[79]
fig = figure(
    "Gold: the change in each orbital energy from relativity",
    "orbital",
    "$(\\varepsilon_{\\mathrm{RLDA}} - \\varepsilon_{\\mathrm{LDA}})/|\\varepsilon_{\\mathrm{LDA}}|$ (%)",
)
labels, shifts = [], []
for o in au.orbitals:
    nr = next(p.energy for p in au_nr.orbitals if p.n == o.n and p.l == o.l)
    j = o.l + 0.5 if o.kappa < 0 else o.l - 0.5
    labels.append("%d%s%d/2" % (o.n, names[o.l], round(2 * j)))
    shifts.append(100.0 * (o.energy - nr) / abs(nr))
xs = list(range(len(labels)))
fig.bars(xs, shifts, "shift")
for x, lab in zip(xs, labels):
    fig.xtick(lab, x)
for lab, d in zip(labels, shifts):
    print("%-7s %+8.2f %%" % (lab, d))

# %%
# ## How it's checked
#
# `ports/dftatom/test_dftatom.py` holds these runs to NIST at the accuracy
# upstream states (1e-6 hartree in total energies, 2e-6 in eigenvalues,
# NIST printing six decimals), and to upstream's Fortran built with IEEE
# arithmetic and Viv's math library bit for bit.

worst_e = max(abs(lda[z].etot - nist[str(z)]["lda"]["etot"]) for z in Zs)
worst_eig = max(max(eig_diffs(lda[z], nist[str(z)]["lda"])) for z in Zs)
print(
    "over %d LDA atoms: worst |dE_tot| %.1e, worst |d eps| %.1e"
    % (len(Zs), worst_e, worst_eig)
)

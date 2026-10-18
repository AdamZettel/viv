# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Level schemes of neon-20 and magnesium-24 in the sd shell
=========================================================

Diagonalize the USDB interaction for the valence nucleons above oxygen-16 with
`vp.nuclear.shell_model`, a Lanczos solver in the M scheme
(`lib/nuclear/shell/shellmodel.viv`), and draw the low-lying spectra with their
angular momenta. Neon-20 has 4 valence nucleons and a basis of 640 states;
magnesium-24 has 8 and a basis of 28,503. Every energy is compared with
BIGSTICK 8.0.0, an independent shell-model code, run on the same Hamiltonian.
"""

# %%
# ## The two nuclei
#
# `z` and `n` count valence protons and neutrons in the 0d5/2, 1s1/2 and
# 0d3/2 orbits. The result holds the lowest `nstates` eigenvalues, the total
# angular momentum J of each (from <J^2>), and the excitation energies above
# the ground state, in MeV.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

nuclei = {"ne20": ("Ne-20", 2, 2), "mg24": ("Mg-24", 4, 4)}
runs = {}
for key, (name, z, n) in nuclei.items():
    r = vp.nuclear.shell_model(interaction="usdb", z=z, n=n, nstates=10)
    runs[key] = r
    print(
        "%s: M-scheme dimension %d, %d Lanczos iterations, converged %s"
        % (name, r.dim, r.iterations, r.converged)
    )
    for e, j, x in zip(r.energies, r.j, r.excitation):
        print("   J = %d   E = %9.4f MeV   Ex = %6.3f MeV" % (round(j), e, x))

# %%
# ## Level schemes
#
# Each state is a horizontal bar at its excitation energy, labeled with J
# (where two states are closer than the label, one label is shown). The
# lowest 0+, 2+, 4+ and 6+ of neon-20 form its ground-state band. A rigid
# rotor would put them at energies proportional to J(J+1); the second plot
# shows how far the band falls below that, more with each step in J.

fig = figure(
    "Excitation spectra, USDB", "", "$E_x$ (MeV)", window=(0.0, 3.0, -0.5, 9.5)
)
for col, key in enumerate(nuclei):
    r = runs[key]
    x0 = 0.25 + 1.5 * col
    xs, ys = [], []
    for e in r.excitation:
        if e < 9.5:
            xs += [x0, x0 + 1.0, float("nan")]
            ys += [e, e, float("nan")]
    fig.plot(xs, ys, nuclei[key][0])
    fig.style(nuclei[key][0], "s%d thick nomarker" % (col + 1))
    shown = []
    for e, j in sorted(zip(r.excitation, r.j)):
        if e < 9.5 and all(abs(e - s) > 0.35 for s in shown):
            fig.label(
                "J%s%d" % (key, len(shown)), "%d+" % round(j), x0 + 1.02, e
            )
            shown.append(e)
fig.xtick("Ne-20", 0.75)
fig.xtick("Mg-24", 2.25)

rot = runs["ne20"]
e2 = rot.excitation[list(np.round(rot.j)).index(2)]
fig = figure("Neon-20 ground band against a rigid rotor", "$J(J+1)$", "$E_x$ (MeV)")
band = [(0, 0.0)]
for j in (2, 4, 6):
    band.append(
        (j, min(x for x, jj in zip(rot.excitation, rot.j) if round(jj) == j))
    )
fig.plot(
    [j * (j + 1) for j, _ in band], [x for _, x in band], "USDB yrast states"
)
fig.plot(
    [j * (j + 1) for j, _ in band],
    [e2 / 6.0 * j * (j + 1) for j, _ in band],
    "rotor through the 2+",
)
fig.style("rotor through the 2+", "s2 dash nomarker")
for j, x in band:
    print(
        "J = %d   Ex = %.3f MeV   rotor %.3f MeV"
        % (j, x, e2 / 6.0 * j * (j + 1))
    )
print(
    "E(4+) / E(2+) = %.2f (a rigid rotor gives 3.33, a harmonic vibrator 2)"
    % (band[2][1] / band[1][1])
)

# %%
# ## How it's checked
#
# `lib/nuclear/shell/bigstick_reference.json` holds BIGSTICK's ten lowest
# energies for both nuclei, written by `bigstick_compare.py`. The Lanczos
# runs stop when every eigenvalue has changed by less than `tol` = 1e-8.

ref = json.load(open("lib/nuclear/shell/bigstick_reference.json"))
for key, (name, _, _) in nuclei.items():
    r, b = runs[key], ref[key]
    d = np.abs(
        np.asarray(r.energies) - np.asarray(b["energies"][: len(r.energies)])
    )
    print(
        "%s: dimension %d (BIGSTICK %d), worst |E - E_BIGSTICK| = %.1e MeV "
        "over %d states" % (name, r.dim, b["dim"], d.max(), len(d))
    )
print("check():", all(r.check().passed for r in runs.values()))

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The band structure of silicon
=============================

Compute silicon's Kohn-Sham band structure along L - Gamma - X with
`vp.dft.pw`, a plane-wave density-functional code written in Viv (`dft/`:
LDA, GTH pseudopotentials, LOBPCG for the eigenvalues). The run converges the
crystal's density self-consistently on a k-point mesh, then computes the
bands at 61 points of the path. The result shows silicon's indirect gap:
the valence band peaks at Gamma and the conduction band bottoms out most of
the way to X.
"""

# %%
# ## The calculation
#
# The `si` deck sets a 10 hartree plane-wave cutoff, a 4 x 4 x 4 k-mesh,
# the lattice constant 10.26 bohr and 8 bands. The
# band energies come back in eV, measured from the top of the valence band.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

r = vp.dft.pw(material="si")
evb, x = np.asarray(r.evb), np.asarray(r.pathx)
print("SCF iterations: %d, last density change %.1e" % (r.iter, r.drho))
print("bands: %d path points x %d bands" % evb.shape)
iL, iG, iX = 0, len(x) // 2, len(x) - 1
for name, i in (("L", iL), ("Gamma", iG), ("X", iX)):
    print(
        "%-5s " % name
        + " ".join("%7.3f" % (e + 0.0) for e in np.round(evb[i], 3))
    )

# %%
# ## Bands along L - Gamma - X
#
# Four valence bands (8 electrons in the two-atom cell) and the lowest two
# conduction bands. The solver carries 8 bands but converges only those it
# needs; the top two are left out of the plot, as they are of the check
# below. The conduction-band minimum is on the Gamma - X line, the valence
# maximum at Gamma, so the smallest gap is indirect. LDA puts it at about
# 0.5 eV; the measured gap is 1.17 eV, and the underestimate is the known
# failure of LDA for band gaps, not of the solver.

nv = 4
vbm = evb[:, :nv].max()
cb = evb[:, nv]
icb = int(np.argmin(cb))
print(
    "valence maximum %.4f eV at path point %d (Gamma)"
    % (vbm, int(np.argmax(evb[:, nv - 1])))
)
print(
    "conduction minimum %.4f eV at %.0f%% of the way from Gamma to X"
    % (cb[icb], 100 * (icb - iG) / (iX - iG))
)
print(
    "indirect gap %.4f eV, the program's gap %.4f eV; direct gap at Gamma "
    "%.4f eV" % (cb[icb] - vbm, r.gap, cb[iG] - evb[iG, nv - 1])
)

fig = figure("Silicon, LDA band structure", "", "$E - E_{\\mathrm{VBM}}$ (eV)")
for b in range(6):
    lab = "valence" if b < nv else "conduction"
    fig.plot(x, evb[:, b], lab)
fig.style("valence", "s1 nomarker")
fig.style("conduction", "s2 nomarker")
for name, i in (("L", iL), ("Gamma", iG), ("X", iX)):
    fig.xtick(name, x[i])
fig.vline("Gamma", x[iG])
fig.hline("valence maximum", 0.0)
fig.scatter([x[icb]], [cb[icb]], "conduction minimum")
fig.style("conduction minimum", "s3 ring")

# %%
# ## How it's checked
#
# `check()` runs `dft/si_it_check.viv` against the result: every band at
# every point of the path (the lowest six) within 1e-5 eV of an independent dense NumPy
# implementation (`dft/ref`, which builds and diagonalizes the full
# Hamiltonian at each k), and the total energy within 1e-7 hartree.

print("check():", r.check().passed)

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Orbital energy levels of water and its cation
=============================================

Draw the molecular-orbital energy diagram of water from a restricted
Hartree-Fock run, and of the water cation from an unrestricted one, where the
alpha and beta electrons get orbitals of their own. Both calls are
`vp.scf.run`, the Viv port of PySCF's SCF, in the cc-pVDZ basis; every
orbital energy is compared with PySCF 2.5.0 run on the same molecule, and the
cation's spin contamination <S^2> is computed from the returned orbitals.
"""

# %%
# ## Two runs
#
# Water is closed-shell: 10 electrons in 5 doubly occupied orbitals. Removing
# one electron leaves a doublet (`charge=1, spin=1`, spin being the number
# of unpaired electrons), which UHF treats with 5 alpha and 4 beta electrons.
# Energies in hartree; the geometry is the default, PySCF's water example.
# The difference of the two total energies is the ionization energy with the
# orbitals allowed to relax; Koopmans' theorem estimates it from water's
# highest occupied orbital alone, and comes out higher.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

rhf = vp.scf.run(basis="cc-pvdz", method="rhf")
uhf = vp.scf.run(basis="cc-pvdz", method="uhf", charge=1, spin=1)
ref_r = json.load(open("study/ports/pyscf/oracle/water_ccpvdz.json"))
ref_u = json.load(open("study/ports/pyscf/oracle/watercat_ccpvdz_uhf.json"))
print(
    "water   RHF  E = %.10f   PySCF %.10f   %d iterations"
    % (rhf.e_tot, float(ref_r["e_tot"]), rhf.iterations)
)
print(
    "water+  UHF  E = %.10f   PySCF %.10f   %d iterations"
    % (uhf.e_tot, float(ref_u["e_tot"]), uhf.iterations)
)
ip = uhf.e_tot - rhf.e_tot
eps_r = np.asarray(rhf.mo_energy)
print(
    "ionization energy, E(H2O+) - E(H2O): %.4f hartree = %.2f eV"
    % (ip, ip * 27.211386)
)
print(
    "Koopmans' estimate, -eps(HOMO):       %.4f hartree = %.2f eV"
    % (-eps_r[4], -eps_r[4] * 27.211386)
)

# %%
# ## The diagram
#
# Occupied levels are solid, virtual ones thin. The cation's alpha and beta
# levels differ. Exchange acts only between electrons of the same spin, and
# there are five alpha electrons to four beta, so each occupied alpha level
# lies below its beta partner. The lowest empty beta level is the hole the
# ionization left. The oxygen 1s orbital, near -20.5 hartree, is below the
# plot.

ea, eb = np.asarray(uhf.mo_energy[0]), np.asarray(uhf.mo_energy[1])
cols = [("H2O RHF", eps_r, 5), ("H2O+ alpha", ea, 5), ("H2O+ beta", eb, 4)]
fig = figure(
    "Orbital energies in cc-pVDZ",
    "",
    "$\\varepsilon$ (hartree)",
    window=(0.0, 4.6, -2.0, 1.0),
)
for k, (name, eps, nocc) in enumerate(cols):
    x0 = 0.2 + 1.5 * k
    for occ, lab in ((True, name + ": occupied"), (False, name + ": virtual")):
        xs, ys = [], []
        for i, e in enumerate(eps):
            if (i < nocc) == occ and -2.0 < e < 1.0:
                xs += [x0, x0 + 1.1, float("nan")]
                ys += [e, e, float("nan")]
        fig.plot(xs, ys, lab)
        fig.style(lab, "s%d %s nomarker" % (k + 1, "thick" if occ else "thin"))
    fig.xtick(name, x0 + 0.55)
for name, eps, nocc in cols:
    print(
        "%-11s HOMO %8.4f   LUMO %8.4f   gap %.4f hartree"
        % (name, eps[nocc - 1], eps[nocc], eps[nocc] - eps[nocc - 1])
    )

# %%
# ## Spin contamination
#
# A UHF doublet's <S^2> exceeds the
# 0.75 of a pure doublet by the overlap the alpha and beta orbitals fail to
# have, <S^2> = Sz(Sz + 1) + N_beta - sum_ij |<a_i|b_j>|^2. The overlap
# matrix S is not returned, but the orbitals are orthonormal in it, so
# S = (C C^T)^-1 from either square coefficient matrix.

Ca, Cb = np.asarray(uhf.mo_coeff[0]), np.asarray(uhf.mo_coeff[1])
S = np.linalg.inv(Ca @ Ca.T)
ov = Ca[:, :5].T @ S @ Cb[:, :4]
s2 = 0.5 * 1.5 + 4 - np.sum(ov**2)
print(
    "<S^2> = %.6f   PySCF %.6f   (pure doublet 0.75)"
    % (s2, float(ref_u["s2"]))
)

# %%
# ## How it's checked
#
# The PySCF numbers are `study/ports/pyscf/oracle/`, written by PySCF 2.5.0
# itself. The worst differences over every orbital, occupied and virtual:

d_r = np.max(np.abs(eps_r - np.array([float(v) for v in ref_r["mo_energy"]])))
d_a = np.max(
    np.abs(ea - np.array([float(v) for v in ref_u["mo_energy_alpha"]]))
)
d_b = np.max(
    np.abs(eb - np.array([float(v) for v in ref_u["mo_energy_beta"]]))
)
print(
    "RHF: |dE| %.1e, worst orbital %.1e   UHF: |dE| %.1e, worst alpha %.1e, "
    "worst beta %.1e"
    % (
        abs(rhf.e_tot - float(ref_r["e_tot"])),
        d_r,
        abs(uhf.e_tot - float(ref_u["e_tot"])),
        d_a,
        d_b,
    )
)
print("check():", rhf.check().passed and uhf.check().passed)

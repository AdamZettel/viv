# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The usual suspects: crystal + calculation settings, shared by every driver."""
import os
import numpy as np
import pyscf
from pwdft_ref import load_gth, Crystal

GTH = os.path.join(os.path.dirname(pyscf.__file__), 'pbc/gto/pseudo/gth-pade.dat')

def fcc(a):
    return [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]

# name: (lattice constant / bohr, [(GTH entry, fractional position)], ecut / Ha, k-mesh, smearing / Ha, bands)
# Diamond and zincblende put the origin at the bond centre: atoms at +/- 1/8.
MATERIALS = {
    # CUTOFFS ARE MEASURED, not inherited: cutoff_convergence.py runs each material up a
    # ladder of cutoffs, and the one here is the first whose asserted band energies sit
    # within ~0.02 eV of the top of the ladder. The first version of this table used 6 Ha
    # for silicon, germanium and aluminum. It produced plausible band structures, passed
    # every comparison with the numpy reference (which shared the cutoff), and had
    # germanium's band ORDER wrong.
    'Si':   (10.26, [("Si GTH-PADE-q4", [0.125] * 3), ("Si GTH-PADE-q4", [-0.125] * 3)], 10.0, 2, 0.0, 8),
    # Germanium at 6 Ha: the s-like Gamma2' conduction state sits 0.88 eV ABOVE the
    # valence-band top and germanium passes for a semiconductor with its gap at L.
    # Converged (14, 18, 24 Ha agree to 0.01 eV) it sits 0.015 eV BELOW: LDA germanium
    # has no gap, which is the published result.
    'Ge':   (10.68, [("Ge GTH-PADE-q4", [0.125] * 3), ("Ge GTH-PADE-q4", [-0.125] * 3)], 14.0, 2, 0.0, 8),
    'GaAs': (10.68, [("Ga GTH-PADE-q3", [0.125] * 3), ("As GTH-PADE-q5", [-0.125] * 3)], 14.0, 2, 0.0, 8),
    'C':    (6.74,  [("C GTH-PADE-q4", [0.125] * 3), ("C GTH-PADE-q4", [-0.125] * 3)], 30.0, 2, 0.0, 8),
    # Copper's smearing is 0.01, not wider: at 0.02 Ha (0.54 eV) the Fermi-Dirac tail
    # reaches the top of the d bands 1.5 eV down and pushes E_F up by 0.26 eV; from
    # 0.01 to 0.005 it moves 0.04 eV (cu_smearing.py). 50 Ha converges the d
    # bands to ~0.1 eV, 60 Ha to ~0.05 (cu_convergence.py).
    'Cu':   (6.82,  [("Cu GTH-PADE-q11", [0.0] * 3)], 50.0, 4, 0.01, 12),
    'Al':   (7.65,  [("Al GTH-PADE-q3", [0.0] * 3)], 10.0, 4, 0.01, 6),
}

# Too many plane waves for the Viv DENSE engine to be worth waiting for (it builds and
# diagonalizes H at every k and SCF step): these have an iterative-engine check only.
NO_DENSE_ENGINE = {'Ge', 'GaAs', 'C', 'Cu'}
# ... and too many even for numpy to diagonalize densely: the reference is pwdft_iter.py.
ITERATIVE_REFERENCE = {'Cu'}

def build(name):
    a, atoms, ecut, nk, smear, nb = MATERIALS[name]
    names = []
    for entry, _ in atoms:
        if entry not in names:
            names.append(entry)
    pps = [load_gth(GTH, n) for n in names]
    cr = Crystal(fcc(a), [(names.index(e), f) for e, f in atoms], pps)
    return cr, ecut, nk, smear, nb, names

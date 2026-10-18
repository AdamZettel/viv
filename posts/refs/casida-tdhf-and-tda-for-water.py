# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-casida-tdhf-and-tda-for-water.viv.

PySCF 2.5.0, which shares no code with the post: its own integrals (libcint),
SCF, and the TDHF and TDA (CIS) solvers in pyscf.tdscf, which work with
Davidson iterations on matrix-vector products rather than building A and B.
Water in 6-31G at the geometry of lib/hfref/water_631g (O 0 0 0; H 0 -+0.757
0.587 angstrom), RHF with conv_tol 1e-12. All singlet and all triplet
excitations are requested (nstates = n_occ * n_virt = 40), with conv_tol
1e-10 on the eigenvalues.

    python3 posts/refs/casida-tdhf-and-tda-for-water.py > posts/refs/casida-tdhf-and-tda-for-water.out
"""
import numpy as np
import pyscf
from pyscf import gto, scf, tdscf

mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="6-31g", unit="Angstrom",
            verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
nocc = mol.nelectron // 2
nvir = mol.nao - nocc
nov = nocc * nvir
print(f"pyscf {pyscf.__version__}")
print(f"E_RHF = {mf.e_tot:.12f}  nao = {mol.nao}  nocc = {nocc}  nvir = {nvir}")


def run(cls, singlet):
    td = cls(mf)
    td.nstates = nov
    td.singlet = singlet
    td.conv_tol = 1e-10
    td.max_cycle = 500
    td.kernel()
    assert all(td.converged)
    return np.sort(np.asarray(td.e))


for singlet in (True, False):
    label = "singlet" if singlet else "triplet"
    e_rpa = run(tdscf.TDHF, singlet)
    e_tda = run(tdscf.TDA, singlet)
    print(f"{label} TDHF (hartree), all {len(e_rpa)}:")
    print("  " + ", ".join(f"{e:.12f}" for e in e_rpa))
    print(f"{label} TDA (hartree), all {len(e_tda)}:")
    print("  " + ", ".join(f"{e:.12f}" for e in e_tda))
    print(f"{label}: TDA - TDHF, smallest {np.min(e_tda - e_rpa):.3e}, largest {np.max(e_tda - e_rpa):.3e}")

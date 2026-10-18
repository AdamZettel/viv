# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-h2-rhf-uhf-and-full-ci-along-the-bond.viv.

PySCF, which shares no code with the post: its own STO-3G basis, integrals,
SCF and full CI.

  - RHF: scf.RHF, conv_tol 1e-12.
  - UHF: scf.UHF from a broken-symmetry start (the alpha and beta HOMO and
    LUMO mixed at 45 degrees in opposite senses), then restarted from its
    stability analysis until no lower state is found; the lower of that and
    the RHF energy is kept, and <S^2> is spin_square().
  - FCI: fci.FCI on the RHF orbitals.
  - The Coulson-Fischer point, two ways: (a) the R where the RHF triplet
    instability sets in, e_u - e_g = (gg|uu) + (gu|gu) in PySCF's MO integrals,
    found by brentq; (b) bisection on whether the UHF above lies below RHF by
    more than 1e-10 Ha.
  - The hydrogen atom in STO-3G: UHF of a lone H.

    python3 posts/refs/h2-rhf-uhf-and-full-ci-along-the-bond.py > posts/refs/h2-rhf-uhf-and-full-ci-along-the-bond.out

Output recorded 2026-09-25 (PySCF 2.5.0, scipy 1.12.0) is in
posts/refs/h2-rhf-uhf-and-full-ci-along-the-bond.out.
"""
import numpy as np
from pyscf import ao2mo, fci, gto, scf
from scipy.optimize import brentq


def mol_at(R):
    return gto.M(atom=f"H 0 0 0; H 0 0 {R!r}", unit="Bohr", basis="sto-3g", verbose=0)


def rhf(R):
    m = scf.RHF(mol_at(R))
    m.conv_tol = 1e-12
    m.kernel()
    return m


def uhf(R, e_rhf):
    mol = mol_at(R)
    m = scf.UHF(mol)
    m.conv_tol = 1e-12
    r = rhf(R)
    g, u = r.mo_coeff[:, 0], r.mo_coeff[:, 1]
    c = np.cos(np.pi / 4)
    ca = np.array([c * g + c * u, -c * g + c * u]).T
    cb = np.array([c * g - c * u, c * g + c * u]).T
    dm = (np.outer(ca[:, 0], ca[:, 0]), np.outer(cb[:, 0], cb[:, 0]))
    m.kernel(dm0=dm)
    for _ in range(10):
        mo = m.stability()[0]
        dm1 = m.make_rdm1(mo, m.mo_occ)
        e_old = m.e_tot
        m.kernel(dm0=dm1)
        if m.e_tot > e_old - 1e-12:
            break
    if m.e_tot > e_rhf:
        return e_rhf, 0.0
    return m.e_tot, m.spin_square()[0]


print("STO-3G H, as PySCF holds it (exponent, contraction coefficient):")
for e, c in mol_at(1.4)._basis["H"][0][1:]:
    print(f"  {e!r}  {c!r}")

atom = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
ha = scf.UHF(atom)
ha.conv_tol = 1e-12
E_H = ha.kernel()
print(f"E(H atom, STO-3G) = {E_H!r}")

print("R (bohr), E_RHF, E_UHF, <S^2>_UHF, E_FCI:")
R_GRID = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
for R in R_GRID:
    r = rhf(R)
    e_fci = fci.FCI(r).kernel()[0]
    e_u, s2 = uhf(R, r.e_tot)
    print(f"  {R:5.2f}  {r.e_tot!r}  {e_u!r}  {s2:.12f}  {e_fci!r}")

r = rhf(1.4)
e_fci = fci.FCI(r).kernel()[0]
print(f"correlation energy at R = 1.4: E_FCI - E_RHF = {e_fci - r.e_tot!r}")
mol = mol_at(1.4)
S = mol.intor("int1e_ovlp")
h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri = mol.intor("int2e")
print(f"AO integrals at R = 1.4: S12 = {S[0, 1]!r}, h11 = {h[0, 0]!r}, h12 = {h[0, 1]!r}")
print(f"  (11|11) = {eri[0, 0, 0, 0]!r}, (11|22) = {eri[0, 0, 1, 1]!r}, "
      f"(12|12) = {eri[0, 1, 0, 1]!r}, (11|12) = {eri[0, 0, 0, 1]!r}")
# Past about R = 50 PySCF's RHF from its default guess lands on a higher
# state (both electrons on one atom, near -0.16 Ha), so the far check is at 20.
for R in [20.0]:
    e = rhf(R).e_tot
    print(f"E_RHF at R = {R}: {e!r}; minus 2 E(H): {e - 2 * E_H!r}")
print(f"(11|11) at R = 10 (on one atom, independent of R): {mol_at(10.0).intor('int2e')[0, 0, 0, 0]!r}")


def triplet_margin(R):
    """e_u - e_g - (gg|uu) - (gu|gu): negative once RHF is unstable to UHF."""
    m = rhf(R)
    mo = ao2mo.restore(1, ao2mo.kernel(m.mol, m.mo_coeff), 2)
    eg, eu = m.mo_energy
    return eu - eg - mo[0, 0, 1, 1] - mo[0, 1, 0, 1]


R_cf = brentq(triplet_margin, 1.5, 3.5, xtol=1e-12)
print(f"Coulson-Fischer point from the triplet instability: R = {R_cf!r} bohr")

lo, hi = 1.5, 3.5
for _ in range(30):
    mid = 0.5 * (lo + hi)
    e_r = rhf(mid).e_tot
    e_u, _s = uhf(mid, e_r)
    if e_u < e_r - 1e-10:
        hi = mid
    else:
        lo = mid
print(f"Coulson-Fischer point where PySCF's UHF first lies 1e-10 below RHF: between {lo:.6f} and {hi:.6f}")

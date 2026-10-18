# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write lib/hfref/*.viv: PySCF 2.5.0's integrals, densities and energies in the
pack format of lib/hf, to hold lib/hf against.

    python3 tools/hf_reference.py            regenerate every module
    python3 tools/hf_reference.py --audit    literal statistics of the written files
    python3 tools/hf_reference.py he c2s     regenerate only the named modules

Every module is generated; nobody edits one by hand. Running the generator
twice gives the same bytes: PySCF's runs repeat exactly on one machine, and
every float is written with repr(), which round-trips a double. The thread
settings are left at their defaults on purpose. Pinning numpy's BLAS to one
thread (OMP_NUM_THREADS=1) moves water 6-31G's RHF energy from
-75.98394849810573 to -75.98394849810569 (measured), and the lock ladder quotes
the first.

What each system gets, from one PySCF molecule built twice (cart=False and
cart=True), conv_tol 1e-12, minao guess, symmetry off:

- the packs: atoms[natm,4] = (Z, x, y, z) in Bohr; shells[nsh,8] = (atom, l,
  nprim, p0, cart0, sph0, ncart, nsph) with every general contraction split
  into segmented shells (shell, then contraction, then component);
  prims[nprim,2] = (exponent, raw bas_ctr_coeff).
- one-electron integrals in the unit-Cartesian convention (PySCF's Cartesian
  integrals divided by F F^T) and in PySCF's spherical basis, and the
  transform M = diag(F) cart2sph_coeff() between them.
- two-electron integrals packed 8-fold, in full when the basis has at most 19
  functions. Above that: two checksums, the pair diagonal (ij|ij), and 256
  seeded samples.
- energies for each configuration, with the state PySCF lands on from the
  minao guess (DEFAULT) and the lowest state the stability-restart loop
  reaches (STABLE), and converged densities when the basis has at most 19
  functions.
- each stored density re-converged from itself at conv_tol_grad 1e-10 or
  tighter (TIGHT_LADDER), with `_SELF_STEP`: how far one PySCF Roothaan step
  moves it. The generator asserts every one is below 1e-10, so a fixed-point
  check in Viv can hold its own step to the same size. The energy constants
  stay those of the conv_tol 1e-12 runs; the generator asserts that PySCF's
  energy of each stored density is within 1e-12 of them. UHF <S^2> constants
  are computed from the stored densities.

The generator asserts the conventions before it writes anything, so a module
that exists is one whose identities held.

It never writes lib/hfref/boys.viv or lib/hfref/pyscf_tests.viv (other
generators own them) and never touches lib/qcref.viv or
tools/pyscf_reference.py.
"""
import math
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pyscf
from pyscf import ao2mo, gto, lib, scf
from pyscf.scf import addons


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "lib", "hfref")

CONV_TOL = 1e-12
MODULE_LIMIT = 20000   # values per module
FULL_ERI_MAX_NAO = 19  # water 6-31G* Cartesian, 18,145 packed values, is the largest full set
NSAMPLE = 256
SEED_WSUM = 20260923   # weights for the second checksum
SEED_SAMPLE = 777      # which packed entries are sampled
LCG_M = 2147483647     # Park-Miller: s <- 16807 s mod (2^31 - 1); every product is exact in a double
LCG_A = 16807

MODULES = [
    "water_sto3g", "water_631g", "water_631g_stretched", "water_631gs_cart", "water_631gs_sph",
    "hf_631gs", "oh_sto3g", "oh_631g", "oh_631gs", "o2_sto3g", "o2_631g", "o2_631gs",
    "water_ccpvdz", "water_svp", "water_tzvp", "breadth_631gs", "he", "h_sto3g", "h4_diffuse",
    "c2s", "prims",
]

# Geometries in Angstrom, as the PySCF tests and the measurements quote them.
WATER = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
WATER_STRETCHED = "O 0 0 0; H 0 -1.757 1.587; H 0 1.757 1.587"
OH = "O 0 0 0; H 0 0 0.9697"
O2 = "O 0 0 0; O 0 0 1.2075"
HFMOL = "F 0 0 0; H 0 0 0.917"
NH3 = "N 0 0 0.1173; H 0 0.9377 -0.2737; H 0.8121 -0.4689 -0.2737; H -0.8121 -0.4689 -0.2737"
CH4 = "C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276"
N2 = "N 0 0 0; N 0 0 1.0977"
HE = "He 0 0 0"
H1 = "H 0 0 0"
H4 = [("H", 0, 0, i * 0.5) for i in range(4)]
H4_BASIS = ("sto-3g", [[0, [0.002, 1]]])  # pyscf/scf/test/test_addons.py:414

CLOSED = [("RHF", "RHF", 0, 0)]
OPEN1 = [("UHF", "UHF", 0, 1), ("ROHF", "ROHF", 0, 1)]
OPEN2 = [("UHF", "UHF", 0, 2), ("ROHF", "ROHF", 0, 2)]

# module -> list of systems. kinds: "pure" (max l <= 1, one basis), "cart",
# "sph", or "mixed" (both). eri: which bases carry the full packed set.
SPECS = {
    "water_sto3g": [dict(P="WATERSTO3G", atom=WATER, basis="sto-3g", kind="pure",
                         configs=CLOSED + [("UHF", "UHF", 0, 0)])],
    "water_631g": [dict(P="WATER631G", atom=WATER, basis="6-31g", kind="pure",
                        configs=CLOSED + [("UHF", "UHF", 0, 0),
                                          ("CATION_ROHF", "ROHF", 1, 1), ("CATION_UHF", "UHF", 1, 1),
                                          ("SPINM2_UHF", "UHF", 0, -2), ("SPINM2_ROHF", "ROHF", 0, -2),
                                          ("SPIN2_UHF", "UHF", 0, 2), ("SPIN2_ROHF", "ROHF", 0, 2)])],
    "water_631g_stretched": [dict(P="WATER631G_STRETCHED", atom=WATER_STRETCHED, basis="6-31g",
                                  kind="pure", configs=CLOSED, alias_rhf=True)],
    "water_631gs_cart": [dict(P="WATER631GS_CART", atom=WATER, basis="6-31g*", kind="cart", configs=CLOSED)],
    "water_631gs_sph": [dict(P="WATER631GS_SPH", atom=WATER, basis="6-31g*", kind="sph", configs=CLOSED)],
    "hf_631gs": [dict(P="HF_631GS", atom=HFMOL, basis="6-31g*", kind="mixed", configs=CLOSED,
                      full=("cart",))],
    "oh_sto3g": [dict(P="OH_STO3G", atom=OH, basis="sto-3g", kind="pure", configs=OPEN1)],
    "oh_631g": [dict(P="OH_631G", atom=OH, basis="6-31g", kind="pure", configs=OPEN1)],
    "oh_631gs": [dict(P="OH_631GS", atom=OH, basis="6-31g*", kind="mixed", configs=OPEN1, full=("cart",))],
    "o2_sto3g": [dict(P="O2_STO3G", atom=O2, basis="sto-3g", kind="pure", configs=OPEN2)],
    "o2_631g": [dict(P="O2_631G", atom=O2, basis="6-31g", kind="pure", configs=OPEN2)],
    "o2_631gs": [dict(P="O2_631GS", atom=O2, basis="6-31g*", kind="mixed", configs=OPEN2, full=())],
    "water_ccpvdz": [dict(P="WATERCCPVDZ", atom=WATER, basis="cc-pvdz", kind="mixed", configs=CLOSED, full=())],
    "water_svp": [dict(P="WATERSVP", atom=WATER, basis="def2-svp", kind="mixed", configs=CLOSED, full=())],
    "water_tzvp": [dict(P="WATERTZVP", atom=WATER, basis="def2-tzvp", kind="mixed", configs=CLOSED, full=())],
    "breadth_631gs": [dict(P="NH3_631GS", atom=NH3, basis="6-31g*", kind="mixed", configs=CLOSED, full=()),
                      dict(P="CH4_631GS", atom=CH4, basis="6-31g*", kind="mixed", configs=CLOSED, full=()),
                      dict(P="N2_631GS", atom=N2, basis="6-31g*", kind="mixed", configs=CLOSED, full=())],
    "he": [dict(P="HE_STO3G", atom=HE, basis="sto-3g", kind="pure",
                configs=CLOSED + [("UHF", "UHF", 0, 0), ("DICATION_RHF", "RHF", 2, 0)], eps1e=True),
           dict(P="HE_CCPVDZ", atom=HE, basis="cc-pvdz", kind="pure",
                configs=CLOSED + [("UHF", "UHF", 0, 0)], eps1e=True)],
    "h_sto3g": [dict(P="H_STO3G", atom=H1, basis="sto-3g", kind="pure",
                     configs=[("UHF", "UHF", 0, 1), ("ROHF", "ROHF", 0, 1), ("SPINM1_UHF", "UHF", 0, -1)],
                     eps1e=True)],
    "h4_diffuse": [dict(P="H4_DIFFUSE", atom=H4, basis=H4_BASIS, kind="pure",
                        configs=[("RHF", "RHF_LINDEP", 0, 0)])],
}

TITLES = {
    "water_sto3g": "Water in STO-3G, as PySCF has it",
    "water_631g": "Water in 6-31G: the closed shell, the cation and the triplets",
    "water_631g_stretched": "Stretched water in 6-31G, where RHF has two states",
    "water_631gs_cart": "Water in 6-31G*, six Cartesian d functions",
    "water_631gs_sph": "Water in 6-31G*, five spherical d functions",
    "hf_631gs": "Hydrogen fluoride in 6-31G*",
    "oh_sto3g": "The OH radical in STO-3G",
    "oh_631g": "The OH radical in 6-31G",
    "oh_631gs": "The OH radical in 6-31G*",
    "o2_sto3g": "Triplet O2 in STO-3G",
    "o2_631g": "Triplet O2 in 6-31G",
    "o2_631gs": "Triplet O2 in 6-31G*",
    "water_ccpvdz": "Water in cc-pVDZ, the first general contraction",
    "water_svp": "Water in def2-SVP",
    "water_tzvp": "Water in def2-TZVP, the first f functions",
    "breadth_631gs": "NH3, CH4 and N2 in 6-31G*",
    "he": "Helium: STO-3G, the bare dication, and cc-pVDZ",
    "h_sto3g": "One hydrogen atom, one electron",
    "h4_diffuse": "An H4 chain whose basis is nearly linearly dependent",
    "c2s": "Cartesian to spherical, one shell at a time",
    "prims": "Single primitives, up to f, with nothing contracted",
}


# ---------------------------------------------------------------------------
# conventions


def dfact(n):
    """(n)!! with (-1)!! = 1."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def powers(l):
    """PySCF's Cartesian component order: lx from l down, then ly from l - lx down."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def fc2(p):
    """F^2 for one component: PySCF-Cartesian = F x unit-Cartesian."""
    i, j, k = p
    l = i + j + k
    if l <= 1:
        return 1.0
    return 4.0 * math.pi * dfact(2 * i - 1) * dfact(2 * j - 1) * dfact(2 * k - 1) / dfact(2 * l + 1)


def ncart(l):
    return (l + 1) * (l + 2) // 2


class Lcg:
    """Park-Miller. The Viv side reproduces it with mod() on exact integers."""

    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (LCG_A * self.s) % LCG_M
        return self.s / LCG_M


def lcg_weights(seed, n):
    w = np.empty(n)
    s = seed
    for k in range(n):
        s = (LCG_A * s) % LCG_M
        w[k] = s / LCG_M
    return w


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def maxabs(a):
    a = np.asarray(a)
    return float(np.abs(a).max()) if a.size else 0.0


# ---------------------------------------------------------------------------
# packed-8 storage: ij = i(i+1)/2 + j for i >= j, ijkl = ij(ij+1)/2 + kl for ij >= kl


def pack8(G):
    n = G.shape[0]
    i, j = np.tril_indices(n)
    Gp = G[i[:, None], j[:, None], i[None, :], j[None, :]]
    a, b = np.tril_indices(len(i))
    return np.ascontiguousarray(Gp[a, b]), np.ascontiguousarray(np.diag(Gp))


def len8(n):
    p = n * (n + 1) // 2
    return p * (p + 1) // 2


def packed_quartets(n):
    """For every packed index, the four AO indices."""
    i, j = np.tril_indices(n)
    a, b = np.tril_indices(len(i))
    return i[a], j[a], i[b], j[b]


def checksums(g):
    w = lcg_weights(SEED_WSUM, len(g))
    return math.fsum((g * g).tolist()), math.fsum((w * g).tolist())


def samples(g, ao_l, n):
    """256 distinct packed indices: half uniform, half from quartets that hold the
    basis's highest angular momentum, so d and f classes are never left out."""
    ii, jj, kk, ll = packed_quartets(n)
    ao_l = np.asarray(ao_l)
    lq = np.maximum(np.maximum(ao_l[ii], ao_l[jj]), np.maximum(ao_l[kk], ao_l[ll]))
    top = np.nonzero(lq == ao_l.max())[0]
    rng = Lcg(SEED_SAMPLE)
    chosen, seen = [], set()
    while len(chosen) < NSAMPLE // 2:
        k = int(rng.next() * len(g))
        if k not in seen:
            seen.add(k)
            chosen.append(k)
    while len(chosen) < NSAMPLE:
        k = int(top[int(rng.next() * len(top))])
        if k not in seen:
            seen.add(k)
            chosen.append(k)
    idx = np.array(chosen)
    return idx, g[idx]


# ---------------------------------------------------------------------------
# molecules, packs and integrals


def build(atom, basis, charge, spin, cart):
    return gto.M(atom=atom, basis=basis, charge=charge, spin=spin, cart=cart, unit="Angstrom",
                 symmetry=False, verbose=0)


def packs(mol):
    """Split general contractions; order shell, then contraction, then component."""
    shells, prims, fvec, ao_l, labels = [], [], [], [], []
    p0 = cart0 = sph0 = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        cs = mol.bas_ctr_coeff(ib)
        for ic in range(mol.bas_nctr(ib)):
            col = cs[:, ic]
            keep = [p for p in range(len(es)) if col[p] != 0.0]
            for p in keep:
                prims.append((es[p], col[p]))
            nc, ns = ncart(l), 2 * l + 1
            shells.append((mol.bas_atom(ib), l, len(keep), p0, cart0, sph0, nc, ns))
            p0 += len(keep)
            cart0 += nc
            sph0 += ns
            for pw in powers(l):
                fvec.append(math.sqrt(fc2(pw)))
                ao_l.append(l)
                labels.append("x" * pw[0] + "y" * pw[1] + "z" * pw[2])
    atoms = [(mol.atom_charge(a),) + tuple(mol.atom_coord(a)) for a in range(mol.natm)]
    return dict(atoms=np.array(atoms, dtype=float), shells=np.array(shells, dtype=float),
                prims=np.array(prims, dtype=float), F=np.array(fvec), ao_l=ao_l, labels=labels,
                nao_cart=cart0, nao_sph=sph0)


def sph_ao_l(pk):
    out = []
    for sh in pk["shells"]:
        out += [int(sh[1])] * int(sh[7])
    return out


def cart2sph_blocks(pk):
    """M assembled shell by shell from gto.cart2sph(l, 'sp'), to check cart2sph_coeff()."""
    M = np.zeros((pk["nao_cart"], pk["nao_sph"]))
    for sh in pk["shells"]:
        l, c0, s0, nc, ns = int(sh[1]), int(sh[4]), int(sh[5]), int(sh[6]), int(sh[7])
        blk = np.eye(nc) if l <= 1 else gto.cart2sph(l, normalized="sp")
        M[c0:c0 + nc, s0:s0 + ns] = np.asarray(pk["F"][c0:c0 + nc])[:, None] * blk
    return M


def integrals(atom, basis):
    mol = build(atom, basis, 0, None, False)
    molc = build(atom, basis, 0, None, True)
    pk = packs(mol)
    n, m = pk["nao_cart"], pk["nao_sph"]
    check(n == molc.nao and m == mol.nao, "AO count from the split shells")
    # PySCF's own Cartesian labels, in its own order, against powers(l)
    got = [lab[3] for lab in molc.ao_labels(fmt=False)]
    check(got == pk["labels"], "component order differs from cart_labels")
    F = pk["F"]
    Sc = molc.intor("int1e_ovlp_cart")
    check(maxabs(np.diag(Sc) - F * F) < 1e-14, "diag(ovlp_cart) = F^2")
    FF = np.outer(F, F)
    S_u, T_u, V_u = Sc / FF, molc.intor("int1e_kin_cart") / FF, molc.intor("int1e_nuc_cart") / FF
    G_c = molc.intor("int2e_cart")
    G_u = G_c / np.einsum("p,q,r,s->pqrs", F, F, F, F)
    S_s, T_s, V_s = mol.intor("int1e_ovlp_sph"), mol.intor("int1e_kin_sph"), mol.intor("int1e_nuc_sph")
    G_s = mol.intor("int2e_sph")
    M = F[:, None] * mol.cart2sph_coeff()
    check(maxabs(M - cart2sph_blocks(pk)) == 0.0, "cart2sph_coeff() is not the block-diagonal M")
    for X_u, X_s, what in ((S_u, S_s, "S"), (T_u, T_s, "T"), (V_u, V_s, "V")):
        d = maxabs(M.T @ X_u @ M - X_s)
        check(d < 1e-13, "M^T %s_unit M vs %s_sph: %.3g" % (what, what, d))
    G4 = np.tensordot(G_u, M, axes=([3], [0]))
    G4 = np.tensordot(G4, M, axes=([2], [0])).transpose(0, 1, 3, 2)
    G4 = np.einsum("pqkl,qj->pjkl", G4, M)
    G4 = np.einsum("pjkl,pi->ijkl", G4, M)
    d = maxabs(G4 - G_s)
    check(d < 1e-13, "M^4 G_unit vs G_sph: %.3g" % d)
    gc, dc = pack8(G_u)
    gs, ds = pack8(G_s)
    check(maxabs(pack8(G_c)[0] - ao2mo.restore(8, G_c, n)) == 0.0, "packed-8 order differs from ao2mo")
    return dict(mol=mol, molc=molc, pk=pk, M=M, cart=dict(S=S_u, T=T_u, V=V_u, g=gc, gdiag=dc, n=n, G=G_u),
                sph=dict(S=S_s, T=T_s, V=V_s, g=gs, gdiag=ds, n=m, G=G_s), enuc=mol.energy_nuc())


# ---------------------------------------------------------------------------
# SCF states


# The minao, atom, 1e and huckel guesses must reach the same RHF state to
# 1e-10. The one known exception is N2 with huckel, which converges to an
# excited RHF solution about 0.74 Eh up and still reports converged=True; it
# is flagged, and its energy is asserted below.
RHF_GUESS_FLAGGED = {("N2_631GS", "huckel")}
RHF_GUESS_RUNS = []   # the RHF runs whose four guesses were compared
GUESS_SEEN = set()    # the GUESS_STATES entries some run reached

# Open-shell runs have more than one SCF solution, and which one PySCF reaches
# depends on the guess. A guess that lands on neither the default nor the
# stable state lands on one of these third states, labeled
# `_E_<CFG>_GUESS_<G>` in the module. The energies are PySCF 2.5.0's (measured)
# and are asserted to 1e-10, so a landing that moves fails the generator.
# (system, configuration, basis kind, guess) -> energy
GUESS_STATES = {
    ("WATER631G", "SPINM2_UHF", "cart", "huckel"): -75.63554429431865,
    ("WATER631G", "SPINM2_ROHF", "cart", "1e"): -75.64297383696433,
    ("WATER631G", "SPIN2_UHF", "cart", "huckel"): -75.63554429431863,
    ("WATER631G", "SPIN2_ROHF", "cart", "huckel"): -75.63112377325734,
    ("O2_STO3G", "UHF", "cart", "1e"): -147.37855914166403,
    ("O2_STO3G", "UHF", "cart", "huckel"): -147.37855914166408,
    ("N2_631GS", "RHF", "sph", "huckel"): -108.20313259582834,
    ("N2_631GS", "RHF", "cart", "huckel"): -108.2037380712142,
}


def new_mf(mol, method):
    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "ROHF":
        mf = scf.ROHF(mol)
    elif method == "RHF_LINDEP":
        # pyscf/scf/test/test_addons.py:414: canonical orthogonalization at 1e-8
        mf = addons.remove_linear_dep_(scf.RHF(mol), threshold=1e-8, lindep=1e-9)
    else:
        raise ValueError(method)
    mf.conv_tol = CONV_TOL
    mf.verbose = 0
    return mf


def rohf_negative_spin(mol, method):
    """PySCF 2.5.0's ROHF orbital gradient stays at 0.249 for water spin = -2 while the
    energy and density settle to 1e-13 (measured), so its converged flag never
    sets. For these runs convergence is judged on the energy alone."""
    return method == "ROHF" and mol.spin < 0


def run(mol, method, dm0=None, guess="minao"):
    mf = new_mf(mol, method)
    if rohf_negative_spin(mol, method):
        mf.conv_tol_grad = 1.0
    mf.init_guess = guess
    mf.kernel(dm0)
    check(mf.converged, "%s did not converge (guess %s)" % (method, guess))
    return mf


def states(mol, method):
    """The minao state, then restarts from the stability-analysis orbitals until
    the energy stops dropping. Returns (default, stable, how the label was set)."""
    default = run(mol, method)
    if mol.nelectron <= 1:
        return default, default, "one electron: E is the lowest H_core level"
    if mol.nao <= max(mol.nelec):
        return default, default, "no virtual orbitals: nothing to rotate"
    if method == "RHF_LINDEP":
        return default, default, "not analyzed: canonical orthogonalization drops a vector"
    if rohf_negative_spin(mol, method):
        return default, default, "not analyzed: PySCF's ROHF gradient is wrong for spin < 0"
    best = mf = default
    for _ in range(10):
        r = mf.stability()
        mo = r[0] if isinstance(r, tuple) else r
        dm = mf.make_rdm1(mo, mf.mo_occ)
        nxt = run(mol, method, dm0=dm)
        if nxt.e_tot < best.e_tot - 1e-10:
            best = mf = nxt
        else:
            break
    return default, best, "stability restart"


# Settings for re-converging a state before its density is stored, tried in
# this order from the state's own density. conv_tol 1e-12 alone stops at
# PySCF's default gradient test (about 1e-6), where one Roothaan step still
# moves the density by up to 2e-7 (stretched water, measured), too far for a
# fixed-point check at 1e-8. DIIS keeps the stretched default state and the O2
# ROHF states on course; the stretched stable state only reaches this gradient
# with DIIS off (measured).
TIGHT_LADDER = [
    dict(conv_tol=1e-14, conv_tol_grad=1e-11, diis=True),
    dict(conv_tol=1e-13, conv_tol_grad=1e-10, diis=True),
    dict(conv_tol=1e-15, conv_tol_grad=1e-11, diis=False),
]
SELF_STEP_MAX = 1e-10  # every stored density is this close to PySCF's fixed point; the Viv tests write 1.0e-10
POLISH_STEPS = 200


def self_step(mf, dm):
    """One PySCF Roothaan step from dm with no DIIS, damping or level shift:
    get_fock, eig, get_occ, make_rdm1. Returns the new density."""
    f = mf.get_fock(dm=dm)
    e, c = mf.eig(f, mf.get_ovlp())
    return np.asarray(mf.make_rdm1(c, mf.get_occ(e, c)))


def tighten(mol, method, mf):
    """Re-converge mf's state from its own density under each TIGHT_LADDER setting
    and keep the run whose density PySCF's own step moves least. A run counts only
    if PySCF flags it converged and it stays on mf's state (energy within 1e-10).
    Returns (density, how far PySCF's step moves it, the setting used)."""
    dm0 = np.asarray(mf.make_rdm1())
    best = None
    for k, t in enumerate(TIGHT_LADDER):
        m = new_mf(mol, method)
        m.conv_tol = t["conv_tol"]
        m.conv_tol_grad = t["conv_tol_grad"]
        m.max_cycle = 500
        if not t["diis"]:
            m.diis = None
        if rohf_negative_spin(mol, method):
            m.conv_tol_grad = 1.0
        m.kernel(dm0)
        if not m.converged or abs(m.e_tot - mf.e_tot) > 1e-10:
            continue
        dm = np.asarray(m.make_rdm1())
        if rohf_negative_spin(mol, method):
            # The gradient test is off here, so the run stops on the energy alone,
            # with the density still 4e-9 from the step's fixed point (measured).
            # Plain PySCF steps converge to it.
            for _ in range(POLISH_STEPS):
                nxt = self_step(m, dm)
                if maxabs(nxt - dm) < 1e-13:
                    break
                dm = nxt
        gap = maxabs(self_step(m, dm) - dm)
        if best is None or gap < best[1]:
            best = (dm, gap, k)
    check(best is not None, "no tight setting reconverged %s on its state" % method)
    check(maxabs(best[0] - dm0) < 1e-6, "the tight %s density left its state" % method)
    return best


def tight_e(mf, dm):
    """PySCF's energy of a density, to compare the re-converged state with the stored energy."""
    return float(mf.energy_tot(dm=dm))


def s2_of(dm, S):
    """<S^2> of a UHF determinant from its spin densities:
    ((Na - Nb)/2)^2 + (Na + Nb)/2 - Tr(Da S Db S)."""
    da, db = dm[0], dm[1]
    na, nb = float(np.sum(da * S)), float(np.sum(db * S))
    return ((na - nb) / 2) ** 2 + (na + nb) / 2 - float(np.trace(da @ S @ db @ S))


def eps1e(mol):
    """Lowest eigenvalue of H_core C = S C e."""
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    Li = np.linalg.inv(np.linalg.cholesky(mol.intor("int1e_ovlp")))
    return float(np.linalg.eigvalsh(Li @ h @ Li.T)[0])


def dens_of(d, method):
    d = np.asarray(d)
    if method in ("RHF", "RHF_LINDEP"):
        return [d]
    return [d[0], d[1]]


def e_from(h, G, ds, enuc):
    """E = sum D h + 1/2 sum (ij|kl) [Dt_ij Dt_kl - Da_ik Da_jl - Db_ik Db_jl] + E_nuc."""
    if len(ds) == 1:
        da = db = ds[0] / 2
    else:
        da, db = ds
    dt = da + db
    J = np.einsum("ijkl,kl->ij", G, dt)
    Ka = np.einsum("ijkl,jl->ik", G, da)
    Kb = np.einsum("ijkl,jl->ik", G, db)
    return float(np.sum(dt * h) + 0.5 * (np.sum(dt * J) - np.sum(da * Ka) - np.sum(db * Kb)) + enuc)


def config_block(ints, basis_kind, P, key, method, charge, spin, atom, basis, report):
    """Run one configuration in one basis kind ("cart" or "sph")."""
    cart = basis_kind == "cart"
    mol = build(atom, basis, charge, spin, cart)
    default, stable, how = states(mol, method)
    stored = mol.nao <= FULL_ERI_MAX_NAO
    out = dict(key=key, method=method, nalpha=mol.nelec[0], nbeta=mol.nelec[1], spin=spin,
               e_default=float(default.e_tot), e_stable=float(stable.e_tot), how=how,
               is_stable=(-1 if how.startswith("not analyzed") else 1 if stable is default else 0))
    check(out["e_stable"] <= out["e_default"] + 1e-12, "stable state above default")
    if method == "UHF":
        out["s2_default"] = float(default.spin_square()[0])
        out["s2_stable"] = float(stable.spin_square()[0])
    if mol.nelectron == 1:
        eps = eps1e(mol)
        check(abs(eps + mol.energy_nuc() - out["e_default"]) < 1e-10, "one electron: E is the lowest Hcore level")
    # RHF: every guess reaches the minao state to 1e-10 (N2 huckel flagged).
    # UHF/ROHF: every guess reaches the default, the stable or a labeled third state.
    if method != "RHF_LINDEP" and mol.nelectron > 1:
        for guess in ("atom", "1e", "huckel"):
            eg = float(run(mol, method, guess=guess).e_tot)
            if abs(eg - out["e_default"]) < 1e-10:
                continue
            gk = (P, key, basis_kind, guess)
            if method == "RHF":
                check((P, guess) in RHF_GUESS_FLAGGED,
                      "%s %s %s: the %s guess gives %r, the minao guess %r"
                      % (P, key, basis_kind, guess, eg, out["e_default"]))
            elif abs(eg - out["e_stable"]) < 1e-10:
                report.append("  %s %s %s: the %s guess lands on the stable state" % (P, key, basis_kind, guess))
                continue
            check(gk in GUESS_STATES and abs(eg - GUESS_STATES[gk]) < 1e-10,
                  "%s %s %s: the %s guess lands at %r, on no labeled state" % (P, key, basis_kind, guess, eg))
            out.setdefault("guess_states", []).append((guess, eg))
            if method == "RHF":
                report.append("  %s %s %s: huckel guess flagged, lands at %r (an excited RHF solution)"
                              % (P, key, basis_kind, eg))
            else:
                report.append("  %s %s %s: the %s guess lands on the labeled third state %s_E_%s_GUESS_%s = %r"
                              % (P, key, basis_kind, guess, P, key, guess.upper(), eg))
        GUESS_SEEN.update((P, key, basis_kind, g) for g, _ in out.get("guess_states", []))
        if method == "RHF":
            RHF_GUESS_RUNS.append((P, key, basis_kind))
    # densities in the module's convention: unit-Cartesian or spherical
    b = ints[basis_kind]
    F = ints["pk"]["F"]
    conv = (lambda d: d * np.outer(F, F)) if cart else (lambda d: d)
    h = b["T"] + b["V"]
    for tag, mf in (("default", default), ("stable", stable)):
        dm = np.asarray(mf.make_rdm1())
        if stored:
            if tag == "stable" and stable is default:
                dm, gap = out["dm_default"], out["step_default"]
            else:
                dm, gap, rung = tighten(mol, method, mf)
                # the step measured in the module's convention, which is what a
                # Viv fixed-point test compares against
                gap = max(maxabs(conv(a) - conv(b)) for a, b in
                          zip(dens_of(self_step(mf, dm), method), dens_of(dm, method)))
                check(gap < SELF_STEP_MAX, "%s %s %s %s: PySCF's step moves the stored density by %.3g"
                      % (P, key, basis_kind, tag, gap))
                report.append("  %s %s %s %s: tight setting %d, self-step %.3g"
                              % (P, key, basis_kind, tag, rung, gap))
            out["dm_" + tag], out["step_" + tag] = dm, gap
            if method == "UHF":
                # <S^2> of the density that is stored, not of the looser run
                S = mol.intor("int1e_ovlp")
                check(abs(s2_of(np.asarray(mf.make_rdm1()), S) - mf.spin_square()[0]) < 1e-12,
                      "<S^2> from the spin densities vs spin_square()")
                out["s2_" + tag] = s2_of(dm, S)
            check(abs(tight_e(mf, dm) - mf.e_tot) < 1e-12, "%s %s %s %s: the re-converged state's energy moved"
                  % (P, key, basis_kind, tag))
        ds = [conv(d) for d in dens_of(dm, method)]
        out["d_" + tag] = ds
        e = e_from(h, b["G"], ds, ints["enuc"])
        check(abs(e - mf.e_tot) < 1e-10, "%s %s %s: E from the emitted density is off by %.3g"
              % (P, key, tag, e - mf.e_tot))
        ne = sum(float(np.sum(d * b["S"])) for d in ds)
        check(abs(ne - mol.nelectron) < 1e-10, "Tr(D S) != N")
    return out


# ---------------------------------------------------------------------------
# Viv emission


def num(x):
    return repr(float(x))


def legacy_names():
    """Top-level names in the legacy modules, so no generated name collides."""
    names = set()
    for f in ("qcref", "qcfuzz", "gto", "gto_md", "linalg"):
        path = os.path.join(ROOT, "lib", f + ".viv")
        if os.path.exists(path):
            for line in open(path):
                m = re.match(r"^(?:def\s+([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*=)", line)
                if m:
                    names.add(m.group(1) or m.group(2))
    return names


class Module:
    def __init__(self, name, mp):
        self.name = name
        self.mp = mp  # lowercase helper prefix
        self.lines = []
        self.nvalues = 0
        self.maxlit = 0
        self.names = set()
        self.tests = []
        self.seed = "%s_SEED_WSUM" % mp.upper()

    def claim(self, nm):
        check(nm not in self.names, "%s: %s defined twice" % (self.name, nm))
        self.names.add(nm)

    def out(self, *ls):
        self.lines.extend(ls)

    def const(self, name, value, doc=None, integer=False):
        self.claim(name)
        if doc:
            self.out("# " + doc)
        self.out("%s = %s" % (name, str(int(value)) if integer else num(value)))
        self.nvalues += 1

    def vector(self, name, values, doc=None, integer=False):
        """A flat vector of values as one def returning one literal. The compiler
        emits a long numeric literal as a C table, so splitting it no longer
        pays: on water_631gs_cart (18,145 packed ERIs) cc -O2 took 0.29 s with
        512-value chunks and 0.18 s with whole literals (best of 3, measured)."""
        vals = np.asarray(values, dtype=float).ravel()
        check(len(vals) > 0, "empty literal " + name)
        fmt = (lambda v: str(int(v))) if integer else num
        if doc:
            self.out("#| " + doc)
        body = [fmt(v) for v in vals]
        rows = [", ".join(body[k:k + 4]) for k in range(0, len(body), 4)]
        self.nvalues += len(vals)
        self.maxlit = max(self.maxlit, len(vals))
        self.claim(name)
        self.out("def %s() -> Arr {" % name, "    return [" + (",\n        ".join(rows)) + "]", "}", "")

    def matrix(self, name, A, doc=None, integer=False):
        A = np.asarray(A, dtype=float)
        self.vector(name + "_flat", A, doc=doc, integer=integer)
        self.claim(name)
        self.out("def %s() -> Arr {" % name,
                 "    return %s_mat(%s_flat(), %d, %d)" % (self.mp, name, A.shape[0], A.shape[1]), "}", "")

    def alias(self, name, target, doc=None):
        self.claim(name)
        if doc:
            self.out("#| " + doc)
        self.out("def %s() -> Arr {" % name, "    return %s()" % target, "}", "")

    def test(self, title, body):
        self.tests.append((title, body))

    def text(self):
        t = list(self.lines)
        for title, body in self.tests:
            t.append('test "%s" {' % title)
            t.extend("    " + b for b in body)
            t.extend(["}", ""])
        return "\n".join(t).rstrip("\n") + "\n"


def helpers(M):
    """Small defs every module's tests use. Each module carries its own copy under its
    own prefix, so two modules can be used together without a duplicate def."""
    p = M.mp
    for nm in ("mat", "eye", "asym", "diag_dev", "trace2", "e_elec", "sumsq", "wsum", "schwarz",
               "sample_schwarz", "mtxm_dev", "pack_bad", "pair_table"):
        M.claim("%s_%s" % (p, nm))
    M.out(
        "# ---- helpers for the tests below ----",
        "",
        "#| Reshape a flat row-major vector into r x c.",
        "def %s_mat(v: Arr, r, c) -> Arr {" % p,
        "    a = zeros(r, c)",
        "    for i to r {",
        "        for j to c {",
        "            a[i, j] = v[i * c + j]",
        "        }",
        "    }",
        "    return a",
        "}",
        "",
        "def %s_eye(n) -> Arr {" % p,
        "    a = zeros(n, n)",
        "    for i to n {",
        "        a[i, i] = 1.0",
        "    }",
        "    return a",
        "}",
        "",
        "#| Largest |A_ij - A_ji| / (1 + |A_ij|): a kinetic entry near 100 carries",
        "#| rounding of about 1e-14, so the gap is measured relative to the entry.",
        "def %s_asym(a: Arr, n) {" % p,
        "    w = 0.0",
        "    for i to n {",
        "        for j to n {",
        "            w = max(w, abs(a[i, j] - a[j, i]) / (1.0 + abs(a[i, j])))",
        "        }",
        "    }",
        "    return w",
        "}",
        "",
        "#| Largest |A_ii - 1|.",
        "def %s_diag_dev(a: Arr, n) {" % p,
        "    w = 0.0",
        "    for i to n {",
        "        w = max(w, abs(a[i, i] - 1.0))",
        "    }",
        "    return w",
        "}",
        "",
        "#| sum_ij A_ij B_ij: Tr(D S) is the electron count.",
        "def %s_trace2(a: Arr, b: Arr, n) {" % p,
        "    t = 0.0",
        "    for i to n {",
        "        for j to n {",
        "            t = t + a[i, j] * b[i, j]",
        "        }",
        "    }",
        "    return t",
        "}",
        "",
        "#| pr[i, j] = the packed pair index of (i, j), in either order.",
        "def %s_pair_table(n) -> Arr {" % p,
        "    pr = zeros(n, n)",
        "    for i to n {",
        "        for j to i + 1 {",
        "            pr[i, j] = i * (i + 1) / 2 + j",
        "            pr[j, i] = pr[i, j]",
        "        }",
        "    }",
        "    return pr",
        "}",
        "",
        "#| One- plus two-electron energy from spin densities and packed (ij|kl):",
        "#| sum Dt h + 1/2 sum (ij|kl) [Dt_ij Dt_kl - Da_ik Da_jl - Db_ik Db_jl].",
        "#| For RHF pass Da = Db = D/2. Written over all n^4 quartets because it is",
        "#| a check, not a J/K build.",
        "def %s_e_elec(h: Arr, g: Arr, da: Arr, db: Arr, n) {" % p,
        "    pr = %s_pair_table(n)" % p,
        "    e = 0.0",
        "    for i to n {",
        "        for j to n {",
        "            e = e + (da[i, j] + db[i, j]) * h[i, j]",
        "        }",
        "    }",
        "    e2 = 0.0",
        "    for i to n {",
        "        for j to n {",
        "            pij = pr[i, j]",
        "            dij = da[i, j] + db[i, j]",
        "            for k to n {",
        "                for l to n {",
        "                    pkl = pr[k, l]",
        "                    a = max(pij, pkl)",
        "                    b = min(pij, pkl)",
        "                    x = g[a * (a + 1) / 2 + b]",
        "                    e2 = e2 + x * (dij * (da[k, l] + db[k, l]) - da[i, k] * da[j, l] - db[i, k] * db[j, l])",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return e + 0.5 * e2",
        "}",
        "",
        "def %s_sumsq(g: Arr) {" % p,
        "    t = 0.0",
        "    for k to len(g) {",
        "        t = t + g[k] * g[k]",
        "    }",
        "    return t",
        "}",
        "",
        "#| sum_k w_k g_k with w_k = s_k / (2^31 - 1), s_k = 16807 s_(k-1) mod (2^31 - 1),",
        "#| s_0 = seed. Every product stays below 2^53, so the weights are exact.",
        "def %s_wsum(g: Arr, seed) {" % p,
        "    s = seed",
        "    t = 0.0",
        "    for k to len(g) {",
        "        s = mod(16807.0 * s, 2147483647.0)",
        "        t = t + s / 2147483647.0 * g[k]",
        "    }",
        "    return t",
        "}",
        "",
        "#| Largest (ij|kl)^2 - (ij|ij)(kl|kl) over the packed set, relative to the",
        "#| bound. Cauchy-Schwarz says it is at most 0.",
        "def %s_schwarz(g: Arr, n) {" % p,
        "    npair = n * (n + 1) / 2",
        "    w = -1.0",
        "    for a to npair {",
        "        for b to a + 1 {",
        "            x = g[a * (a + 1) / 2 + b]",
        "            q = g[a * (a + 1) / 2 + a] * g[b * (b + 1) / 2 + b]",
        "            w = max(w, (x * x - q) / (q + 1.0e-300))",
        "        }",
        "    }",
        "    return w",
        "}",
        "",
        "#| The same bound on sampled entries, with the pair diagonal given.",
        "def %s_sample_schwarz(idx: Arr, val: Arr, gd: Arr) {" % p,
        "    w = -1.0",
        "    for k to len(idx) {",
        "        a = floor((sqrt(8.0 * idx[k] + 1.0) - 1.0) / 2.0)",
        "        if a * (a + 1) / 2 > idx[k] {",
        "            a = a - 1",
        "        }",
        "        if (a + 1) * (a + 2) / 2 <= idx[k] {",
        "            a = a + 1",
        "        }",
        "        b = idx[k] - a * (a + 1) / 2",
        "        q = gd[a] * gd[b]",
        "        w = max(w, (val[k] * val[k] - q) / (q + 1.0e-300))",
        "    }",
        "    return w",
        "}",
        "",
        "#| Largest |M^T X M - Y|.",
        "def %s_mtxm_dev(m: Arr, x: Arr, y: Arr, nc, ns) {" % p,
        "    w = 0.0",
        "    for a to ns {",
        "        for b to ns {",
        "            t = 0.0",
        "            for i to nc {",
        "                for j to nc {",
        "                    t = t + m[i, a] * x[i, j] * m[j, b]",
        "                }",
        "            }",
        "            w = max(w, abs(t - y[a, b]))",
        "        }",
        "    }",
        "    return w",
        "}",
        "",
        "#| Count what is wrong with a shells pack: offsets that do not run on, a",
        "#| component count that does not match l, a primitive range out of bounds.",
        "def %s_pack_bad(sh: Arr, nsh, nprim, natm) {" % p,
        "    bad = 0",
        "    p0 = 0",
        "    c0 = 0",
        "    s0 = 0",
        "    for k to nsh {",
        "        l = sh[k, 1]",
        "        if sh[k, 0] < 0 or sh[k, 0] >= natm {",
        "            bad = bad + 1",
        "        }",
        "        if sh[k, 3] != p0 or sh[k, 4] != c0 or sh[k, 5] != s0 {",
        "            bad = bad + 1",
        "        }",
        "        if sh[k, 6] != (l + 1) * (l + 2) / 2 or sh[k, 7] != 2 * l + 1 {",
        "            bad = bad + 1",
        "        }",
        "        p0 = p0 + sh[k, 2]",
        "        c0 = c0 + sh[k, 6]",
        "        s0 = s0 + sh[k, 7]",
        "    }",
        "    if p0 != nprim {",
        "        bad = bad + 1",
        "    }",
        "    return bad",
        "}",
        "",
    )


def m4_helpers(M):
    """The defs that turn a full unit-Cartesian packed set into the spherical one.
    Emitted only in modules that need them (oh_631gs, hf_631gs)."""
    p = M.mp
    for nm in ("quarter", "eri8_m4", "sample_dev"):
        M.claim("%s_%s" % (p, nm))
    M.out(
        "#| One quarter transform with the indices rotated: x has shape (n1, n2, n3, nc),",
        "#| and y[d, i, j, k] = sum_l x[i, j, k, l] M[l, d] has shape (ns, n1, n2, n3).",
        "#| Four of these take (ij|kl) to (ab|cd) in the original index order.",
        "def %s_quarter(x: Arr, m: Arr, n1, n2, n3, nc, ns) -> Arr {" % p,
        "    y = zeros(ns * n1 * n2 * n3)",
        "    for i to n1 {",
        "        for j to n2 {",
        "            for k to n3 {",
        "                base = ((i * n2 + j) * n3 + k) * nc",
        "                for d to ns {",
        "                    t = 0.0",
        "                    for l to nc {",
        "                        t = t + x[base + l] * m[l, d]",
        "                    }",
        "                    y[((d * n1 + i) * n2 + j) * n3 + k] = t",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return y",
        "}",
        "",
        "#| Packed-8 spherical ERIs from packed-8 unit-Cartesian ones:",
        "#| (ab|cd)_sph = sum_ijkl M_ia M_jb M_kc M_ld (ij|kl)_unit.",
        "def %s_eri8_m4(g: Arr, m: Arr, nc, ns) -> Arr {" % p,
        "    pr = %s_pair_table(nc)" % p,
        "    x = zeros(nc * nc * nc * nc)",
        "    for i to nc {",
        "        for j to nc {",
        "            pij = pr[i, j]",
        "            for k to nc {",
        "                for l to nc {",
        "                    pkl = pr[k, l]",
        "                    a = max(pij, pkl)",
        "                    b = min(pij, pkl)",
        "                    x[((i * nc + j) * nc + k) * nc + l] = g[a * (a + 1) / 2 + b]",
        "                }",
        "            }",
        "        }",
        "    }",
        "    x = %s_quarter(x, m, nc, nc, nc, nc, ns)" % p,
        "    x = %s_quarter(x, m, ns, nc, nc, nc, ns)" % p,
        "    x = %s_quarter(x, m, ns, ns, nc, nc, ns)" % p,
        "    x = %s_quarter(x, m, ns, ns, ns, nc, ns)" % p,
        "    npair = ns * (ns + 1) / 2",
        "    out = zeros(npair * (npair + 1) / 2)",
        "    for a to ns {",
        "        for b to a + 1 {",
        "            ab = a * (a + 1) / 2 + b",
        "            for c to ns {",
        "                for d to c + 1 {",
        "                    cd = c * (c + 1) / 2 + d",
        "                    if cd <= ab {",
        "                        out[ab * (ab + 1) / 2 + cd] = x[((a * ns + b) * ns + c) * ns + d]",
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
        "    return out",
        "}",
        "",
        "#| Largest |g[idx_k] - val_k| over the samples.",
        "def %s_sample_dev(g: Arr, idx: Arr, val: Arr) {" % p,
        "    w = 0.0",
        "    for k to len(idx) {",
        "        w = max(w, abs(g[idx[k]] - val[k]))",
        "    }",
        "    return w",
        "}",
        "",
    )


def emit_basis(M, P, p, k, b, full, ao_l):
    """One basis kind: one-electron matrices and the ERIs, full or sampled.
    ERI names always say which basis they are in: _eri8_cart or _eri8_sph."""
    n = b["n"]
    unit = k == "cart"
    tag = "unit" if unit else "sph"
    word = "unit-Cartesian" if unit else "spherical"
    M.matrix("%s_s_%s" % (p, tag), b["S"], doc="Overlap, %s, [%d,%d]." % (word, n, n))
    M.matrix("%s_t_%s" % (p, tag), b["T"], doc="Kinetic energy, %s." % word)
    M.matrix("%s_v_%s" % (p, tag), b["V"], doc="Nuclear attraction, %s." % word)
    ss, sq = checksums(b["g"])
    PK = "%s_ERI8_%s" % (P, k.upper())
    pk = "%s_eri8_%s" % (p, k)
    M.const(PK + "_LEN", len(b["g"]), integer=True, doc="Packed-8 length n(n+1)/2 (n(n+1)/2 + 1)/2 for n = %d." % n)
    M.const(PK + "_SUMSQ", ss, doc="sum of Gp^2 (math.fsum).")
    M.const(PK + "_WSUM", sq, doc="sum of w_k Gp_k with the LCG weights from %s (math.fsum)." % M.seed + "")
    if full:
        M.vector(pk, b["g"], doc="Every packed-8 two-electron integral, %s, chemist's notation." % word)
    else:
        idx, val = samples(b["g"], ao_l, n)
        M.vector(pk + "_idx", idx, integer=True,
                 doc="256 packed indices: 128 uniform, then 128 from quartets holding the top l.")
        M.vector(pk + "_val", val, doc="The packed values at those indices.")
        M.vector(pk + "_diag", b["gdiag"], doc="(ij|ij) for every pair ij, in pair order: the Schwarz bounds.")
    return tag


def emit_system(M, S, report):
    P, p = S["P"], S["P"].lower()
    kind = S["kind"]
    ints = integrals(S["atom"], S["basis"])
    pk = ints["pk"]
    nc, ns = pk["nao_cart"], pk["nao_sph"]
    if kind == "pure":
        check(nc == ns and maxabs(ints["M"] - np.eye(nc)) == 0.0, "pure basis must have M = I")
    kinds = {"pure": ["cart"], "cart": ["cart"], "sph": ["sph"], "mixed": ["sph", "cart"]}[kind]
    full = S.get("full")
    fullset = set(kinds) if full is None else set(full)
    for k in kinds:
        if ints[k]["n"] > FULL_ERI_MAX_NAO:
            fullset.discard(k)
    sfx = {"pure": {"cart": ""}, "cart": {"cart": ""}, "sph": {"sph": ""},
           "mixed": {"sph": "", "cart": "_cart"}}[kind]
    M.out("# ======== %s: %s, %s ========" % (P, S["basis"] if isinstance(S["basis"], str) else "sto-3g + s(0.002)",
                                              {"pure": "s and p only, so Cartesian = spherical",
                                               "cart": "Cartesian (6D)", "sph": "spherical (5D)",
                                               "mixed": "spherical, and Cartesian with the _cart suffix"}[kind]), "")
    M.const(P + "_NATOM", len(pk["atoms"]), integer=True)
    M.const(P + "_NSH", len(pk["shells"]), integer=True)
    M.const(P + "_NPRIMTOT", len(pk["prims"]), integer=True)
    M.const(P + "_NAO_CART", nc, integer=True)
    M.const(P + "_NAO_SPH", ns, integer=True)
    M.const(P + "_E_NUC", ints["enuc"])
    M.out("")
    M.matrix(p + "_atoms", pk["atoms"], doc="(Z, x, y, z) in Bohr, one atom per row.")
    M.matrix(p + "_shells", pk["shells"], integer=True,
             doc="(atom, l, nprim, p0, cart0, sph0, ncart, nsph); general contractions split.")
    M.matrix(p + "_prims", pk["prims"], doc="(exponent, raw bas_ctr_coeff).")
    M.vector(p + "_fc", pk["F"], doc="F per Cartesian AO: PySCF-Cartesian = F x unit-Cartesian.")
    ao = {"cart": pk["ao_l"], "sph": sph_ao_l(pk)}
    tags = {}
    for k in kinds:
        tags[k] = emit_basis(M, P, p, k, ints[k], k in fullset, ao[k])
    if kind in ("sph", "mixed"):
        M.matrix(p + "_m", ints["M"], doc="M = diag(F) cart2sph_coeff(), [%d,%d]: X_sph = M^T X_unit M." % (nc, ns))
    # Both full sets would pass the per-module budget (OH 6-31G*: 11,781 + 9,316
    # values), so the Cartesian set is stored and the spherical one is M^4 of it.
    # The generator asserted M^4 G_unit = G_sph to 1e-13 above; the test below
    # checks the result against PySCF's spherical checksums, diagonal and samples.
    derived = (kind == "mixed" and "cart" in fullset and "sph" not in fullset
               and ns <= FULL_ERI_MAX_NAO)
    if derived:
        m4_helpers(M)
        g_s = ints["sph"]["g"]
        g_m = np.einsum("ijkl,ia,jb,kc,ld->abcd", ints["cart"]["G"], ints["M"], ints["M"], ints["M"], ints["M"],
                        optimize=True)
        check(maxabs(pack8(g_m)[0] - g_s) < 1e-13, "M^4 of the packed Cartesian set vs PySCF's spherical set")
        M.claim(p + "_eri8_sph")
        M.out("#| Every packed-8 spherical ERI, computed from the stored unit-Cartesian set as",
              "#| M^4 (ij|kl): PySCF's spherical set agrees to 1e-13.",
              "def %s_eri8_sph() -> Arr {" % p,
              "    return %s_eri8_m4(%s_eri8_cart(), %s_m(), %d, %d)" % (M.mp, p, p, nc, ns),
              "}", "")
    if kind == "pure":
        for x in ("s", "t", "v"):
            M.alias("%s_%s_sph" % (p, x), "%s_%s_unit" % (p, x))
        M.alias(p + "_eri8_sph", p + "_eri8_cart", doc="Only s and p: the spherical and Cartesian sets are the same.")
        M.claim(p + "_m")
        M.out("def %s_m() -> Arr {" % p, "    return %s_eye(%d)" % (M.mp, nc), "}", "")

    # tests on the packs and integrals
    body = ["assert_near(%s_pack_bad(%s_shells(), %s_NSH, %s_NPRIMTOT, %s_NATOM), 0.0, 0.0)" % (M.mp, p, P, P, P)]
    dsum = int(sum(int(s[6]) - int(s[7]) for s in pk["shells"]))
    body.append("# nao_cart - nao_sph = sum over shells of ncart - nsph: 1 per d shell, 3 per f shell")
    body.append("assert_near(%s_NAO_CART - %s_NAO_SPH, %d.0, 0.0)" % (P, P, dsum))
    body.append("assert_near(len(%s_fc()), %s_NAO_CART, 0.0)" % (p, P))
    M.test("%s packs" % P.lower(), body)
    for k in kinds:
        b = ints[k]
        t = tags[k]
        n = b["n"]
        sf = sfx[k]
        body = []
        for x in ("s", "t", "v"):
            body.append("assert_near(%s_asym(%s_%s_%s(), %d), 0.0, 1.0e-14)" % (M.mp, p, x, t, n))
        body.append("assert_near(%s_diag_dev(%s_s_%s(), %d), 0.0, 1.0e-14)" % (M.mp, p, t, n))
        M.test("%s %s one-electron integrals are symmetric with a unit overlap diagonal" % (P.lower(), t), body)
        PK = "%s_ERI8_%s" % (P, k.upper())
        pkn = "%s_eri8_%s" % (p, k)
        if k in fullset:
            body = ["assert_near(len(%s()), %s_LEN, 0.0)" % (pkn, PK),
                    "assert_near(%s_sumsq(%s()) / %s_SUMSQ, 1.0, 1.0e-12)" % (M.mp, pkn, PK),
                    "assert_near(%s_wsum(%s(), %s), %s_WSUM, 1.0e-12 * %s_SUMSQ)" % (M.mp, pkn, M.seed, PK, PK),
                    "assert_lt(%s_schwarz(%s(), %d), 1.0e-12)" % (M.mp, pkn, n)]
            M.test("%s %s packed ERIs match their checksums and the Schwarz bound" % (P.lower(), t), body)
        else:
            body = ["assert_near(len(%s_idx()), 256.0, 0.0)" % pkn,
                    "assert_near(len(%s_diag()), %d.0, 0.0)" % (pkn, n * (n + 1) // 2),
                    "assert_lt(%s_sample_schwarz(%s_idx(), %s_val(), %s_diag()), 1.0e-12)" % (M.mp, pkn, pkn, pkn),
                    "assert_lt(0.0, %s_SUMSQ)" % PK]
            M.test("%s %s sampled ERIs obey the Schwarz bound" % (P.lower(), t), body)
    if derived:
        PK = "%s_ERI8_SPH" % P
        pkn = "%s_eri8_sph" % p
        q = M.mp  # test locals carry the module prefix (library rule)
        body = ["%s_g = %s()" % (q, pkn),
                "assert_near(len(%s_g), %s_LEN, 0.0)" % (q, PK),
                "assert_near(%s_sumsq(%s_g) / %s_SUMSQ, 1.0, 1.0e-12)" % (M.mp, q, PK),
                "assert_near(%s_wsum(%s_g, %s), %s_WSUM, 1.0e-12 * %s_SUMSQ)" % (M.mp, q, M.seed, PK, PK),
                "assert_lt(%s_sample_dev(%s_g, %s_idx(), %s_val()), 1.0e-13)" % (M.mp, q, pkn, pkn),
                "assert_lt(%s_schwarz(%s_g, %d), 1.0e-12)" % (M.mp, q, ns),
                "# the pair diagonal (ij|ij) sits at ij(ij+1)/2 + ij",
                "%s_gd = %s_diag()" % (q, pkn),
                "%s_w = 0.0" % q,
                "for %s_a to len(%s_gd) {" % (q, q),
                "    %s_w = max(%s_w, abs(%s_g[%s_a * (%s_a + 1) / 2 + %s_a] - %s_gd[%s_a]))" % ((q,) * 8),
                "}",
                "assert_lt(%s_w, 1.0e-13)" % q]
        M.test("%s sph ERIs from M^4 of the Cartesian set match PySCF's checksums, diagonal and samples"
               % P.lower(), body)
    if kind == "mixed":
        body = []
        for x in ("s", "t", "v"):
            body.append("assert_lt(%s_mtxm_dev(%s_m(), %s_%s_unit(), %s_%s_sph(), %d, %d), 1.0e-13)"
                        % (M.mp, p, p, x, p, x, nc, ns))
        M.test("%s M^T X_unit M is the spherical X" % P.lower(), body)

    # configurations
    results = {}
    for (key, method, charge, spin) in S["configs"]:
        for k in kinds:
            r = config_block(ints, k, P, key, method, charge, spin, S["atom"], S["basis"], report)
            results[(key, k)] = r
            emit_config(M, P, p, sfx[k], ints[k], r, k in fullset or (derived and k == "sph"), tags[k], k)
    # orderings between configurations: UHF <= ROHF at the same charge and spin
    for (key, method, charge, spin) in S["configs"]:
        if method != "UHF":
            continue
        for (key2, method2, charge2, spin2) in S["configs"]:
            if method2 == "ROHF" and charge2 == charge and spin2 == spin:
                body = []
                for k in kinds:
                    ru, rr = results[(key, k)], results[(key2, k)]
                    check(ru["e_default"] <= rr["e_default"] + 1e-12 and ru["e_stable"] <= rr["e_stable"] + 1e-12,
                          "E_UHF > E_ROHF for %s %s" % (P, key))
                    sf = sfx[k].upper()
                    for st in ("DEFAULT", "STABLE"):
                        body.append("assert_lt(%s_E_%s_%s%s, %s_E_%s_%s%s + 1.0e-12)"
                                    % (P, key, st, sf, P, key2, st, sf))
                M.test("%s E_%s <= E_%s" % (P.lower(), key.lower(), key2.lower()), body)
    if S.get("eps1e"):
        eps = eps1e(ints["mol"])
        M.const(P + "_EPS1E", eps, doc="Lowest eigenvalue of H_core C = S C e.")
    if P == "HE_STO3G":
        M.const(P + "_ERI_1111", ints["sph"]["g"][0], doc="(11|11), pyscf/gto/test/test_mole.py:942.")
    if P == "H4_DIFFUSE":
        sev = np.linalg.eigvalsh(ints["sph"]["S"])
        M.const(P + "_S_MIN_EIG", sev[0], doc="Smallest overlap eigenvalue: below the 1e-8 threshold.")
        M.const(P + "_NDROP", int(np.sum(sev < 1e-8)), integer=True,
                doc="Vectors canonical orthogonalization drops at threshold 1e-8.")
    if S.get("alias_rhf"):
        r = results[("RHF", "cart")]
        M.const(P + "_E_DEFAULT", r["e_default"], doc="The RHF state PySCF lands on from minao; internally unstable.")
        M.const(P + "_E_STABLE", r["e_stable"], doc="The RHF state reached by restarting from the stability orbitals.")
    return ints, kinds, results


def emit_config(M, P, p, sfx, b, r, full, tag, k):
    key, method = r["key"], r["method"]
    SF = sfx.upper()
    n = b["n"]
    M.out("# ---- %s %s%s: nalpha %d, nbeta %d; default state %s ----"
          % (P, key, " (cart)" if sfx else "", r["nalpha"], r["nbeta"],
             {1: "is stable", 0: "is not the lowest", -1: "not analyzed"}[r["is_stable"]]), "")
    M.const("%s_E_%s%s" % (P, key, SF), r["e_default"])
    M.const("%s_E_%s_DEFAULT%s" % (P, key, SF), r["e_default"])
    M.const("%s_E_%s_STABLE%s" % (P, key, SF), r["e_stable"])
    M.const("%s_%s_IS_STABLE%s" % (P, key, SF), r["is_stable"], integer=True,
            doc="1: no restart from the stability orbitals lowers the minao state; 0: one does; "
            "-1: not analyzed (%s)." % r["how"])
    M.const("%s_%s_NALPHA%s" % (P, key, SF), r["nalpha"], integer=True)
    M.const("%s_%s_NBETA%s" % (P, key, SF), r["nbeta"], integer=True)
    if "s2_default" in r:
        M.const("%s_%s_S2%s" % (P, key, SF), r["s2_default"], doc="<S^2> of the default UHF state.")
        M.const("%s_%s_S2_STABLE%s" % (P, key, SF), r["s2_stable"])
    for guess, eg in r.get("guess_states", []):
        if method == "RHF":
            doc = "Flagged: the %s guess converges to this excited RHF solution." % guess
        else:
            doc = "Third state: the %s guess converges here, to neither the default nor the stable state." % guess
        M.const("%s_E_%s_GUESS_%s%s" % (P, key, guess.upper(), SF), eg, doc=doc)
    M.out("")
    s = abs(r["spin"]) / 2.0
    body = ["assert_lt(%s_E_%s_STABLE%s, %s_E_%s_DEFAULT%s + 1.0e-12)" % (P, key, SF, P, key, SF)]
    if "s2_default" in r:
        body.append("assert_lt(%r - 1.0e-10, %s_%s_S2%s)" % (s * (s + 1), P, key, SF))
        body.append("assert_lt(%r - 1.0e-10, %s_%s_S2_STABLE%s)" % (s * (s + 1), P, key, SF))
    if n <= FULL_ERI_MAX_NAO:
        states_ = [("", r["d_default"], "DEFAULT")]
        if r["is_stable"] == 0:
            states_.append(("_stable", r["d_stable"], "STABLE"))
        for st, ds, lab in states_:
            base = "%s_%s%s" % (p, key.lower(), st)
            SS = "%s_%s%s_SELF_STEP%s" % (P, key, st.upper(), SF)
            M.const(SS, r["step_" + lab.lower()],
                    doc="Max |D' - D| after one PySCF step (get_fock, eig, get_occ, make_rdm1) from the "
                    "%s density below." % lab.lower())
            body.append("assert_lt(%s, 1.0e-10)" % SS)
            if len(ds) == 1:
                M.matrix(base + "_d" + sfx, ds[0], doc="Converged total density (alpha + beta), %s state." % lab.lower())
                da = "%s_d%s() * 0.5" % (base, sfx)
                db = da
                ne = "%s_trace2(%s_d%s(), %s_s_%s(), %d)" % (M.mp, base, sfx, p, tag, n)
                body.append("assert_near(%s_asym(%s_d%s(), %d), 0.0, 1.0e-12)" % (M.mp, base, sfx, n))
            else:
                M.matrix(base + "_da" + sfx, ds[0], doc="Converged alpha density, %s state." % lab.lower())
                M.matrix(base + "_db" + sfx, ds[1], doc="Converged beta density, %s state." % lab.lower())
                da = "%s_da%s()" % (base, sfx)
                db = "%s_db%s()" % (base, sfx)
                ne = "%s_trace2(%s, %s_s_%s(), %d) + %s_trace2(%s, %s_s_%s(), %d)" % (
                    M.mp, da, p, tag, n, M.mp, db, p, tag, n)
                body.append("assert_near(%s_asym(%s, %d) + %s_asym(%s, %d), 0.0, 1.0e-12)" % (M.mp, da, n, M.mp, db, n))
            body.append("assert_near(%s, %d.0, 1.0e-10)" % (ne, r["nalpha"] + r["nbeta"]))
            if full:
                body.append("assert_near(%s_e_elec(%s_t_%s() + %s_v_%s(), %s_eri8_%s(), %s, %s, %d) + %s_E_NUC, %s_E_%s_%s%s, 1.0e-10)"
                            % (M.mp, p, tag, p, tag, p, k, da, db, n, P, P, key, lab, SF))
    M.test("%s %s%s energies, labels and densities" % (P.lower(), key.lower(), " cart" if sfx else ""), body)


def header(M, name, extra):
    M.out(
        "###",
        "post: " + TITLES[name],
        "agent: claude",
        "",
        "Generated by `tools/hf_reference.py` from PySCF %s. Do not edit; rerun the" % pyscf.__version__,
        "generator. Settings: conv_tol %g, init_guess minao, symmetry off, and each" % CONV_TOL,
        "molecule built with cart=False and with cart=True. Every stored density is",
        "then re-converged from itself at conv_tol_grad 1e-10 or tighter, and its",
        "`_SELF_STEP` constant says how far one PySCF step moves it (below %g)." % SELF_STEP_MAX,
        "",
    )
    M.out(*extra)
    M.out("###", "")


SYSTEM_TEXT = [
    "Conventions:",
    "",
    "- atoms[natm,4] = (Z, x, y, z) in Bohr; shells[nsh,8] = (atom, l, nprim, p0,",
    "  cart0, sph0, ncart, nsph); prims[nprim,2] = (exponent, raw bas_ctr_coeff).",
    "  General contractions are split into segmented shells, and primitives whose",
    "  coefficient is exactly zero are dropped.",
    "- `_unit` matrices are unit-Cartesian: PySCF's Cartesian integrals divided by",
    "  F_i F_j, so every component has self-overlap 1. `_sph` matrices are PySCF's",
    "  spherical ones, and `_m()` is M with X_sph = M^T X_unit M.",
    "- Packed-8 ERIs in chemist's notation: ij = i(i+1)/2 + j for i >= j, and",
    "  ijkl = ij(ij+1)/2 + kl for ij >= kl. Full when the basis has at most 19",
    "  functions; otherwise the pair diagonal and 256 seeded samples. Both carry",
    "  _SUMSQ = sum Gp^2 and _WSUM = sum w_k Gp_k, where w_k = s_k / (2^31 - 1),",
    "  s_k = 16807 s_(k-1) mod (2^31 - 1) and s_0 = the module's _SEED_WSUM (%d)." % SEED_WSUM,
    "- Energies: `_E_<CFG>_DEFAULT` is the state PySCF reaches from the minao",
    "  guess, `_E_<CFG>_STABLE` the lowest state reached by restarting from the",
    "  stability-analysis orbitals until the energy stops dropping, and `_E_<CFG>`",
    "  repeats the default. Densities (total for RHF, alpha and beta otherwise) are",
    "  in the same basis convention as the matrices beside them: a unit-Cartesian",
    "  density is F_i F_j times PySCF's. A `_stable_` density appears only when the",
    "  stable state differs from the default.",
    "- In a module with both bases, spherical names carry no suffix and Cartesian",
    "  ones end in `_cart` / `_CART`.",
]


def gen_system_module(name, report):
    specs = SPECS[name]
    # helpers share the system's prefix when there is one system, so a module has
    # one prefix; a module with several systems uses its own name
    mp = specs[0]["P"].lower() if len(specs) == 1 else name.replace("_", "")
    M = Module(name, mp)
    extra = list(SYSTEM_TEXT)
    if any(S["kind"] == "mixed" and S.get("full") == ("cart",) for S in specs):
        extra += [
            "- One full packed set fits in this module's 20,000-value budget, not two, so",
            "  the Cartesian set is stored and `_eri8_sph()` computes the full spherical",
            "  set as M^4 of it. A test holds it to PySCF's spherical checksums, pair",
            "  diagonal and samples.",
        ]
    header(M, name, extra)
    # each module has its own copy of the seed: one shared name defined in two
    # modules would be a duplicate global in any program that uses both
    M.const(M.seed, SEED_WSUM, integer=True, doc="Seed of the LCG weights in every _WSUM checksum.")
    M.out("")
    helpers(M)
    summary = []
    for S in specs:
        ints, kinds, results = emit_system(M, S, report)
        summary.append((S, ints, kinds, results))
    txt = M.text()
    check(M.nvalues <= MODULE_LIMIT, "%s: %d values > %d" % (name, M.nvalues, MODULE_LIMIT))
    S, ints, kinds, results = summary[0]
    k0 = kinds[0]
    key0 = S["configs"][0][0]
    line = "%s nao %d E_%s %r" % (name, ints[k0]["n"], key0.lower(), results[(key0, k0)]["e_default"])
    return M, txt, line


# ---------------------------------------------------------------------------
# c2s: the unit-Cartesian M_l, one shell at a time


def gen_c2s(report):
    M = Module("c2s", "c2s")
    header(M, "c2s", [
        "M_l maps the unit-Cartesian components of one shell (PySCF order: lx from l",
        "down, then ly) to PySCF's spherical ones (p as x, y, z; d as xy, yz, z^2,",
        "xz, x^2 - y^2; f and g as m = -l..l): M_l = diag(F) cart2sph(l, 'sp'), with",
        "F^2 = 4 pi (2i-1)!! (2j-1)!! (2k-1)!! / (2l+1)!! for l >= 2 and F = 1 below.",
        "",
        "Also the two constants pyscf/gto/test/test_mole.py:520 and :529 assert: the",
        "s and p factors of the unnormalized cart2sph, 1/(2 sqrt(pi)) and",
        "sqrt(3/(4 pi)).",
    ])
    helpers(M)
    s_const = float(gto.cart2sph(0)[0, 0])
    p_const = float(gto.cart2sph(1)[0, 0])
    check(abs(s_const - 0.282094791773878143) < 1e-15 and abs(p_const - 0.488602511902919921) < 1e-15,
          "test_mole.py:520/529 constants")
    M.const("C2S_S_FACTOR", s_const, doc="gto.cart2sph(0): test_mole.py:520 asserts 0.282094791773878143.")
    M.const("C2S_P_FACTOR", p_const, doc="gto.cart2sph(1) diagonal: test_mole.py:529 asserts 0.488602511902919921.")
    M.const("C2S_LMAX", 4, integer=True)
    M.const("C2S_PI", math.pi)
    M.out("")
    # powers(l) against PySCF's own labels, l = 0..4, on a single-shell molecule
    for l in range(5):
        mol = gto.M(atom="H 0 0 0", basis={"H": [[l, [1.0, 1.0]]]}, spin=1, cart=True, verbose=0)
        labs = [x[3] for x in mol.ao_labels(fmt=False)]
        check(labs == ["x" * a + "y" * b + "z" * c for (a, b, c) in powers(l)], "powers(%d) vs cart_labels" % l)
        diag = np.diag(mol.intor("int1e_ovlp_cart"))
        F2 = np.array([fc2(pw) for pw in powers(l)])
        check(maxabs(diag - F2) < 1e-14, "diag(ovlp_cart) = F^2 for l = %d" % l)
    body = ["assert_near(C2S_S_FACTOR, 1.0 / (2.0 * sqrt(C2S_PI)), 1.0e-16)",
            "assert_near(C2S_P_FACTOR, sqrt(3.0 / (4.0 * C2S_PI)), 1.0e-16)"]
    for l in range(5):
        nc, ns = ncart(l), 2 * l + 1
        blk = np.eye(nc) if l <= 1 else gto.cart2sph(l, normalized="sp")
        F = np.array([math.sqrt(fc2(pw)) for pw in powers(l)])
        Ml = F[:, None] * blk
        M.matrix("c2s_m%d" % l, Ml, doc="M_%d, [%d,%d]." % (l, nc, ns))
        M.matrix("c2s_powers%d" % l, np.array(powers(l)), integer=True, doc="Powers (i, j, k) of each component.")
        M.vector("c2s_fc2_%d" % l, np.array([fc2(pw) for pw in powers(l)]), doc="F^2 per component.")
        # the spherical functions are orthonormal under the one-center unit-Cartesian overlap
        body.append("assert_lt(c2s_mtxm_dev(c2s_m%d(), c2s_shell_overlap(c2s_powers%d(), %d), c2s_eye(%d), %d, %d), 1.0e-14)"
                    % (l, l, nc, ns, nc, ns))
        body.append("assert_lt(c2s_fc2_dev(c2s_powers%d(), c2s_fc2_%d(), %d), 1.0e-15)" % (l, l, nc))
    M.claim("c2s_dfact")
    M.claim("c2s_shell_overlap")
    M.claim("c2s_fc2_dev")
    M.out(
        "#| (n)!! with (-1)!! = 1.",
        "def c2s_dfact(n) {",
        "    r = 1.0",
        "    k = n",
        "    while k > 1 {",
        "        r = r * k",
        "        k = k - 2",
        "    }",
        "    return r",
        "}",
        "",
        "#| Overlap of two unit-normalized components on one center with one exponent:",
        "#| per axis (a + b - 1)!! / sqrt((2a - 1)!! (2b - 1)!!) when a + b is even, else 0.",
        "def c2s_shell_overlap(pw: Arr, nc) -> Arr {",
        "    s = zeros(nc, nc)",
        "    for i to nc {",
        "        for j to nc {",
        "            t = 1.0",
        "            for d to shape(pw)[1] {",
        "                a = pw[i, d]",
        "                b = pw[j, d]",
        "                if mod(a + b, 2.0) == 1.0 {",
        "                    t = 0.0",
        "                } else {",
        "                    t = t * c2s_dfact(a + b - 1) / sqrt(c2s_dfact(2 * a - 1) * c2s_dfact(2 * b - 1))",
        "                }",
        "            }",
        "            s[i, j] = t",
        "        }",
        "    }",
        "    return s",
        "}",
        "",
        "#| Largest relative gap between the tabulated F^2 and the closed form.",
        "def c2s_fc2_dev(pw: Arr, f2: Arr, nc) {",
        "    w = 0.0",
        "    for i to nc {",
        "        a = pw[i, 0]",
        "        b = pw[i, 1]",
        "        c = pw[i, 2]",
        "        l = a + b + c",
        "        x = 1.0",
        "        if l >= 2 {",
        "            x = 4.0 * C2S_PI * c2s_dfact(2 * a - 1) * c2s_dfact(2 * b - 1) * c2s_dfact(2 * c - 1) / c2s_dfact(2 * l + 1)",
        "        }",
        "        w = max(w, abs(f2[i] - x) / x)",
        "    }",
        "    return w",
        "}",
        "",
    )
    M.test("c2s spherical functions are orthonormal and F matches its closed form", body)
    report.append("  powers(l) = cart_labels and diag(ovlp_cart) = F^2 for l = 0..4")
    return M, M.text(), "c2s l 0..4 S %r P %r" % (s_const, p_const)


# ---------------------------------------------------------------------------
# prims: single primitives up to l = 3, normalization divided out

NPRIM_CASES = 64
NPICK = 8


def gen_prims(report):
    M = Module("prims", "prims")
    header(M, "prims", [
        "64 seeded cases. Each has four single-primitive shells a, b, c, d (angular",
        "momentum up to 3, exponent log-uniform in [0.1, 10], center in a box of",
        "side 2 Bohr) and a point charge Z = 1 at C. Case k has l_a = k mod 4,",
        "l_b = (k / 4) mod 4 and l_c = l_d = (k / 16) mod 4, so every (l_a, l_b) pair",
        "and (dd|dd) through (ff|ff) appear.",
        "",
        "Every value is for unit-normalized Cartesian components (self-overlap 1),",
        "which is PySCF's cart=True value divided by F for each index. For each case,",
        "eight component picks (ia, ib, ic, id), index into powers(l) in PySCF order:",
        "S_ab, T_ab, V_ab = -<a|1/|r - C||b>, (ab|cd), and the Schwarz diagonals",
        "(ab|ab) and (cd|cd).",
        "",
        "prims_shells() rows: (l, exponent, x, y, z), four rows per case.",
    ])
    helpers(M)
    rng = Lcg(424242)
    shells, cpts, picks, S_, T_, V_, G_, GA, GC = [], [], [], [], [], [], [], [], []
    for case in range(NPRIM_CASES):
        ls = [case % 4, (case // 4) % 4, (case // 16) % 4, (case // 16) % 4]
        sh = []
        for l in ls:
            a = 0.1 * (100.0 ** rng.next())
            c = [2.0 * rng.next() - 1.0 for _ in range(3)]
            sh.append((l, a, c))
            shells.append([l, a] + c)
        C = [2.0 * rng.next() - 1.0 for _ in range(3)]
        cpts.append(C)
        atoms = [["H@%d" % (k + 1), sh[k][2]] for k in range(4)]
        basis = {"H@%d" % (k + 1): [[sh[k][0], [sh[k][1], 1.0]]] for k in range(4)}
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
        F = np.concatenate([[math.sqrt(fc2(pw)) for pw in powers(l)] for l in ls])
        off = np.cumsum([0] + [ncart(l) for l in ls])
        FF = np.outer(F, F)
        S = mol.intor("int1e_ovlp_cart") / FF
        T = mol.intor("int1e_kin_cart") / FF
        with mol.with_rinv_origin(C):
            V = -mol.intor("int1e_rinv_cart") / FF
        G = mol.intor("int2e_cart") / np.einsum("p,q,r,s->pqrs", F, F, F, F)
        check(maxabs(np.diag(S) - 1.0) < 1e-13, "unit self-overlap of a primitive")
        for q in range(NPICK):
            pk = [int(rng.next() * ncart(ls[k])) for k in range(4)]
            ia, ib, ic, id_ = [off[k] + pk[k] for k in range(4)]
            picks.append(pk)
            S_.append(S[ia, ib])
            T_.append(T[ia, ib])
            V_.append(V[ia, ib])
            G_.append(G[ia, ib, ic, id_])
            GA.append(G[ia, ib, ia, ib])
            GC.append(G[ic, id_, ic, id_])
    check(min(abs(x) for x in G_ if x != 0.0) > 1e-12, "a sampled primitive ERI is too small to compare")
    M.const("PRIMS_NCASE", NPRIM_CASES, integer=True)
    M.const("PRIMS_NPICK", NPICK, integer=True)
    M.out("")
    M.matrix("prims_shells", np.array(shells), doc="(l, exponent, x, y, z), four rows per case, Bohr.")
    M.matrix("prims_charge_at", np.array(cpts), doc="Where the unit point charge sits, per case.")
    M.matrix("prims_picks", np.array(picks), integer=True, doc="(ia, ib, ic, id) per pick, 8 picks per case.")
    M.vector("prims_s", S_, doc="S_ab per pick.")
    M.vector("prims_t", T_, doc="T_ab per pick.")
    M.vector("prims_v", V_, doc="-<a|1/|r - C||b> per pick.")
    M.vector("prims_eri", G_, doc="(ab|cd) per pick.")
    M.vector("prims_eri_abab", GA, doc="(ab|ab) per pick.")
    M.vector("prims_eri_cdcd", GC, doc="(cd|cd) per pick.")
    M.claim("prims_bad")
    M.out(
        "#| Count the picks that break a bound every correct integral obeys:",
        "#| |S_ab| <= 1 for unit functions, (ab|cd)^2 <= (ab|ab)(cd|cd), (ab|ab) >= 0.",
        "def prims_bad(s: Arr, g: Arr, ga: Arr, gc: Arr) {",
        "    bad = 0",
        "    for k to len(s) {",
        "        if abs(s[k]) > 1.0 + 1.0e-14 {",
        "            bad = bad + 1",
        "        }",
        "        if abs(g[k]) > sqrt(ga[k]) * sqrt(gc[k]) * (1.0 + 1.0e-12) {",
        "            bad = bad + 1",
        "        }",
        "        if ga[k] < 0.0 or gc[k] < 0.0 {",
        "            bad = bad + 1",
        "        }",
        "    }",
        "    return bad",
        "}",
        "",
    )
    M.test("prims obey the overlap and Schwarz bounds", [
        "assert_near(len(prims_s()), PRIMS_NCASE * PRIMS_NPICK, 0.0)",
        "assert_near(prims_bad(prims_s(), prims_eri(), prims_eri_abab(), prims_eri_cdcd()), 0.0, 0.0)",
    ])
    return M, M.text(), "prims cases %d picks %d" % (NPRIM_CASES, NPICK)


# ---------------------------------------------------------------------------
# audit: literal statistics of the files on disk


def audit():
    worst_lit, worst_mod = 0, 0
    for name in MODULES:
        path = os.path.join(OUTDIR, name + ".viv")
        text = open(path).read()
        lits = re.findall(r"def \w+\(\) -> Arr \{\n    return \[(.*?)\]\n\}", text, re.S)
        sizes = [len([x for x in c.replace("\n", " ").split(",") if x.strip()]) for c in lits]
        consts = len(re.findall(r"^[A-Z][A-Z0-9_]* = ", text, re.M))
        total = sum(sizes) + consts
        ml = max(sizes) if sizes else 0
        worst_lit, worst_mod = max(worst_lit, ml), max(worst_mod, total)
        print("%-22s values %6d  literals %4d  largest literal %6d  bytes %8d" % (name, total, len(sizes), ml, len(text)))
    print("max literal %d" % worst_lit)
    print("max module values %d (limit %d)" % (worst_mod, MODULE_LIMIT))
    return 0 if worst_mod <= MODULE_LIMIT else 1


def main():
    if "--audit" in sys.argv[1:]:
        return audit()
    check(pyscf.__version__ == "2.5.0", "PySCF %s; the references are pinned to 2.5.0" % pyscf.__version__)
    legacy = legacy_names()
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    written = []
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for a in only:
        check(a in MODULES, "no module " + a)
    for name in MODULES:
        if only and name not in only:
            continue
        if name == "c2s":
            M, txt, line = gen_c2s(report)
        elif name == "prims":
            M, txt, line = gen_prims(report)
        else:
            M, txt, line = gen_system_module(name, report)
        clash = sorted(M.names & legacy)
        check(not clash, "%s: names collide with the legacy modules: %s" % (name, clash))
        check(M.nvalues <= MODULE_LIMIT, "%s literal budget" % name)
        written.append((name, txt))
        print(line)
    # names must not repeat across modules either
    seen = {}
    for name, txt in written:
        for m in re.finditer(r"^(?:def\s+(\w+)|([A-Za-z_]\w*)\s*=)", txt, re.M):
            nm = m.group(1) or m.group(2)
            check(seen.setdefault(nm, name) == name, "%s is defined in %s and %s" % (nm, seen.get(nm), name))
    for name, txt in written:
        with open(os.path.join(OUTDIR, name + ".viv"), "w") as f:
            f.write(txt)
    for r in report:
        print(r)
    # a GUESS_STATES entry nobody reached is stale: the table must match the runs
    if len(sys.argv) == 1:
        check(GUESS_SEEN == set(GUESS_STATES), "GUESS_STATES entries never reached: %s"
              % sorted(set(GUESS_STATES) - GUESS_SEEN))
    print("asserts held: M^T S_unit M vs S_sph < 1e-13; M^4 G_unit vs G_sph < 1e-13; "
          "powers(l) = cart_labels for l <= 4; diag(ovlp_cart) = F^2 to 1e-14; "
          "minao, atom, 1e and huckel guesses agree to 1e-10 in all %d RHF runs "
          "(N2 huckel flagged); every open-shell guess lands on a labeled state"
          % len(RHF_GUESS_RUNS))
    print("wrote %d modules to lib/hfref" % len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())

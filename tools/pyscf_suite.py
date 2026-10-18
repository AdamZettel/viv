# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The PySCF v2.5.0 SCF tests the Viv Hartree-Fock code is held to.

    python3 tools/pyscf_suite.py --validate    re-run every manifest row with local PySCF
    python3 tools/pyscf_suite.py --measure     re-run, and rewrite the measured/target columns
    python3 tools/pyscf_suite.py --emit        write lib/hfref/pyscf_tests.viv from the manifest
    python3 tools/pyscf_suite.py --src DIR     also check each row's literal against the v2.5.0
                                               test sources saved in DIR (see SOURCES below)

The manifest is tools/pyscf_suite.tsv: one row per assertion harvested from
pyscf/scf/test and pyscf/gto/test at tag v2.5.0, classified by what a Viv code
needs before it can reproduce the number:

    T1     s and p functions, RHF or a closed form; reproducible first
    T1s    the one assertTrue in the set (the Schwarz inequality)
    T1x    s and p RHF, plus one SCF-driver feature (EDIIS, DIIS rollback or
           restore, state following, dynamic level shift, canonical
           orthogonalization, batched J/K on an imported complex density)
    T2     needs d functions, RHF
    T3     needs UHF or ROHF, s and p functions
    T2+T3  needs both

A row tagged `path` asserts a number that depends on the iteration path --
the guess, the DIIS variant, how many cycles ran -- not only on the state the
SCF converges to. A scorer counts those separately: a correct SCF with a
different DIIS can miss them and still be right.

Columns: loc (file:line), test, tier, system, method, asserted (the right-hand
side as written in the test), places, measured (the left-hand side, computed
here), target (the right-hand side as a number: the literal, or the locally
evaluated expression when the test compares two computed quantities), note
(which solution the number belongs to), tags.

Each row is re-run the way its test runs it: same molecule, same class, same
settings, same random seed. Whether it passes is decided with unittest's own
rule, round(abs(lhs - rhs), places) == 0, so "validated" here means exactly
what a green PySCF test run means.
"""
import functools
import math
import os
import sys
import tempfile

import numpy
from pyscf import gto, lib, scf
from pyscf.scf import addons, diis

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TSV = os.path.join(HERE, "pyscf_suite.tsv")
VIV = os.path.join(ROOT, "lib", "hfref", "pyscf_tests.viv")

COLUMNS = ["loc", "test", "tier", "system", "method", "asserted", "places",
           "measured", "target", "note", "tags"]
TIERS = ["T1", "T1s", "T1x", "T2", "T3", "T2+T3"]
# The counts the plan fixes. A row moving between tiers is a decision, so it
# has to show up as a changed number here and not slip through.
TIER_COUNTS = {"T1": 17, "T1s": 1, "T1x": 8, "T2": 33, "T3": 19, "T2+T3": 30}

# The raw files at raw.githubusercontent.com/pyscf/pyscf/v2.5.0/pyscf/<dir>/test/<name>,
# saved as <dir>_test_<name> for --src.
SOURCES = {
    "test_h2o.py": "scf_test_test_h2o.py", "test_h2o_vdz.py": "scf_test_test_h2o_vdz.py",
    "test_he.py": "scf_test_test_he.py", "test_rhf.py": "scf_test_test_rhf.py",
    "test_uhf.py": "scf_test_test_uhf.py", "test_diis.py": "scf_test_test_diis.py",
    "test_addons.py": "scf_test_test_addons.py", "test_bz.py": "scf_test_test_bz.py",
    "test_mole.py": "gto_test_test_mole.py", "test_moleintor.py": "gto_test_test_moleintor.py",
}

WATER = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
WATER_STRETCHED = "O 0 0 0; H 0 -1.757 1.587; H 0 1.757 1.587"
N2 = "N 0 0 0; N 0 0 1"
BENZENE = [
    ("C", (-0.65830719, 0.61123287, -0.00800148)),
    ("C", (0.73685281, 0.61123287, -0.00800148)),
    ("C", (1.43439081, 1.81898387, -0.00800148)),
    ("C", (0.73673681, 3.02749287, -0.00920048)),
    ("C", (-0.65808819, 3.02741487, -0.00967948)),
    ("C", (-1.35568919, 1.81920887, -0.00868348)),
    ("H", (-1.20806619, -0.34108413, -0.00755148)),
    ("H", (1.28636081, -0.34128013, -0.00668648)),
    ("H", (2.53407081, 1.81906387, -0.00736748)),
    ("H", (1.28693681, 3.97963587, -0.00925948)),
    ("H", (-1.20821019, 3.97969587, -0.01063248)),
    ("H", (-2.45529319, 1.81939187, -0.00886348)),
]


def M(**kw):
    # The tests send their log to /dev/null at verbose 7; verbosity changes no number.
    kw.setdefault("verbose", 0)
    return gto.M(**kw)


# ---- shared fixtures: the setUpModule objects, built once ----

@functools.lru_cache(None)
def water631():
    return M(atom=WATER, basis="6-31g")


@functools.lru_cache(None)
def water631_sym():
    m = water631().copy()
    m.symmetry = True
    m.build(0, 0)
    return m


@functools.lru_cache(None)
def water_vdz():
    return M(atom=WATER, basis="cc-pvdz")


@functools.lru_cache(None)
def rhf_vdz():
    # test_rhf.py setUpModule: conv_tol 1e-10 and a checkfile, which test_update_from_chk reads.
    mf = scf.RHF(water_vdz())
    mf.conv_tol = 1e-10
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.kernel()
    return mf


@functools.lru_cache(None)
def n2sym():
    return M(atom=N2, symmetry=1, basis="cc-pvdz")


@functools.lru_cache(None)
def n2_rhf():
    return scf.RHF(n2sym()).set(conv_tol=1e-10).run()


@functools.lru_cache(None)
def uhf_vdz():
    # test_uhf.py setUpModule: conv_tol 1e-14.
    mf = scf.UHF(water_vdz())
    mf.conv_tol = 1e-14
    mf.scf()
    return mf


@functools.lru_cache(None)
def uhf_vdz_triplet():
    return scf.UHF(M(atom=WATER, basis="cc-pvdz", spin=2)).run(conv_tol=1e-10)


@functools.lru_cache(None)
def n2_uhf():
    return scf.UHF(n2sym()).set(conv_tol=1e-10).run()


@functools.lru_cache(None)
def addons_rhf():
    return scf.RHF(water631()).run()


@functools.lru_cache(None)
def he_vdz():
    return M(atom=[[2, (0., 0., 0.)]], basis={"He": "cc-pvdz"})


@functools.lru_cache(None)
def moleintor_benzene():
    c4 = gto.basis.parse('''
    C    S
       6665.0000000              0.0006920             -0.0001460
       1000.0000000              0.0053290             -0.0011540
        228.0000000              0.0270770             -0.0057250
         64.7100000              0.1017180             -0.0233120
         21.0600000              0.2747400             -0.0639550
          7.4950000              0.4485640             -0.1499810
          2.7970000              0.2850740             -0.1272620
          0.5215000              0.0152040              0.5445290
    C    S
          0.1596000              1.0000000
    C    P
          9.4390000              0.0381090
          2.0020000              0.2094800
          0.5456000              0.5085570
    C    P
          0.1517000              1.0000000
    C    D
          0.5500000              1.0000000        ''')
    atoms = [["C", BENZENE[0][1]], ["C1", BENZENE[1][1]], ["C2", BENZENE[2][1]],
             ["C3", BENZENE[3][1]], ["C4", BENZENE[4][1]], ["C5", BENZENE[5][1]]]
    atoms += [["H", xyz] for _, xyz in BENZENE[6:]]
    m = gto.Mole()
    m.verbose = 0
    m.atom = atoms
    m.basis = {"H": "cc-pvdz", "C1": "CC PVDZ", "C2": "CC PVDZ", "C3": "cc-pVDZ",
               "C4": c4, "C": "CC PVDZ"}
    m.ecp = {"C1": "LANL2DZ"}
    m.build()
    return m


def natm0_model(cls, spin):
    m = gto.M(verbose=0)
    if spin:
        m.spin = spin
    m.nelectron = 2
    mf = cls(m)
    mf.get_hcore = lambda *args: numpy.diag(numpy.arange(3))
    mf.get_ovlp = lambda *args: numpy.eye(3)
    mf._eri = numpy.zeros((3, 3, 3, 3))
    for i in range(3):
        mf._eri[i, i, i, i] = .2
    return mf


def cation(m, **kw):
    p = m.copy()
    p.charge = 1
    p.spin = 1
    for k, v in kw.items():
        setattr(p, k, v)
    p.build(False, False)
    return p


def symm(m, **kw):
    p = m.copy()
    p.symmetry = 1
    for k, v in kw.items():
        setattr(p, k, v)
    p.build(False, False)
    return p


def seeded(n, count=1):
    numpy.random.seed(1)
    return [numpy.random.random((n, n)) for _ in range(count)]


# ---- one recipe per row, keyed by file:line ----
# A recipe returns the left-hand side, or (lhs, rhs) when the right-hand side
# is itself computed rather than a literal.

R = {}


def row(loc):
    def reg(f):
        assert loc not in R, loc
        R[loc] = f
        return f
    return reg


# test_h2o.py: water 6-31G

@row("test_h2o.py:55")
def _():
    return scf.RHF(water631()).set(conv_tol=1e-11).scf()


@row("test_h2o.py:118")
def _():
    return scf.RHF(water631()).set(conv_tol=1e-11, max_memory=0).scf()


@row("test_h2o.py:131")
def _():
    return scf.RHF(water631()).set(conv_tol=1e-11, max_memory=0, direct_scf=False).scf()


@row("test_h2o.py:156")
def _():
    return water631().energy_nuc()


@row("test_h2o.py:164")
def _():
    m = water631().copy()
    m.symmetry = 1
    m.build()
    return scf.hf.RHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o.py:72")
def _():
    m = M(atom=WATER, basis="6-31g", charge=1, spin=1)
    return scf.ROHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o.py:182")
def _():
    m = M(atom=WATER, basis="6-31g", charge=1, spin=1, symmetry=True)
    return scf.RHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o.py:80")
def _():
    return scf.UHF(water631()).set(conv_tol=1e-11).scf()


@row("test_h2o.py:124")
def _():
    return scf.UHF(water631()).set(conv_tol=1e-11, max_memory=0).scf()


@row("test_h2o.py:138")
def _():
    return scf.UHF(water631()).set(conv_tol=1e-11, max_memory=0, direct_scf=False).scf()


@row("test_h2o.py:153")
def _():
    return scf.UHF(water631()).set(level_shift=.2).scf()


@row("test_h2o.py:190")
def _():
    m = water631().copy()
    m.symmetry = 1
    m.build()
    return scf.UHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o.py:255")
def _():
    return numpy.linalg.norm(scf.hf.init_guess_by_1e(water631()))


@row("test_h2o.py:265")
def _():
    dm = scf.rohf.ROHF(water631()).init_guess_by_1e(water631())
    return numpy.linalg.norm(dm), 5.3700828975288122 / numpy.sqrt(2)


@row("test_h2o.py:270")
def _():
    dm = scf.rohf.ROHF(water631_sym()).init_guess_by_1e(water631())
    return numpy.linalg.norm(dm), 5.3700828975288122 / numpy.sqrt(2)


# test_h2o_vdz.py: water cc-pVDZ

def vdz_run(cls, mol=None, **kw):
    return cls(mol or water_vdz()).set(**kw).scf()


R["test_h2o_vdz.py:45"] = lambda: vdz_run(scf.RHF, conv_tol=1e-11)
R["test_h2o_vdz.py:61"] = lambda: vdz_run(
    scf.rohf.ROHF, M(atom=WATER, basis="cc-pvdz", charge=1, spin=1), conv_tol=1e-11)
R["test_h2o_vdz.py:66"] = lambda: vdz_run(scf.UHF, conv_tol=1e-11)
R["test_h2o_vdz.py:98"] = lambda: vdz_run(scf.RHF, conv_tol=1e-11, max_memory=0)
R["test_h2o_vdz.py:104"] = lambda: vdz_run(scf.UHF, conv_tol=1e-11, max_memory=0)
R["test_h2o_vdz.py:111"] = lambda: vdz_run(scf.RHF, conv_tol=1e-11, max_memory=0, direct_scf=False)
R["test_h2o_vdz.py:118"] = lambda: vdz_run(scf.UHF, conv_tol=1e-11, max_memory=0, direct_scf=False)
R["test_h2o_vdz.py:133"] = lambda: vdz_run(scf.UHF, level_shift=.2)


@row("test_h2o_vdz.py:141")
def _():
    m = water_vdz().copy()
    m.symmetry = 1
    m.build()
    return scf.hf.RHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o_vdz.py:158")
def _():
    m = M(atom=WATER, basis="cc-pvdz", charge=1, spin=1, symmetry=True)
    return scf.hf_symm.ROHF(m).set(conv_tol=1e-11).scf()


@row("test_h2o_vdz.py:166")
def _():
    m = water_vdz().copy()
    m.symmetry = 1
    m.build()
    return scf.uhf_symm.UHF(m).set(conv_tol=1e-11).scf()


# test_he.py: He cc-pVDZ

@row("test_he.py:47")
def _():
    mf = scf.RHF(he_vdz())
    e, c = mf.eig(mf.get_hcore(he_vdz()), mf.get_ovlp(he_vdz()))
    return e[0]


R["test_he.py:52"] = lambda: scf.RHF(he_vdz()).set(conv_tol=1e-10).scf()
R["test_he.py:57"] = lambda: scf.UHF(he_vdz()).set(conv_tol=1e-10).scf()


# test_rhf.py

R["test_rhf.py:261"] = lambda: scf.rohf.HF1e(water_vdz()).scf()
R["test_rhf.py:264"] = lambda: scf.RHF(M(atom="H", spin=1)).kernel()
R["test_rhf.py:276"] = lambda: scf.hf_symm.HF1e(M(atom=WATER, basis="cc-pvdz", symmetry=1)).scf()


@row("test_rhf.py:280")
def _():
    mf = rhf_vdz()
    return abs(mf.get_grad(mf.mo_coeff, mf.mo_occ)).max()


@row("test_rhf.py:301")
def _():
    mf = n2_rhf()
    return mf.energy_tot(mf.make_rdm1()), mf.e_tot


@row("test_rhf.py:307")
def _():
    dm, = seeded(water_vdz().nao_nr())
    return rhf_vdz().energy_elec(dm)[0]


R["test_rhf.py:356"] = lambda: rhf_vdz().e_tot
R["test_rhf.py:386"] = lambda: scf.ROHF(M(atom=WATER, basis="6-31g", spin=-2)).run(conv_tol=1e-10).e_tot


@row("test_rhf.py:406")
def _():
    p = water_vdz().copy()
    p.cart = True
    return scf.RHF(p).run().e_tot


R["test_rhf.py:414"] = lambda: scf.rohf.ROHF(cation(water_vdz())).scf()


@row("test_rhf.py:435")
def _():
    m = water_vdz()
    numpy.random.seed(1)
    s = scf.hf.get_ovlp(m)
    d = numpy.random.random((m.nao_nr(), m.nao_nr()))
    d = d + d.T
    return numpy.linalg.norm(scf.hf.level_shift(s, d, scf.hf.get_hcore(m), .5))


@row("test_rhf.py:444")
def _():
    d1, d2 = seeded(water_vdz().nao_nr(), 2)
    return numpy.linalg.norm(scf.hf.get_veff(water_vdz(), (d1 + d1.T, d2 + d2.T)))


def cart631_complex_dm():
    p = gto.Mole()
    p.atom = WATER
    p.basis = "6-31g"
    p.cart = True
    p.verbose = 0
    p.build()
    numpy.random.seed(1)
    return p, numpy.random.random((2, 3, p.nao_nr(), p.nao_nr())) - .5 + 0j


@row("test_rhf.py:469")
def _():
    p, dm = cart631_complex_dm()
    mf1 = scf.hf.SCF(p)
    mf1.direct_scf = True
    mf1.max_memory = 0
    return lib.fp(mf1.get_veff(p, dm, hermi=0))


R["test_rhf.py:477"] = lambda: scf.hf_symm.RHF(symm(water_vdz())).scf()
R["test_rhf.py:498"] = lambda: scf.hf_symm.ROHF(symm(water_vdz(), charge=1, spin=1)).scf()
R["test_rhf.py:516"] = lambda: scf.hf_symm.RHF(n2sym()).scf()


@row("test_rhf.py:523")
def _():
    p = n2sym().copy()
    p.charge = 1
    p.spin = 1
    return scf.hf_symm.ROHF(p).scf()


def dot_eri_dm_norms():
    dm, = seeded(water_vdz().nao_nr())
    return scf.hf.dot_eri_dm(rhf_vdz()._eri, dm, hermi=0)


R["test_rhf.py:547"] = lambda: numpy.linalg.norm(dot_eri_dm_norms()[0])
R["test_rhf.py:548"] = lambda: numpy.linalg.norm(dot_eri_dm_norms()[1])


@row("test_rhf.py:562")
def _():
    m = gto.Mole()
    m.atom = [["O", (0., 0., 0.)], ["ghost", (0., -0.757, 0.587)], [1, (0., 0.757, 0.587)]]
    m.verbose = 0
    m.spin = 1
    m.symmetry = True
    m.basis = {"O": "ccpvdz", "H": "ccpvdz", "GHOST": gto.basis.load("ccpvdz", "H")}
    m.build()
    return scf.RHF(m).kernel()


R["test_rhf.py:636"] = lambda: scf.RHF(M(atom="He")).run().e_tot
R["test_rhf.py:640"] = lambda: scf.RHF(M(atom="He", charge=2)).run().e_tot


def complex_start(cls, shape, **kw):
    mf = cls(water_vdz()).set(**kw)
    mf.kernel(numpy.zeros(shape) * 0j)
    return mf.e_tot


NV = 24  # water cc-pVDZ, spherical
R["test_rhf.py:689"] = lambda: complex_start(scf.RHF, (NV, NV))
R["test_rhf.py:693"] = lambda: complex_start(scf.RHF, (NV, NV), max_memory=0)
R["test_rhf.py:697"] = lambda: complex_start(scf.rohf.ROHF, (NV, NV))
R["test_rhf.py:701"] = lambda: complex_start(scf.rohf.ROHF, (NV, NV), max_memory=0)


@row("test_rhf.py:711")
def _():
    mf = rhf_vdz()
    return scf.RHF(water_vdz()).update(mf.chkfile).e_tot, mf.e_tot


@row("test_rhf.py:728")
def _():
    mf = rhf_vdz()
    scanner = mf.as_scanner().as_scanner()
    scanner.chkfile = None
    return scanner(water_vdz()), mf.e_tot


@row("test_rhf.py:757")
def _():
    mf = natm0_model(scf.hf.RHF, 0)
    mf.kernel()
    return mf.e_tot


@row("test_rhf.py:870")
def _():
    dm, = seeded(water_vdz().nao_nr())
    mf1 = scf.RHF(water_vdz())
    mf1.max_memory = 0
    return numpy.linalg.norm(mf1.get_j(water_vdz(), dm))


def jk_identity():
    return rhf_vdz().get_jk(water_vdz(), numpy.eye(water_vdz().nao_nr(), dtype=int))


R["test_rhf.py:881"] = lambda: lib.fp(jk_identity()[0])
R["test_rhf.py:882"] = lambda: lib.fp(jk_identity()[1])


@row("test_rhf.py:897")
def _():
    dm, = seeded(water_vdz().nao)
    return lib.fp(rhf_vdz().get_k(water_vdz(), dm, hermi=0))


@row("test_rhf.py:944")
def _():
    # assertTrue(eri < q[i,j] * q[k,l]); lhs is the integral, rhs the Schwarz bound.
    m = M(atom="H 0 0 0; H 0 0 4.", unit="B", basis=[[0, (2.7, 1)], [0, (1e2, 1)]])
    mf = scf.RHF(m)
    mf.direct_scf_tol = 1e-18
    q = mf.init_direct_scf().q_cond
    i, j, k, l = 0, 2, 3, 3
    return m.intor_by_shell("int2e", (i, j, k, l)).ravel()[0], q[i, j] * q[k, l]


# test_uhf.py

R["test_uhf.py:106"] = lambda: scf.uhf.HF1e(water_vdz()).scf()
R["test_uhf.py:109"] = lambda: scf.UHF(M(atom="H", spin=-1)).kernel()
R["test_uhf.py:111"] = lambda: scf.UHF(M(atom="H", spin=1, symmetry=1)).kernel()


@row("test_uhf.py:115")
def _():
    mf2 = uhf_vdz_triplet()
    return abs(mf2.get_grad(mf2.mo_coeff, mf2.mo_occ)).max()


@row("test_uhf.py:123")
def _():
    dm = tuple(seeded(water_vdz().nao_nr(), 2))
    return uhf_vdz().energy_elec(dm)[0]


R["test_uhf.py:143"] = lambda: uhf_vdz().e_tot
R["test_uhf.py:155"] = lambda: scf.UHF(M(atom=WATER, basis="6-31g", spin=-2)).run(conv_tol=1e-10).e_tot


@row("test_uhf.py:175")
def _():
    p = water_vdz().copy()
    p.cart = True
    return scf.UHF(p).run().e_tot


@row("test_uhf.py:183")
def _():
    p = water_vdz().copy()
    p.cart = True
    p.symmetry = 1
    p.build()
    return scf.UHF(p).run().e_tot


R["test_uhf.py:186"] = lambda: uhf_vdz().spin_square(uhf_vdz().mo_coeff)[0]
R["test_uhf.py:193"] = lambda: scf.uhf_symm.UHF(symm(water_vdz())).scf()
R["test_uhf.py:207"] = lambda: scf.uhf_symm.UHF(n2sym()).scf()


@row("test_uhf.py:213")
def _():
    p = n2sym().copy()
    p.charge = 1
    p.spin = 1
    return scf.uhf_symm.UHF(p).scf()


R["test_uhf.py:297"] = lambda: complex_start(scf.UHF, (2, NV, NV))
R["test_uhf.py:301"] = lambda: complex_start(scf.UHF, (2, NV, NV), max_memory=0)


@row("test_uhf.py:308")
def _():
    mf2 = uhf_vdz_triplet()
    mo_ba = (mf2.mo_coeff[1], mf2.mo_coeff[0])
    mo_occ = uhf_vdz().mo_occ
    det, x = mf2.det_ovlp(mf2.mo_coeff, mo_ba, mo_occ, mo_occ)
    return det


@row("test_uhf.py:335")
def _():
    d1, d2 = seeded(water_vdz().nao_nr(), 2)
    return numpy.linalg.norm(scf.uhf.get_veff(water_vdz(), (d1 + d1.T, d2 + d2.T)))


@row("test_uhf.py:359")
def _():
    p, dm = cart631_complex_dm()
    mf1 = scf.uhf.UHF(p)
    mf1.direct_scf = True
    mf1.max_memory = 0
    return lib.fp(mf1.get_veff(p, dm, hermi=0))


@row("test_uhf.py:374")
def _():
    mf = natm0_model(scf.UHF, 2)
    mf.kernel()
    return mf.e_tot


@row("test_uhf.py:387")
def _():
    mf = n2_uhf()
    return mf.energy_tot(mf.make_rdm1()), mf.e_tot


R["test_uhf.py:392"] = lambda: scf.UHF(M(atom="He")).run().e_tot
R["test_uhf.py:396"] = lambda: scf.UHF(M(atom="He", charge=2)).run().e_tot


def max_cycle0():
    mf = scf.UHF(water_vdz())
    mf.max_cycle = 0
    mf.kernel(mf.get_init_guess())
    e1 = mf.e_tot
    mf.kernel(mf.make_rdm1())
    return e1, mf.e_tot


R["test_uhf.py:453"] = lambda: max_cycle0()[0]
R["test_uhf.py:457"] = lambda: max_cycle0()[1]


@row("test_uhf.py:465")
def _():
    h1 = scf.hf.get_hcore(n2sym())
    mf = scf.UHF(n2sym())
    mf.get_hcore = lambda *args: (h1, h1)
    return mf.kernel()


@row("test_uhf.py:471")
def _():
    import scipy.linalg
    h1 = scf.hf.get_hcore(n2sym())
    s1 = scf.hf.get_ovlp(n2sym())
    mf = scf.uhf.HF1e(n2sym())
    mf.get_hcore = lambda *args: (h1, h1)
    eref = scipy.linalg.eigh(h1, s1)[0][0] + n2sym().energy_nuc()
    return mf.kernel(), eref


# test_diis.py

@row("test_diis.py:54")
def _():
    mf1 = scf.RHF(M(atom=WATER, basis="631g"))
    mf1.DIIS = diis.EDIIS
    mf1.max_cycle = 4
    return mf1.kernel()


@row("test_diis.py:72")
def _():
    mf1 = scf.RHF(M(atom=WATER_STRETCHED, basis="631g"))
    mf1.diis_space = 4
    mf1.diis_space_rollback = 1
    mf1.max_cycle = 10
    return mf1.kernel()


def diis_restart():
    m = M(atom=WATER_STRETCHED, basis="631g")
    tmpf = tempfile.NamedTemporaryFile()
    mf = scf.RHF(m)
    mf.diis_file = tmpf.name
    eref = mf.kernel()
    mf = scf.RHF(m)
    mf.diis = scf.diis.DIIS().restore(tmpf.name)
    mf.max_cycle = 3
    return eref, mf.kernel()


R["test_diis.py:89"] = lambda: diis_restart()[0]


@row("test_diis.py:95")
def _():
    eref, e = diis_restart()
    return e, eref


# test_addons.py

@row("test_addons.py:217")
def _():
    mf = addons_rhf()
    return addons.follow_state(mf).run().e_tot, mf.e_tot


@row("test_addons.py:221")
def _():
    mf = addons_rhf()
    return addons.follow_state(mf, mf.mo_coeff[:, [0, 1, 2, 3, 5]]).scf()


@row("test_addons.py:261")
def _():
    mf = scf.ROHF(water631()).run()
    mo0 = mf.mo_coeff
    occ = mf.mo_occ
    setocc = numpy.zeros((2, occ.size))
    setocc[:, occ == 2] = 1
    setocc[0][4] = 0
    setocc[0][5] = 1
    newocc = setocc[0][:] + setocc[1][:]
    mf = addons.mom_occ(mf, mo0, setocc)
    mf.kernel(mf.make_rdm1(mo0, newocc))
    return mf.e_tot


@row("test_addons.py:271")
def _():
    mf = addons.dynamic_level_shift(scf.RHF(water631()))
    mf.init_guess = "hcore"
    mf.diis = False
    mf.max_cycle = 4
    mf.kernel()
    return mf.e_tot


@row("test_addons.py:414")
def _():
    m = M(atom=[("H", 0, 0, i * .5) for i in range(4)], basis=("sto-3g", [[0, [.002, 1]]]))
    return addons.remove_linear_dep_(scf.RHF(m), threshold=1e-8, lindep=1e-9).run().e_tot


# test_bz.py

R["test_bz.py:56"] = lambda: scf.RHF(M(atom=BENZENE, basis="ccpvdz")).set(conv_tol=1e-11).scf()


# gto/test/test_mole.py

@row("test_mole.py:520")
def _():
    c0 = gto.mole.cart2sph(0, normalized="sp")
    c1 = gto.mole.cart2sph(0, numpy.eye(1))
    return abs(c0 * 0.282094791773878143 - c1).max()


@row("test_mole.py:529")
def _():
    c0 = gto.mole.cart2sph(1, normalized="sp")
    c1 = gto.mole.cart2sph(1, numpy.eye(3).T)
    return abs(c0 * 0.488602511902919921 - c1).max()


R["test_mole.py:942"] = lambda: M(atom="He").ao2mo(numpy.eye(1))[0, 0]


@row("test_mole.py:980")
def _():
    m = frac_water()
    ref = 0
    for ia in range(m.natm):
        with m.with_rinv_origin(m.atom_coord(ia)):
            ref -= m.intor("int1e_rinv") * m.atom_charge(ia)
    return abs(ref - m.intor("int1e_nuc")).max()


def frac_water():
    m = M(atom=[["h", (0., 1., 1.)], ["O", (0., 0., 0.)], ["h", (1., 1., 0.)]], basis="sto3g")
    m._atm[1, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
    m._env[m._atm[1, gto.PTR_FRAC_CHARGE]] = 2.5
    return m


# gto/test/test_moleintor.py (fp there is abs(mat).sum())

R["test_moleintor.py:96"] = lambda: abs(moleintor_benzene().intor("int1e_ovlp_sph")).sum()
R["test_moleintor.py:100"] = lambda: abs(moleintor_benzene().intor_symmetric("int1e_ovlp_sph")).sum()
R["test_moleintor.py:104"] = lambda: abs(moleintor_benzene().intor_asymmetric("int1e_ovlp_sph")).sum()


@row("test_moleintor.py:109")
def _():
    m = moleintor_benzene()
    shls_slice = (0, m.nbas // 4, m.nbas // 4, m.nbas)
    return abs(m.intor("int1e_ovlp_sph", shls_slice=shls_slice)).sum()


R["test_moleintor.py:303"] = lambda: lib.fp(
    M(atom="He 0 0 0; Ne 3 0 0", basis="ccpvdz").intor("int2e_sph", aosym="s8"))


# ---- the manifest ----

def read_tsv():
    rows = []
    with open(TSV) as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == COLUMNS, f"{TSV}: header is {header}, want {COLUMNS}"
        for n, line in enumerate(f, 2):
            cells = line.rstrip("\n").split("\t")
            assert len(cells) == len(COLUMNS), f"{TSV}:{n}: {len(cells)} columns"
            rows.append(dict(zip(COLUMNS, cells)))
    return rows


def write_tsv(rows):
    with open(TSV, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in COLUMNS) + "\n")


def literal(text):
    try:
        return float(text)
    except ValueError:
        return None


def real(x):
    # lib.fp of a complex result is complex with a zero imaginary part; the
    # tests compare it to a real literal, and so do we, after checking.
    x = complex(x)
    assert abs(x.imag) <= 1e-12 * max(1.0, abs(x.real)), x
    return x.real


def run_row(r):
    """(lhs, rhs, passed) for one manifest row, judged the way its test judges it."""
    out = R[r["loc"]]()
    if isinstance(out, tuple):
        lhs, rhs = real(out[0]), real(out[1])
    else:
        lhs, rhs = real(out), literal(r["asserted"])
        assert rhs is not None, f"{r['loc']}: '{r['asserted']}' is not a literal, so its recipe must compute it"
    if r["tier"] == "T1s":
        return lhs, rhs, lhs < rhs
    places = int(r["places"])
    return lhs, rhs, round(abs(lhs - rhs), places) == 0


def check_manifest(rows):
    locs = [r["loc"] for r in rows]
    assert len(set(locs)) == len(locs), "duplicate loc in the manifest"
    missing = set(locs) - set(R)
    extra = set(R) - set(locs)
    assert not missing, f"rows with no recipe: {sorted(missing)}"
    assert not extra, f"recipes with no row: {sorted(extra)}"
    counts = {t: sum(r["tier"] == t for r in rows) for t in TIERS}
    assert counts == TIER_COUNTS, f"tier counts {counts}, want {TIER_COUNTS}"
    for r in rows:
        assert r["tags"] in ("-", "path"), r["loc"]


def check_sources(rows, src):
    """Each row's asserted literal appears on its line of the v2.5.0 source."""
    bad = 0
    for r in rows:
        name, line = r["loc"].split(":")
        with open(os.path.join(src, SOURCES[name])) as f:
            text = f.read().split("\n")
        # A row's line is the one that holds its literal. For an assertion
        # wrapped over two lines (test_h2o.py:264-265) that is the second, so
        # the assert itself may sit on the line before.
        n = int(line) - 1
        here = text[n]
        want = r["asserted"].replace(" ", "")
        if r["tier"] == "T1s":
            want = "assertTrue"
        if want not in here.replace(" ", "") or "assert" not in text[n - 1] + text[n]:
            print(f"source mismatch {r['loc']}: '{want}' not in: {here.strip()}")
            bad += 1
    print(f"sources: {len(rows) - bad}/{len(rows)} rows match their line")
    return bad == 0


def fmt(x):
    return repr(float(x))


def validate(rows, rewrite):
    ok = 0
    for r in rows:
        lhs, rhs, passed = run_row(r)
        drift = ""
        if not rewrite and r["measured"] != "-":
            # the same machine and PySCF should give the same number to far
            # better than the asserted places; a drift says the environment moved
            d = abs(lhs - float(r["measured"]))
            if d > 1e-10 * (1.0 + abs(lhs)):
                drift = f" drift {d:.1e} from the manifest"
                passed = False
        if rewrite:
            r["measured"], r["target"] = fmt(lhs), fmt(rhs)
        elif r["target"] != "-" and abs(rhs - float(r["target"])) > 1e-10 * (1.0 + abs(rhs)):
            drift += f" target moved to {rhs!r}"
            passed = False
        ok += passed
        tag = "ok  " if passed else "FAIL"
        print(f"{tag} {r['loc']:24s} {r['tier']:6s} lhs={lhs!r} rhs={rhs!r} places={r['places']}{drift}",
              flush=True)
    print(f"validated {ok}/{len(rows)}")
    return ok == len(rows)


# ---- the Viv module ----

FILE_CODE = {"test_h2o.py": 1, "test_h2o_vdz.py": 2, "test_he.py": 3, "test_rhf.py": 4,
             "test_uhf.py": 5, "test_diis.py": 6, "test_addons.py": 7, "test_bz.py": 8,
             "test_mole.py": 9, "test_moleintor.py": 10}
TIER_CODE = {"T1": 1, "T1s": 2, "T1x": 3, "T2": 4, "T3": 5, "T2+T3": 6}


def const_name(loc):
    name, line = loc.split(":")
    return "PST_" + name[len("test_"):-len(".py")].upper() + "_" + line


# The systems the T1, T1x and T3 rows run on, in Bohr as PySCF holds them.
# (name, what, builder). The builder returns a built Mole.
SYSTEMS = [
    ("water", "water, the geometry of test_h2o/test_rhf/test_uhf (O at the origin, H at (0, -+0.757, 0.587) A)",
     lambda: M(atom=WATER, basis="6-31g")),
    ("water_stretched", "stretched water of test_diis.py (H at (0, -+1.757, 1.587) A)",
     lambda: M(atom=WATER_STRETCHED, basis="6-31g")),
    ("he", "a helium atom at the origin", lambda: M(atom="He")),
    ("h", "a hydrogen atom at the origin", lambda: M(atom="H", spin=1)),
    ("n2", "N2 at 1 A along z", lambda: M(atom=N2, basis="cc-pvdz")),
    ("h4", "the H4 chain of test_addons.py:414, 0.5 A spacing along z",
     lambda: M(atom=[("H", 0, 0, i * .5) for i in range(4)], basis=("sto-3g", [[0, [.002, 1]]]))),
    ("h2_schwarz", "the two hydrogens of test_rhf.py:944, 4 Bohr apart",
     lambda: M(atom="H 0 0 0; H 0 0 4.", unit="B", basis=[[0, (2.7, 1)], [0, (1e2, 1)]])),
    ("frac", "the water-like molecule of test_mole.py:980; O carries charge 2.5", frac_water),
]


def pack_atoms(m):
    return [[float(m.atom_charge(i)), *map(float, m.atom_coord(i))] for i in range(m.natm)]


def pack_basis(m):
    """shells[nsh,8] and prims[nprim,2] in the frozen layout, general contractions split."""
    shells, prims = [], []
    cart0 = sph0 = 0
    for ib in range(m.nbas):
        l = m.bas_angular(ib)
        expo = m.bas_exp(ib)
        coef = m.bas_ctr_coeff(ib)
        for ic in range(m.bas_nctr(ib)):
            ncart, nsph = (l + 1) * (l + 2) // 2, 2 * l + 1
            shells.append([m.bas_atom(ib), l, len(expo), len(prims), cart0, sph0, ncart, nsph])
            prims.extend([float(e), float(c)] for e, c in zip(expo, coef[:, ic]))
            cart0 += ncart
            sph0 += nsph
    return shells, prims


def places(r):
    # the Schwarz row asserts an inequality and has no places; 0 stands in
    return r["places"] if r["places"] != "-" else "0"


def viv_list(xs):
    if xs and isinstance(xs[0], list):
        return "[" + ", ".join(viv_list(x) for x in xs) + "]"
    return "[" + ", ".join(fmt(x) if isinstance(x, float) else str(x) for x in xs) + "]"


def emit(rows):
    out = []
    w = out.append
    w("###")
    w("post: The PySCF tests a Viv SCF is held to")
    w("date: 2026-09-23")
    w("")
    w("Generated by `tools/pyscf_suite.py --emit` from `tools/pyscf_suite.tsv`, the")
    w("assertions harvested from PySCF v2.5.0's own unit tests (pyscf/scf/test and")
    w("pyscf/gto/test). Each constant is named for the file and line it came from:")
    w("PST_RHF_406 is the number test_rhf.py asserts on line 406, and PST_RHF_406_P")
    w("the places it asserts it to. A number the test computes instead of writing")
    w("down (\"mf.e_tot\", \"eref\") is given as PySCF 2.5.0 computed it here.")
    w("")
    w("The rows are grouped by what a Viv code needs before it can reproduce them:")
    w("T1 needs s and p functions and RHF, T1x one more SCF-driver feature, T2 d")
    w("functions, T3 UHF or ROHF, T2+T3 both. The Schwarz row (T1s) asserts an")
    w("inequality, not a value. A row flagged `path` depends on how the SCF got")
    w("there -- the guess, the DIIS variant, how many cycles -- and is scored apart.")
    w("")
    w("Do not edit this file.")
    w("###")
    w("")
    w("# ---- tier and file codes used by the row tables ----")
    for t in TIERS:
        w(f"PST_TIER_{t.replace('+', '_').upper()} = {TIER_CODE[t]}")
    for t in TIERS:
        w(f"PST_N_{t.replace('+', '_').upper()} = {TIER_COUNTS[t]}")
    w(f"PST_N_ROWS = {len(rows)}")
    w(f"PST_N_PATH = {sum(r['tags'] == 'path' for r in rows)}")
    for name, code in FILE_CODE.items():
        w(f"PST_FILE_{name[len('test_'):-len('.py')].upper()} = {code}")
    w("")
    for t in TIERS:
        tier_rows = [r for r in rows if r["tier"] == t]
        w(f"# ---- {t}: {len(tier_rows)} rows ----")
        for r in tier_rows:
            path = "  [path]" if r["tags"] == "path" else ""
            w(f"# {r['loc']} {r['test']}: {r['system']}; {r['method']}; asserted {r['asserted']}{path}")
            w(f"{const_name(r['loc'])} = {fmt(float(r['target']))}")
            w(f"{const_name(r['loc'])}_P = {places(r)}")
        w("")
    w("#| One row per assertion: tier code, file code, line, places, target, path flag.")
    w("#| Split by tier so no literal passes the 512-value budget.")
    for t in TIERS:
        tier_rows = [r for r in rows if r["tier"] == t]
        table = []
        for r in tier_rows:
            name, line = r["loc"].split(":")
            table.append(f"[{TIER_CODE[t]}, {FILE_CODE[name]}, {line}, {places(r)}, "
                         f"{fmt(float(r['target']))}, {1 if r['tags'] == 'path' else 0}]")
        w(f"def pst_rows_{t.replace('+', '_').lower()}() {{")
        w("    return [" + ",\n        ".join(table) + "]")
        w("}")
    w("")
    w("# ---- the constants test_mole.py:520/529 multiply by: cart2sph's s and p factors ----")
    w(f"PST_C2S_S = {fmt(0.282094791773878143)}")
    w(f"PST_C2S_P = {fmt(0.488602511902919921)}")
    w("")
    w("# ---- the model of test_rhf.py:757 and test_uhf.py:374: no atoms, three orbitals,")
    w("# h = diag(0, 1, 2), S = I, (ii|ii) = 0.2 and every other integral zero, 2 electrons ----")
    w("PST_MODEL_N = 3")
    w("PST_MODEL_NELEC = 2")
    w("PST_MODEL_U = 0.2")
    w("def pst_model_h() {")
    w("    return [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]")
    w("}")
    w("")
    w("# ---- the Schwarz case, test_rhf.py:944: (0 2|3 3) against Q_02 Q_33 ----")
    schwarz = next(r for r in rows if r["tier"] == "T1s")
    w(f"PST_SCHWARZ_ERI = {fmt(float(schwarz['measured']))}")
    w(f"PST_SCHWARZ_BOUND = {fmt(float(schwarz['target']))}")
    w("")
    w("# ---- systems: atoms[natm,4] = (Z, x, y, z) in Bohr ----")
    for name, what, build in SYSTEMS:
        m = build()
        w(f"#| {what}")
        w(f"def pst_{name}_atoms() {{")
        w(f"    return {viv_list(pack_atoms(m))}")
        w("}")
    w("")
    w("# ---- the two custom basis sets, in the frozen shells/prims layout:")
    w("# shells = (atom, l, nprim, p0, cart0, sph0, ncart, nsph),")
    w("# prims = (exponent, PySCF bas_ctr_coeff) ----")
    for name, what, build in SYSTEMS:
        if name not in ("h4", "h2_schwarz"):
            continue
        shells, prims = pack_basis(build())
        w(f"def pst_{name}_shells() {{")
        w(f"    return {viv_list(shells)}")
        w("}")
        w(f"def pst_{name}_prims() {{")
        w(f"    return {viv_list(prims)}")
        w("}")
    w("")
    w(VIV_TESTS)
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(VIV), exist_ok=True)
    with open(VIV, "w") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(VIV, ROOT)}: {len(rows)} rows")


VIV_TESTS = r'''
test "the manifest has the tier counts the plan fixes" {
    pst_n1 = len(pst_rows_t1())
    pst_n1s = len(pst_rows_t1s())
    pst_n1x = len(pst_rows_t1x())
    pst_n2 = len(pst_rows_t2())
    pst_n3 = len(pst_rows_t3())
    pst_n23 = len(pst_rows_t2_t3())
    assert_near(pst_n1, PST_N_T1, 0.0)
    assert_near(pst_n1s, PST_N_T1S, 0.0)
    assert_near(pst_n1x, PST_N_T1X, 0.0)
    assert_near(pst_n2, PST_N_T2, 0.0)
    assert_near(pst_n3, PST_N_T3, 0.0)
    assert_near(pst_n23, PST_N_T2_T3, 0.0)
    assert_near(pst_n1 + pst_n1s + pst_n1x + pst_n2 + pst_n3 + pst_n23, PST_N_ROWS, 0.0)
    # every row sits in the table of its own tier, and its places are what unittest can use
    pst_t = pst_rows_t2_t3()
    for pst_i to len(pst_t) {
        assert_near(pst_t[pst_i, 0], PST_TIER_T2_T3, 0.0)
        assert_lt(0.0, pst_t[pst_i, 3])
        assert_lt(pst_t[pst_i, 3], 15.0)
    }
    # the path rows are the EDIIS, rollback and DIIS-restore rows, the dynamic
    # level shift and the two max_cycle=0 steps; the rest ask only for a state
    pst_npath = 0
    pst_x = pst_rows_t1x()
    for pst_i to len(pst_x) {
        pst_npath = pst_npath + pst_x[pst_i, 5]
    }
    pst_y = pst_rows_t2_t3()
    for pst_i to len(pst_y) {
        pst_npath = pst_npath + pst_y[pst_i, 5]
    }
    assert_near(pst_npath, PST_N_PATH, 0.0)
}

test "water's nuclear repulsion from the pack is test_h2o.py:156" {
    # sum over pairs of Z_A Z_B / R_AB, from the Bohr coordinates; the test asks for 10 places
    pst_a = pst_water_atoms()
    pst_enuc = 0.0
    for pst_i to len(pst_a) {
        for pst_j to pst_i {
            pst_dx = pst_a[pst_i, 1] - pst_a[pst_j, 1]
            pst_dy = pst_a[pst_i, 2] - pst_a[pst_j, 2]
            pst_dz = pst_a[pst_i, 3] - pst_a[pst_j, 3]
            pst_enuc = pst_enuc + pst_a[pst_i, 0] * pst_a[pst_j, 0] / sqrt(pst_dx * pst_dx + pst_dy * pst_dy + pst_dz * pst_dz)
        }
    }
    assert_near(pst_enuc, PST_H2O_156, 0.5e-10)
}

test "the cart2sph factors of test_mole.py:520/529 are 1/(2 sqrt(pi)) and sqrt(3/(4 pi))" {
    # a unit s function is 1/sqrt(4 pi) times a unit Cartesian one, and a unit p
    # function sqrt(3/(4 pi)) times x/r; those are the real solid-harmonic norms
    pst_pi = 4.0 * atan(1.0)
    assert_near(PST_C2S_S, 1.0 / (2.0 * sqrt(pst_pi)), 1.0e-16)
    assert_near(PST_C2S_P, sqrt(3.0 / (4.0 * pst_pi)), 1.0e-16)
    assert_near(PST_MOLE_520, 0.0, 1.0e-15)
    assert_near(PST_MOLE_529, 0.0, 1.0e-15)
}

test "the no-atom model's energies follow from h and (ii|ii) alone" {
    # RHF puts both electrons in orbital 0: E = 2 h00 + (00|00).
    # UHF with S = 1 puts one in orbital 0 and one in 1: E = h00 + h11 + (00|11) - (01|10),
    # and both of those integrals are zero in this model.
    pst_h = pst_model_h()
    assert_near(2.0 * pst_h[0, 0] + PST_MODEL_U, PST_RHF_757, 1.0e-15)
    assert_near(pst_h[0, 0] + pst_h[1, 1], PST_UHF_374, 1.0e-15)
}

test "closed-shell UHF and RHF targets agree, and bare nuclei have zero energy" {
    assert_near(PST_H2O_80, PST_H2O_55, 1.0e-12)
    assert_near(PST_HE_57, PST_HE_52, 1.0e-12)
    assert_near(PST_UHF_392, PST_RHF_636, 1.0e-12)
    assert_near(PST_UHF_143, PST_RHF_356, 1.0e-12)
    assert_near(PST_UHF_207, PST_RHF_516, 1.0e-12)
    assert_near(PST_UHF_175, PST_RHF_406, 1.0e-12)
    assert_near(PST_UHF_106, PST_RHF_261, 1.0e-12)
    assert_near(PST_UHF_109, PST_RHF_264, 1.0e-12)
    assert_near(PST_RHF_640, 0.0, 0.0)
    assert_near(PST_UHF_396, 0.0, 0.0)
}

test "orderings and identities among the open-shell and DIIS targets" {
    # UHF has more freedom than ROHF, so for the same triplet it is at or below it
    assert_lt(PST_UHF_155, PST_RHF_386)
    # the ROHF 1e guess splits the RHF density evenly between the two spins
    assert_near(PST_H2O_265 * sqrt(2.0), PST_H2O_255, 1.0e-14)
    assert_near(PST_H2O_270, PST_H2O_265, 0.0)
    # an EDIIS run cut off at four cycles sits above the converged energy
    assert_lt(PST_H2O_55, PST_DIIS_54)
    # the MOM triplet and the spin = -2 ROHF are the same state reached two ways
    assert_near(PST_ADDONS_261, PST_RHF_386, 1.0e-9)
    # a restored DIIS lands on the same stretched-water state as the first run
    assert_near(PST_DIIS_95, PST_DIIS_89, 1.0e-9)
    # That default state is a saddle point: an internal instability leads to a
    # lower RHF state near -75.4739. So the unconverged rollback energy can sit
    # below it without breaking the variational principle, and it does.
    assert_lt(PST_DIIS_72, PST_DIIS_89)
}

test "the Schwarz row: the integral sits under its bound" {
    assert_lt(0.0, PST_SCHWARZ_ERI)
    assert_lt(PST_SCHWARZ_ERI, PST_SCHWARZ_BOUND)
}

test "the custom basis packs are self-consistent" {
    # H4: four STO-3G hydrogens (3 primitives) and four diffuse s shells of one primitive
    pst_s = pst_h4_shells()
    pst_p = pst_h4_prims()
    pst_np = 0
    for pst_i to len(pst_s) {
        assert_near(pst_s[pst_i, 3], pst_np, 0.0)
        assert_near(pst_s[pst_i, 4], pst_i, 0.0)
        pst_np = pst_np + pst_s[pst_i, 2]
    }
    assert_near(pst_np, len(pst_p), 0.0)
    assert_near(len(pst_s), 8.0, 0.0)
    assert_near(pst_p[len(pst_p) - 1, 0], 0.002, 0.0)
    pst_q = pst_h2_schwarz_shells()
    assert_near(len(pst_q), 4.0, 0.0)
    pst_a = pst_h2_schwarz_atoms()
    assert_near(pst_a[1, 3], 4.0, 0.0)
}'''


def main(argv):
    rows = read_tsv()
    check_manifest(rows)
    good = True
    if "--src" in argv:
        good &= check_sources(rows, argv[argv.index("--src") + 1])
    if "--measure" in argv:
        good &= validate(rows, rewrite=True)
        write_tsv(rows)
    elif "--validate" in argv:
        good &= validate(rows, rewrite=False)
    if "--emit" in argv:
        emit(rows)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

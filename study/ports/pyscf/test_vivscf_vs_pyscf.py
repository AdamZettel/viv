# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivscf.viv against PySCF 2.5.0, for every case of port.toml.

    python3 -m pytest study/ports/pyscf/test_vivscf_vs_pyscf.py          # all cases, PySCF run live
    VIVSCF_CASES=water_sto3g,watercat_631g_uhf python3 -m pytest ...      # some cases
    VIVSCF_REFERENCE=oracle python3 -m pytest ...                         # oracle/<case>.json, no PySCF
    python3 study/ports/pyscf/test_vivscf_vs_pyscf.py [--oracle] [case ...]   # the table, no pytest

The Viv side is one callable, `run(**params) -> dict`, the signature of the
wrapper `viv export study/ports/pyscf/vivscf.export.toml` generates:
keyword parameters in the manifest's
names (atoms, coords, basis, charge, spin, method, jk, e_tol, grad_tol,
max_iter, diis_size, diis_start), and a dict with e_tot, e_nuc, mo_energy,
mo_coeff, dm, converged, iterations and tests.

    VIVSCF_BACKEND=native   (default) vivscf_native.run: the program built with
                            `viv build`, its parameter lines rewritten (the
                            adapter's docstring says why not --set)
    VIVSCF_BACKEND=export   `import vivscf` from VIVSCF_DIST (default
                            target/export/vivscf, where `viv export` writes
                            without -o): the exported library

Each case is held to port.toml's tolerances: the total energy within
energy_tol (1e-10 Eh) of PySCF's, every orbital energy within mo_energy_tol
(1e-8 Eh), both runs converged at the verification settings (PySCF
conv_tol 1e-12, conv_tol_grad 1e-8; Viv e_tol 1e-12, grad_tol 1e-9). Also:
the nuclear repulsion within 1e-10, the program's own tests passing, and,
against a live PySCF, the density matrix within DM_TOL of PySCF's and the
orbitals orthonormal in PySCF's overlap matrix (which checks that the AOs
come in PySCF's order).

PySCF runs one case at a time with one thread; the cc-pVTZ cases run last.
"""
import json
import os
import sys
import time
import tomllib

# one thread for PySCF's NumPy, set before NumPy loads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

CONF = tomllib.load(open(os.path.join(HERE, "port.toml"), "rb"))
VERIFY = CONF["verify"]

# not a port.toml tolerance: the density is first order in the orbital
# gradient, which conv_tol_grad 1e-8 bounds; 1e-6 leaves two orders
DM_TOL = 1.0e-6
ORTHO_TOL = 1.0e-8
ENUC_TOL = 1.0e-10

# atomic numbers of the symbols port.toml's geometries use (lib/xyz has H to Ar)
ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]


def atoms_text(c):
    """The case's molecule as PySCF's atom text (pyscf_side.atoms_of)."""
    import pyscf_side
    return pyscf_side.atoms_of(c)


def molecule(c):
    """Atomic numbers and coordinates (angstrom) of a case."""
    z, xyz = [], []
    for part in atoms_text(c).split(";"):
        sym, x, y, w = part.split()
        z.append(ELEMENTS.index(sym) + 1)
        xyz.append([float(x), float(y), float(w)])
    return z, xyz


def params(c):
    """The inputs of a case at the verification settings."""
    z, xyz = molecule(c)
    return dict(atoms=z, coords=xyz, basis=c["basis"], charge=c.get("charge", 0), spin=c.get("spin", 0),
                method=c["method"], jk="direct" if c.get("jk") == "direct" else "incore",
                e_tol=VERIFY["viv_e_tol"], grad_tol=VERIFY["viv_grad_tol"])


def case_order():
    """Smallest first by the oracle's AO count, the cc-pVTZ cases last."""
    def nao(c):
        p = os.path.join(HERE, "oracle", c["name"] + ".json")
        return json.load(open(p))["nao"] if os.path.exists(p) else 0
    cases = list(CONF["case"])
    want = os.environ.get("VIVSCF_CASES")
    if want:
        names = want.split(",")
        cases = [c for c in cases if c["name"] in names]
    return sorted(cases, key=lambda c: (c["basis"] == "cc-pvtz", nao(c)))


def backend():
    which = os.environ.get("VIVSCF_BACKEND", "native")
    if which == "export":
        sys.path.insert(0, os.environ.get("VIVSCF_DIST", os.path.join(REPO, "target", "export", "vivscf")))
        import vivscf
        return vivscf.run
    import vivscf_native
    return vivscf_native.run


def reference_oracle(c):
    o = json.load(open(os.path.join(HERE, "oracle", c["name"] + ".json")))
    if c["method"] == "uhf":
        moe = numpy.array([[float(x) for x in o["mo_energy_alpha"]], [float(x) for x in o["mo_energy_beta"]]])
    else:
        moe = numpy.array([float(x) for x in o["mo_energy"]])
    return {"e_tot": float(o["e_tot"]), "e_nuc": float(o["enuc"]), "mo_energy": moe, "converged": True,
            "cycles": o["cycles"], "source": "oracle"}


def reference_pyscf(c):
    """PySCF 2.5.0 at the verification settings (pyscf_side.new_mf)."""
    import pyscf
    from pyscf import lib
    import pyscf_side
    assert pyscf.__version__ == CONF["original"]["version"], pyscf.__version__
    lib.num_threads(1)
    mol = pyscf_side.build(c)
    mf, e, cycles = pyscf_side.run(mol, c, VERIFY["pyscf_conv_tol"], VERIFY["pyscf_conv_tol_grad"])
    return {"e_tot": float(e), "e_nuc": float(mol.energy_nuc()), "mo_energy": numpy.asarray(mf.mo_energy),
            "dm": numpy.asarray(mf.make_rdm1()), "s": mol.intor("int1e_ovlp"), "converged": bool(mf.converged),
            "cycles": cycles, "source": "PySCF " + pyscf.__version__}


def reference(c):
    if os.environ.get("VIVSCF_REFERENCE", "pyscf") == "oracle":
        return reference_oracle(c)
    return reference_pyscf(c)


def compare(c, v, ref):
    """The gaps between Viv's run `v` and the reference, and what failed."""
    nao = v.get("nao") or numpy.asarray(v["mo_coeff"]).shape[-2]
    row = {"case": c["name"], "e_viv": v["e_tot"], "wall": v.get("wall"), "nao": nao, "iterations": v["iterations"], "cycles": ref["cycles"],
           "de": abs(v["e_tot"] - ref["e_tot"]), "denuc": abs(v["e_nuc"] - ref["e_nuc"]),
           "dmo": float(numpy.max(numpy.abs(numpy.asarray(v["mo_energy"]) - ref["mo_energy"])))}
    bad = []
    if not v["converged"] or not ref["converged"]:
        bad.append("not converged")
    if row["de"] >= VERIFY["energy_tol"]:
        bad.append(f"|dE| {row['de']:.1e} >= {VERIFY['energy_tol']:g}")
    if row["dmo"] >= VERIFY["mo_energy_tol"]:
        bad.append(f"max |d eps| {row['dmo']:.1e} >= {VERIFY['mo_energy_tol']:g}")
    if row["denuc"] >= ENUC_TOL:
        bad.append(f"|d E_nuc| {row['denuc']:.1e}")
    # (name, passed) from the native adapter, (name, passed, detail) from the export
    failed = [t[0] for t in v.get("tests", []) if not t[1]]
    if failed:
        bad.append("Viv tests failed: " + "; ".join(failed))
    if "dm" in ref:
        row["ddm"] = float(numpy.max(numpy.abs(numpy.asarray(v["dm"]) - ref["dm"])))
        C = numpy.asarray(v["mo_coeff"])
        Cs = C if C.ndim == 3 else C[None]
        row["dortho"] = max(float(numpy.max(numpy.abs(Ck.T @ ref["s"] @ Ck - numpy.eye(Ck.shape[1])))) for Ck in Cs)
        if row["ddm"] >= DM_TOL:
            bad.append(f"max |d D| {row['ddm']:.1e} >= {DM_TOL:g}")
        if row["dortho"] >= ORTHO_TOL:
            bad.append(f"C^T S C - 1: {row['dortho']:.1e} (AO order?)")
    return row, bad


def check_case(c, run):
    t0 = time.perf_counter()
    v = run(**params(c))
    v.setdefault("wall", time.perf_counter() - t0)
    ref = reference(c)
    return compare(c, v, ref)


# ---- pytest -----------------------------------------------------------------------

try:
    import pytest

    @pytest.fixture(scope="module")
    def run():
        return backend()

    @pytest.mark.parametrize("case", case_order(), ids=lambda c: c["name"])
    def test_case_matches_pyscf(case, run):
        row, bad = check_case(case, run)
        assert not bad, f"{case['name']}: " + ", ".join(bad) + f"  {row}"
except ImportError:  # pragma: no cover
    pass


# ---- the table ----------------------------------------------------------------------

def main(argv):
    if "--oracle" in argv:
        os.environ["VIVSCF_REFERENCE"] = "oracle"
    names = [a for a in argv if not a.startswith("--")]
    if names:
        os.environ["VIVSCF_CASES"] = ",".join(names)
    run = backend()
    nbad = 0
    rows = []
    for c in case_order():
        row, bad = check_case(c, run)
        nbad += bool(bad)
        rows.append(row)
        extra = ""
        if "ddm" in row:
            extra = f"  max|dD| {row['ddm']:.1e}  C^T S C {row['dortho']:.1e}"
        print(f"{c['name']:22s} {row['nao']:4d} AOs  |dE| {row['de']:.1e}  max|d eps| {row['dmo']:.1e}  "
              f"|dE_nuc| {row['denuc']:.1e}{extra}  it {row['iterations']} (ref {row['cycles']})  {row['wall']:.2f} s  E {row['e_viv']!r}"
              + ("" if not bad else "  FAIL: " + ", ".join(bad)), flush=True)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

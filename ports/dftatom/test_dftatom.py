# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The port of dftatom against upstream's Fortran and against NIST.

    python3 ports/dftatom/test_dftatom.py            (or python3 -m pytest ports/dftatom/test_dftatom.py)
    python3 ports/dftatom/test_dftatom.py --table    the tables of COMPARISON.md, with timings

It loads ports/dftatom/atom.viv through vipyr (python/vipyr), which
exports it by atom.export.toml beside it and builds the library, and calls
`run(**params)` for every case of oracle/upstream_reference.json. $VIV names
the compiler (default target/release/viv).

References:
  - upstream: dftatom's Fortran at e49b3042 built with gfortran -O2
    -fno-fast-math -ffp-contract=off, run by oracle/oracle.f90 on the same
    parameters (oracle/make_reference.py wrote the file), linked with Viv's
    libm (oracle/vmath_shim.c) so both call the same exp, log, pow and atan;
  - upstream_libm: the same build with the platform's libm;
  - upstream's own converged energies, tests/lda/nonrel_energies.dat and
    tests/rlda/rel_energies.dat, which its conv_lda and conv_rlda tests hold
    to 1.2e-8 Hartree;
  - NIST Standard Reference Database 141 (oracle/nist_reference.json), LDA
    and RLDA columns, printed to 1e-6 Hartree.

Tolerances, and why:
  - none against upstream: with the same libm the port does upstream's
    arithmetic in upstream's order, so every energy, eigenvalue, density
    sample and iteration count must be the same double.
  - the case's mixing_eps against upstream_libm, on E_tot and every
    eigenvalue. The platform's exp is off by an ulp on some inputs where
    Viv's is correctly rounded; the SCF loop stops once no eigenvalue moves
    more than mixing_eps, so two runs that differ in rounding can stop that
    far apart and no closer is promised. (The components are not variational
    and are reported, not held, against this build.)
  - TOL_CONVERGED = 1.2e-8 Hartree against upstream's converged tables: the
    tolerance upstream's conv_lda and conv_rlda use for its 1e-8 meshes.
  - against NIST, upstream's own statement of NIST's accuracy (README.rst,
    "Accuracy"): 1e-6 Hartree in total energies (TOL_NIST_ETOT) and 2e-6 in
    eigenvalues (TOL_NIST_EIG). NIST prints six decimals. The energy
    components are reported, not held: NIST states no accuracy for them,
    and for heavy atoms upstream's and NIST's differ by up to 6e-5
    (uranium, RLDA) while the totals agree to 5e-7.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("VIV", os.path.join(ROOT, "target", "release", "viv"))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools", "export"))
import vipyr  # noqa: E402
import check  # noqa: E402  (tools/export/check.py: result_hash)

TOL_CONVERGED = 1.2e-8
TOL_NIST_ETOT = 1e-6
TOL_NIST_EIG = 2e-6

REF = json.load(open(os.path.join(HERE, "oracle", "upstream_reference.json")))
NIST = json.load(open(os.path.join(HERE, "oracle", "nist_reference.json")))
HASHES = os.path.join(HERE, "dftatom.exact_hashes.json")
COMPONENTS = ["etot", "ekin", "ecoul", "eenuc", "exc"]

_mod = None
_runs = {}


def atom():
    global _mod
    if _mod is None:
        _mod = vipyr.load(os.path.join(HERE, "atom.viv"))
    return _mod


def run(name):
    if name not in _runs:
        _runs[name] = atom().run(**REF["cases"][name]["params"])
    return _runs[name]


def orbitals(r):
    """{(n, l, kappa): eigenvalue} of a Viv run."""
    return {(int(n), int(l), int(k)): e for n, l, k, e in
            zip(r["orbital_n"], r["orbital_l"], r["orbital_kappa"], r["orbital_energies"])}


def same_as(name, key):
    """How many of the results differ from the Fortran's `key` run, and the
    largest differences: energies, E_tot, eigenvalues, density (relative)."""
    r, u = run(name), REF["cases"][name][key]
    de = max(abs(r[k] - u[k]) for k in COMPONENTS)
    dtot = abs(r["etot"] - u["etot"])
    mine = orbitals(r)
    deig = max(abs(mine[(n, l, k)] - e) for n, l, k, f, e in u["orbitals"])
    dens = max(abs(r["density"][i] - d) / max(abs(d), 1e-300) for i, _, d in u["density"])
    ndiff = sum(r[k] != u[k] for k in COMPONENTS) + sum(mine[(n, l, k)] != e for n, l, k, f, e in u["orbitals"]) \
        + sum(r["density"][i] != d for i, _, d in u["density"]) + (int(r["scf_iterations"]) != u["iterations"])
    return ndiff, de, dtot, deig, dens


def diffs_nist(name):
    """|Viv - NIST| for each component, and per eigenvalue {(n, l, kappa): diff}."""
    r = run(name)
    p = REF["cases"][name]["params"]
    nist = NIST["atoms"][str(p["z"])]["rlda" if p["relativistic"] else "lda"]
    de = {k: abs(r[k] - nist[k]) for k in COMPONENTS}
    mine = orbitals(r)
    deig = {(n, l, k): abs(mine[(n, l, k)] - e) for n, l, k, e in nist["orbitals"]}
    return de, deig


# ---------------------------------------------------------------- the tests

def test_every_case_equals_upstream():
    """Bit for bit, with the same libm: energies, eigenvalues, the density
    at every 500th point, the iteration count."""
    for name in REF["cases"]:
        ndiff, de, dtot, deig, dens = same_as(name, "upstream")
        assert ndiff == 0, "%s: %d results differ from upstream (largest energy difference %.1e)" % (name, ndiff, de)
        assert run(name)["scf_converged"]


def test_platform_libm_within_scf_tolerance():
    for name, c in REF["cases"].items():
        ndiff, de, dtot, deig, dens = same_as(name, "upstream_libm")
        eps = c["params"]["mixing_eps"]
        assert dtot <= eps, "%s: E_tot differs from upstream (platform libm) by %.2e > %.1e" % (name, dtot, eps)
        assert deig <= eps, "%s: an eigenvalue differs from upstream (platform libm) by %.2e > %.1e" % (name, deig, eps)


def test_upstream_converged_tables():
    """Upstream's conv_lda / conv_rlda check: E_tot and every eigenvalue
    within 1.2e-8 of tests/lda/nonrel_energies.dat and rlda/rel_energies.dat."""
    for kind in ("lda", "rlda"):
        for z, (etot, eig) in REF["upstream_converged"][kind].items():
            name = "%s_%s" % (kind, z)
            if name not in REF["cases"]:
                continue
            r = run(name)
            assert abs(r["etot"] - etot) <= TOL_CONVERGED, (name, r["etot"], etot)
            assert np.max(np.abs(np.asarray(r["orbital_energies"]) - eig)) <= TOL_CONVERGED, name


def test_nist():
    for name, c in REF["cases"].items():
        if not (name.startswith("lda_") or name.startswith("rlda_")):
            continue
        de, deig = diffs_nist(name)
        assert de["etot"] <= TOL_NIST_ETOT, "%s: E_tot differs from NIST by %.2e" % (name, de["etot"])
        for key, d in deig.items():
            assert d <= TOL_NIST_EIG, "%s: eigenvalue %s differs from NIST by %.2e" % (name, key, d)


def test_charge_and_sum():
    for name in REF["cases"]:
        r = run(name)
        z = REF["cases"][name]["params"]["z"]
        assert abs(r["charge"] - z) <= 1e-6 * z, (name, r["charge"])
        assert abs(r["ekin"] + r["ecoul"] + r["eenuc"] + r["exc"] - r["etot"]) <= 1e-9 * abs(r["etot"])


def test_exact_hashes():
    """The bits of the exported cases, as tools/export/check.py recorded them
    from the library, the native program and the interpreter."""
    if not os.path.exists(HASHES):
        return
    want = json.load(open(HASHES))
    for name, h in want.items():
        assert check.result_hash(run(name)) == h, name


# ------------------------------------------------------------------- tables

def best_of(f, k=3):
    t = []
    for _ in range(k):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


def table():
    print("| case | Z | rel | N | SCF iterations | results not bit-identical to upstream (Viv libm) | platform libm: max |dE_tot| | max |de| | max |d part| |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, c in REF["cases"].items():
        p = c["params"]
        nd, _, _, _, _ = same_as(name, "upstream")
        _, de, dtot, deig, _ = same_as(name, "upstream_libm")
        print("| %s | %d | %s | %d | %d | %d of %d | %.1e | %.1e | %.1e |" % (
            name, p["z"], "yes" if p["relativistic"] else "no", p["n"], run(name)["scf_iterations"], nd,
            len(COMPONENTS) + len(c["upstream"]["orbitals"]) + len(c["upstream"]["density"]) + 1, dtot, deig, de))
    print()
    print("| atom | method | E_tot NIST | E_tot Viv | |dE_tot| | max |d eigenvalue| | max |d part| |")
    print("|---|---|---|---|---|---|---|")
    for name, c in REF["cases"].items():
        if not (name.startswith("lda_") or name.startswith("rlda_")):
            continue
        p = c["params"]
        kind = "rlda" if p["relativistic"] else "lda"
        nist = NIST["atoms"][str(p["z"])]
        de, deig = diffs_nist(name)
        worst = max(deig, key=deig.get)
        print("| %s (%d) | %s | %.6f | %.10f | %.1e | %.1e (%d%s%s) | %.1e |" % (
            nist["symbol"], p["z"], kind.upper(), nist[kind]["etot"], run(name)["etot"], de["etot"],
            deig[worst], worst[0], "spdf"[worst[1]], "" if not p["relativistic"] else " k=%d" % worst[2],
            max(de[k] for k in COMPONENTS[1:])))
    print()
    # Timings: each engine best of 5, interleaved so the machine's load falls
    # on all of them alike. The upstream binaries are timed live when
    # $DFTATOM_CHECKOUT names the checkout oracle/make_reference.py used.
    checkout = os.environ.get("DFTATOM_CHECKOUT")
    fast_mod = vipyr.load(os.path.join(HERE, "atom.viv"), fast=True)
    print("| case | upstream, -O3 -march=native -ffast-math (s) | upstream, -O2 strict (s) | Viv library, exact (s) | Viv library, fast (s) | Viv exact / upstream -O3 |")
    print("|---|---|---|---|---|---|")
    for name in ["default", "neon_rel", "lda_54", "rlda_54", "lda_92", "rlda_92"]:
        p = REF["cases"][name]["params"]
        t = {"uf": [], "us": [], "ve": [], "vf": []}
        for _ in range(5):
            if checkout:
                sys.path.insert(0, os.path.join(HERE, "oracle"))
                import make_reference
                t["uf"].append(make_reference.run(os.path.join(checkout, "oracle_fast"), p)["seconds"])
                t["us"].append(make_reference.run(os.path.join(checkout, "oracle_strict"), p)["seconds"])
            t["ve"].append(best_of(lambda: atom().run(**p), 1))
            t["vf"].append(best_of(lambda: fast_mod.run(**p), 1))
        u = REF["cases"][name]["upstream"]
        uf = min(t["uf"]) if t["uf"] else u["seconds_fast"]
        us = min(t["us"]) if t["us"] else u["seconds_strict"]
        print("| %s | %.3f | %.3f | %.3f | %.3f | %.2f |" % (name, uf, us, min(t["ve"]), min(t["vf"]), min(t["ve"]) / uf))


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
        sys.exit(0)
    fails = 0
    for name, f in sorted(globals().items()):
        if name.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, "--", e)
    sys.exit(1 if fails else 0)

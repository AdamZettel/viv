# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write lib/basis/<name>.viv: per-element basis-set tables, H through Ar, from
the Basis Set Exchange, for lib/hf/molecule to build molecules from.

    python3 tools/basis_tables.py            write every table
    python3 tools/basis_tables.py 631g       write one table
    python3 tools/basis_tables.py --check    compare the committed tables with
                                             a fresh generation, and every
                                             element with PySCF 2.5.0's basis

It needs the `basis_set_exchange` package, version 0.12 (the data ship inside
the package, so no network is used), and PySCF 2.5.0 for the comparison in
--check. Running it twice gives the same bytes: every float is written with
repr(), which round-trips a double.

**Which data.** Each set is read as `bse.get_basis(name, elements,
version="0", optimize_general=True)`. Version 0 is the data the EMSL basis
set library distributed, which is what PySCF 2.5.0 ships: for all eight sets
and all eighteen elements the exponents are equal to PySCF's to the last bit
and the normalized contraction coefficients within 2.3e-16. BSE's version 1
revises several sets in the 7th or 8th digit (6-31G's coefficients move by up
to 3e-6), which would move the lock ladder's energies by more than 1e-9.
`optimize_general=True` removes a primitive from a general contraction when
the same exponent also stands alone as an uncontracted shell, which is how
Dunning's sets are distributed and how PySCF has them.

**The table.** Every shell is segmented: an SP shell becomes an s shell and a
p shell, and a general contraction becomes one shell per contraction.
Primitives whose coefficient is exactly zero are dropped, as
tools/hf_reference.py does. A shell with one primitive gets the coefficient
1.0 (its sign and scale are a convention: lib/hf normalizes every shell, and
PySCF writes +1). Other coefficients are the raw values of the data, not
PySCF's `bas_ctr_coeff`, which is the same contraction rescaled to unit norm;
lib/hf's `bas_contract` normalizes every shell, so the scale does not reach
any integral.

**The order.** Within an element, shells are sorted by angular momentum, then
by the number of primitives (more first), then by the largest exponent
(tighter first). That puts core before valence before diffuse, and it is
PySCF's order for every element and set except eight, listed in
PYSCF_ORDER_DIFFERS below, where PySCF keeps the order of its source file
among uncontracted shells of one l. There the AOs are the same functions in
another order, and energies do not depend on AO order; --check says so for
each of the eight and fails on anything else.
"""
import os
import sys

import basis_set_exchange as bse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "lib", "basis")

BSE_VERSION = "0.12"
DATA_VERSION = "0"
ZMAX = 18
SYMBOLS = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
           "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]

# (module and def suffix, BSE name, the name as the site writes it)
SETS = [
    ("sto3g", "sto-3g", "STO-3G"),
    ("631g", "6-31g", "6-31G"),
    ("631gs", "6-31g*", "6-31G*"),
    ("631gss", "6-31g**", "6-31G**"),
    ("ccpvdz", "cc-pvdz", "cc-pVDZ"),
    ("ccpvtz", "cc-pvtz", "cc-pVTZ"),
    ("def2svp", "def2-svp", "def2-SVP"),
    ("def2tzvp", "def2-tzvp", "def2-TZVP"),
]

# (set, Z): PySCF keeps its source file's order among uncontracted shells of
# one angular momentum, and it differs from the rule here. Same shells, other
# AO order.
PYSCF_ORDER_DIFFERS = {
    ("cc-pvtz", 12), ("cc-pvtz", 13), ("cc-pvtz", 14), ("cc-pvtz", 15),
    ("cc-pvtz", 16), ("cc-pvtz", 18),
    ("def2-tzvp", 3), ("def2-tzvp", 11),
}

LETTERS = "spdfgh"


def element_shells(name, z):
    """The segmented shells of element z: a list of (l, exponents, coefficients)."""
    data = bse.get_basis(name, elements=[z], version=DATA_VERSION, optimize_general=True)
    shells = []
    for sh in data["elements"][str(z)]["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        am = sh["angular_momentum"]
        cols = [[float(x) for x in c] for c in sh["coefficients"]]
        # an SP shell has one column per l; a general contraction, one per contraction
        pairs = list(zip(am, cols)) if len(am) > 1 else [(am[0], c) for c in cols]
        for l, col in pairs:
            keep = [p for p in range(len(exps)) if col[p] != 0.0]
            es = [exps[p] for p in keep]
            cs = [col[p] for p in keep]
            if len(keep) == 1:
                cs = [1.0]
            shells.append((l, es, cs))
    shells.sort(key=lambda s: (s[0], -len(s[1]), -max(s[1])))
    return shells


def ncart(l):
    return (l + 1) * (l + 2) // 2


def table(name):
    """The shell rows (Z, l, nprim, p0) and primitive rows (exponent, coefficient)."""
    rows, prims, nao = [], [], {}
    for z in range(1, ZMAX + 1):
        nao[z] = [0, 0]
        for l, es, cs in element_shells(name, z):
            rows.append((z, l, len(es), len(prims)))
            for e, c in zip(es, cs):
                prims.append((e, c))
            nao[z][0] += ncart(l)
            nao[z][1] += 2 * l + 1
    return rows, prims, nao


def num(x):
    return repr(float(x))


def integer(x):
    return str(int(x))


def flat_lines(values, per_line):
    out = []
    for i in range(0, len(values), per_line):
        out.append(", ".join(values[i:i + per_line]))
    return (",\n        ").join(out)


def module_text(key, name, title):
    rows, prims, nao = table(name)
    fn = "basis_" + key
    up = "BASIS_" + key.upper()
    lmax = max(r[1] for r in rows)
    lines = []
    w = lines.append
    w("###")
    w("post: The %s basis set, H through Ar, as a table" % title)
    w("agent: claude")
    w("")
    w("Generated by `tools/basis_tables.py` from the Basis Set Exchange (the")
    w("`basis_set_exchange` package %s, data version %s of the set, general" % (BSE_VERSION, DATA_VERSION))
    w("contractions optimized). Do not edit; rerun the generator. The generator's")
    w("docstring says why version %s, how shells are split and in what order they" % DATA_VERSION)
    w("stand, and `tools/basis_tables.py --check` compares every element with")
    w("PySCF 2.5.0's copy of the set.")
    w("")
    w("`%s()` is the `Basis` record of lib/basis/basis: one row per segmented" % fn)
    w("shell, `(Z, l, nprim, p0)` with `p0` the row of its first primitive, and")
    w("one row per primitive, `(exponent, coefficient)`. Elements 1 to %d, %d" % (ZMAX, len(rows)))
    w("shells and %d primitives, angular momentum up to %s." % (len(prims), LETTERS[lmax]))
    w("")
    w("Used with `import basis.%s`, and passed to `molecule` (lib/hf/molecule)." % module_of(key))
    w("###")
    w("")
    w("from basis.basis import Basis, basis_check, basis_nao, basis_of, basis_rows, BS_NAO_LEN")
    w("")
    w("%s_NSHELL = %d" % (up, len(rows)))
    w("%s_NPRIM = %d" % (up, len(prims)))
    w("")
    w("#| (Z, l, nprim, p0), one segmented shell per row, by element.")
    w("def %s_shells_flat() -> Arr {" % fn)
    w("    return [" + flat_lines([integer(v) for r in rows for v in r], 16) + "]")
    w("}")
    w("")
    w("#| (exponent, coefficient), one primitive per row.")
    w("def %s_prims_flat() -> Arr {" % fn)
    w("    return [" + flat_lines([num(v) for p in prims for v in p], 4) + "]")
    w("}")
    w("")
    w("#| The %s basis set as a `Basis` record." % title)
    w("def %s() -> Basis {" % fn)
    w("    return basis_of(%s_shells_flat(), %s_prims_flat())" % (fn, fn))
    w("}")
    w("")
    w("#| The Cartesian and spherical AO counts of one atom of each element, `1` to `%d`," % ZMAX)
    w("#| as the generator counted them.")
    w("def %s_nao() -> Arr {" % fn)
    w("    return [" + flat_lines(["%d, %d" % (nao[z][0], nao[z][1]) for z in range(1, ZMAX + 1)], 6) + "]")
    w("}")
    w("")
    t = "bt" + key
    w('test "the %s table covers H through Ar, each element\'s shells in one run" {' % title)
    w("    %s_b = %s()" % (t, fn))
    w("    assert_near(len(%s_b.shells), %s_NSHELL, 0.0)" % (t, up))
    w("    assert_near(len(%s_b.prims), %s_NPRIM, 0.0)" % (t, up))
    w("    assert_near(basis_check(%s_b), 0.0, 0.0)" % t)
    w("    %s_n = basis_rows(%s_nao(), BS_NAO_LEN)" % (t, fn))
    w("    for %s_z to len(%s_n) {" % (t, t))
    w("        %s_k = basis_nao(%s_b, %s_z + 1)" % (t, t, t))
    w("        assert_near(%s_k[0], %s_n[%s_z, 0], 0.0)" % (t, t, t))
    w("        assert_near(%s_k[1], %s_n[%s_z, 1], 0.0)" % (t, t, t))
    w("    }")
    w("}")
    return "\n".join(lines) + "\n"


def module_of(key):
    """The module a set's table is: its key, or `pople<key>` for the Pople
    sets, whose keys start with a digit (a module name cannot)."""
    return "pople" + key if key[0].isdigit() else key


def out_path(key):
    return os.path.join(OUTDIR, module_of(key) + ".viv")


def write(keys):
    os.makedirs(OUTDIR, exist_ok=True)
    for key, name, title in SETS:
        if keys and key not in keys:
            continue
        with open(out_path(key), "w") as f:
            f.write(module_text(key, name, title))
        print("wrote lib/basis/%s.viv" % module_of(key))


def normalized(l, es, cs):
    import numpy as np
    es = np.array(es)
    cs = np.array(cs)
    a = es[:, None] + es[None, :]
    s = (2.0 * np.sqrt(es[:, None] * es[None, :]) / a) ** (l + 1.5)
    return cs / np.sqrt(cs @ s @ cs)


def pyscf_shells(name, z):
    from pyscf import gto
    mol = gto.M(atom="%s 0 0 0" % SYMBOLS[z], basis=name, spin=z % 2, cart=True, verbose=0)
    out = []
    for ib in range(mol.nbas):
        es = list(mol.bas_exp(ib))
        cs = mol.bas_ctr_coeff(ib)
        for ic in range(cs.shape[1]):
            keep = [p for p in range(len(es)) if cs[p, ic] != 0.0]
            out.append((mol.bas_angular(ib), [es[p] for p in keep], [cs[p, ic] for p in keep]))
    return out


def same_shell(a, b):
    import numpy as np
    if a[0] != b[0] or a[1] != b[1]:
        return False
    return float(np.max(np.abs(normalized(*a) - normalized(*b)))) < 1e-14


def check():
    if bse.version() != BSE_VERSION:
        print("basis_set_exchange is %s; the tables were written from %s" % (bse.version(), BSE_VERSION))
        return 1
    bad = 0
    for key, name, title in SETS:
        with open(out_path(key)) as f:
            if f.read() != module_text(key, name, title):
                print("lib/basis/%s.viv differs from a fresh generation" % module_of(key))
                bad += 1
    try:
        import pyscf
    except ImportError:
        print("PySCF is not installed: the tables match a fresh generation; the comparison with PySCF was skipped")
        return 1 if bad else 0
    if pyscf.__version__ != "2.5.0":
        print("PySCF is %s; the comparison is against 2.5.0" % pyscf.__version__)
        return 1
    reordered = 0
    for key, name, title in SETS:
        for z in range(1, ZMAX + 1):
            mine = element_shells(name, z)
            theirs = pyscf_shells(name, z)
            if len(mine) == len(theirs) and all(same_shell(a, b) for a, b in zip(mine, theirs)):
                if (name, z) in PYSCF_ORDER_DIFFERS:
                    print("%s %s: now in PySCF's order; take it off PYSCF_ORDER_DIFFERS" % (title, SYMBOLS[z]))
                    bad += 1
                continue
            permuted = len(mine) == len(theirs) and all(any(same_shell(a, b) for b in theirs) for a in mine)
            if permuted and (name, z) in PYSCF_ORDER_DIFFERS:
                reordered += 1
                continue
            print("%s %s: the shells differ from PySCF's" % (title, SYMBOLS[z]))
            bad += 1
    total = len(SETS) * ZMAX
    print("%d tables match a fresh generation; %d of %d elements have PySCF's shells in PySCF's order, "
          "%d have them in another order (PYSCF_ORDER_DIFFERS); %d problems"
          % (len(SETS), total - reordered - bad, total, reordered, bad))
    return 1 if bad else 0


def main(argv):
    if argv and argv[0] == "--check":
        return check()
    write(set(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

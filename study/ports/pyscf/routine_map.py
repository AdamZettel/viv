# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The side-by-side: each PySCF routine on the RHF/UHF path and the lib/hf
defs that do its work.

    python3 study/ports/pyscf/routine_map.py      writes results/routine_map.json,
                                                  prints the Markdown table NOTES.md holds

Line numbers are looked up, not typed: PySCF's from the installed source
(ast), Viv's from `def name(` in the repository. A routine or def that is
not found stops the script.
"""
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# (PySCF file, function or Class.method, what it does, [Viv defs], note)
MAP = [
    ("gto/mole.py", "M", "build a molecule from atom text and a basis name", ["molecule", "molecule_in", "parse_xyz", "basis_named", "molecule_from"], ""),
    ("gto/mole.py", "classical_coulomb_energy", "nuclear repulsion (Mole.energy_nuc)", ["mol_enuc"], ""),
    ("gto/mole.py", "MoleBase.intor", "integrals by libcint name", ["ao_overlap", "ao_hcore", "ao_eri"], "PySCF names the integral; lib/hf/ao calls the production member of each stage by name"),
    ("scf/hf.py", "get_ovlp", "overlap S (int1e_ovlp)", ["ao_overlap", "md1_s"], "libcint in PySCF; McMurchie-Davidson in Viv"),
    ("scf/hf.py", "get_hcore", "T + V (int1e_kin, int1e_nuc)", ["ao_hcore", "md1_t", "md1_v"], ""),
    ("scf/hf.py", "SCF.get_jk", "first call: the packed ERIs (int2e, s8)", ["ao_eri", "mderi_screened", "mderi_block", "herm_rfill", "tr4_blocks"], "libcint computes spherical integrals directly; Viv computes Cartesian blocks and transforms the packed array"),
    ("scf/_vhf.py", "incore", "J and K from packed ERIs (libcvhf)", ["jk_of", "jk_packed"], "C in PySCF, Viv in Viv"),
    ("scf/_vhf.py", "direct", "J and K from the basis, screened (libcvhf)", ["int_jk_direct", "mdjk_direct", "mdjk_put"], "PySCF screens by density and builds the Fock matrix incrementally; Viv does neither"),
    ("scf/hf.py", "get_veff", "the two-electron potential, incremental with dm - dm_last", ["jk_of", "den_fock"], "Viv rebuilds J and K from the full density each iteration"),
    ("scf/hf.py", "init_guess_by_1e", "the core-Hamiltonian guess", ["gs_core", "gs_occupy"], "PySCF's default guess (minao) has no Viv counterpart"),
    ("scf/hf.py", "eig", "scipy.linalg.eigh(F, S) each iteration", ["orth_canon", "den_orth", "eig_ql", "eig_tred2", "eig_tql2"], "PySCF solves the generalized problem with LAPACK each time; Viv orthogonalizes once and diagonalizes X^T F X with its own Householder + QL"),
    ("scf/hf.py", "get_occ", "aufbau occupations", ["scf_aufbau", "scf_occupy"], ""),
    ("scf/hf.py", "make_rdm1", "density from occupied orbitals", ["den_from_vectors"], ""),
    ("scf/hf.py", "get_fock", "F = h + vhf, then DIIS from diis_start_cycle", ["den_fock", "accel_push", "accel_extrapolate"], "Viv's diis_start option, added by this port, is PySCF's diis_start_cycle"),
    ("scf/diis.py", "get_err_vec_orth", "DIIS error C^T (S D F - F D S) C", ["scf_commutator", "den_orth_density", "den_commutator"], "same subspace: PySCF's Corth and Viv's X are both orthonormal bases of S"),
    ("lib/diis.py", "DIIS.update", "store a vector and its error", ["accel_push"], ""),
    ("lib/diis.py", "DIIS.extrapolate", "solve the DIIS equations, mix", ["accel_cdiis", "accel_diis_coeffs", "accel_solve"], ""),
    ("scf/hf.py", "energy_elec", "electronic energy", ["den_energy"], ""),
    ("scf/hf.py", "energy_tot", "plus nuclear repulsion", ["den_energy"], ""),
    ("scf/hf.py", "get_grad", "orbital gradient for the convergence test", ["scf_commutator", "den_maxabs"], "PySCF tests the norm of the occupied-virtual gradient; Viv the largest element of the commutator"),
    ("scf/hf.py", "kernel", "the SCF loop", ["scf_iterate"], ""),
    ("scf/hf.py", "SCF.scf", "run: integrals, loop, result", ["scf_with", "integrals", "scf", "scf_solve"], ""),
    ("scf/uhf.py", "get_veff", "UHF potential, both spins", ["jk_of", "den_fock"], ""),
    ("scf/uhf.py", "make_rdm1", "spin densities", ["den_from_vectors", "scf_occupy"], ""),
    ("scf/uhf.py", "spin_square", "<S^2>", ["den_s2"], ""),
]


def py_function(root, fn, qual):
    path = os.path.join(root, fn)
    tree = ast.parse(open(path).read())
    parts = qual.split(".")
    nodes = tree.body
    node = None
    for p in parts:
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p), None)
        if node is None:
            raise SystemExit(f"PySCF {fn}:{qual} not found")
        nodes = node.body
    return node.lineno, node.end_lineno


def viv_defs():
    out = {}
    for base in ("lib",):
        for root, _, files in os.walk(os.path.join(REPO, base)):
            for f in files:
                if f.endswith(".viv"):
                    p = os.path.join(root, f)
                    for i, line in enumerate(open(p), 1):
                        m = re.match(r"^def (\w+)\s*\(", line)
                        if m:
                            out.setdefault(m.group(1), "%s:%d" % (os.path.relpath(p, REPO), i))
    return out


def main():
    import pyscf
    root = os.path.dirname(pyscf.__file__)
    defs = viv_defs()
    rows = []
    for fn, qual, what, vdefs, note in MAP:
        a, b = py_function(root, fn, qual)
        locs = []
        for d in vdefs:
            if d not in defs:
                raise SystemExit(f"Viv def {d} not found")
            locs.append({"def": d, "at": defs[d]})
        rows.append({"pyscf": f"pyscf/{fn}:{a} {qual}", "pyscf_lines": [a, b], "what": what, "viv": locs, "note": note})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"pyscf": pyscf.__version__, "rows": rows}, open(os.path.join(HERE, "results", "routine_map.json"), "w"), indent=1)
    print("| PySCF %s | what it does | lib/hf | note |" % pyscf.__version__)
    print("|---|---|---|---|")
    for r in rows:
        print("| `%s` | %s | %s | %s |" % (r["pyscf"], r["what"], ", ".join("`%s` (%s)" % (l["def"], l["at"]) for l in r["viv"]), r["note"]))


if __name__ == "__main__":
    main()

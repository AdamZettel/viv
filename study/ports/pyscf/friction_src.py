# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Writes FRICTION.json from the entries below and the
measured numbers in results/ (final.json if present, else provisional.json),
so that every `slower` entry quotes a measured row.

    python3 study/ports/pyscf/friction_src.py

Line ranges are looked up by def name, not typed; an entry whose def is not
found stops the script.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def def_range(path, name):
    lines = open(os.path.join(REPO, path)).read().split("\n")
    for i, l in enumerate(lines):
        if re.match(r"^(def|record) %s\b" % re.escape(name), l):
            if l.rstrip().endswith("}"):
                return i + 1, i + 1
            for j in range(i + 1, len(lines)):
                if lines[j] == "}":
                    return i + 1, j + 1
    raise SystemExit(f"{name} not found in {path}")


def excerpt(path, a, b, pick=None):
    lines = open(os.path.join(REPO, path)).read().split("\n")[a - 1:b]
    if pick:
        lines = [l for l in lines if pick in l][:3] or lines[:3]
    return "\n".join(l.rstrip() for l in lines[:3])


def results():
    for f in ("final.json", "provisional.json"):
        p = os.path.join(HERE, "results", f)
        if os.path.exists(p):
            return f, json.load(open(p))["cases"]
    return None, {}


def stage(cases, name, eng, st):
    try:
        return cases[name][eng]["stages"][st]
    except (KeyError, TypeError):
        return None


def fmt(x):
    return "unmeasured" if x is None else ("%.1f ms" % (1e3 * x) if x < 1 else "%.2f s" % x)


# what the hf-fast branch changed, per finding (filled in from its timings)
RESOLVED = {
    "pyscf-003": "hf-fast: `mdjk_screened_q` (lib/hf/md_direct, the production direct member `int_jk_direct`) screens by Schwarz and density with PySCF's rule and threshold (1e-13), and the SCF builds J and K from the change of density (`jk_update`, full every 8 iterations and from the first that meets the convergence test). Measured on benzene cc-pVDZ: the density test keeps 85 to 99% of the 1,098,449 Schwarz-surviving quartets in every build (the change of density stays above 2e-6 at the timing settings), so it gains little here; the per-build time fell from 14.1 s to 7.0 s, mostly from the shell groups (pyscf-021). `mdjk_direct` (Schwarz only) stays as `int_jk_direct_full`, the member that checks it.",
    "pyscf-005": "hf-fast: `mderi_sph` (lib/hf/md_eri) moves each block to spherical functions as it is computed, with the pair transforms built once per store; `tr4_two_pass` (lib/hf/transform) is the old path and checks it (1e-14 on water 6-31G* and def2-TZVP). Water cc-pVTZ two-electron stage, native A/B on one machine: 385 ms before (279 + 108), 300 ms with the in-block transform, 237 ms with the shell groups.",
    "pyscf-007": "hf-fast: the commutator of two symmetric matrices is `A - A^T` with `A = F' P'` (`den_commutator_sym`, one product instead of a scalar triple loop), `S X` is computed once per run, and DIIS keeps its inner products (`Diis.gram`, k per push, PySCF's way) instead of rebuilding them. Water cc-pVTZ DIIS stage (commutator plus extrapolation), native: 16.3 ms before, 2.8 ms after; PySCF 1.3 ms.",
    "pyscf-009": "partial (hf-fast): `round(o.jk) == INT_JK_DIRECT` became `int_jk_is_direct(o.jk)`; making `method` an enum is blocked by the arch lens (pyscf-020).",
    "pyscf-010": "partial (hf-fast): `Diis` gained a `gram` field, the error inner products by ring slot; the four counters are still a number array.",
    "pyscf-021": "hf-fast: `mderi_groups` and `mderi_store_groups` (lib/hf/md_eri); the segmented store (`mderi_store_seg`) is the one `mderi_full` and `mdjk_direct` use, and the two give the same integrals to 1e-14 (same bits on water cc-pVTZ's SCF energy). Benzene cc-pVDZ direct, per J/K build: 14.1 s before, 7.0 s after.",
}


def main():
    which, C = results()
    star = "" if which == "final.json" else " (provisional: load at or above 2)"
    bzstar = " (final pass; the benzene runs saw a 1-minute load of 2.2 to 2.5, their own process included, and three attempts agreed within 0.2%)" if which == "final.json" else star
    tz = "water_ccpvtz"
    bz = "benzene_ccpvdz"
    E = []

    def add(id_, path, defname, pattern, original, lines_now, lines_saved, gap, status, axis, cost, note, pick=None, resolved=None):
        a, b = def_range(path, defname)
        e = {"id": "pyscf-%03d" % id_, "file": path, "start": a, "end": b, "def": defname, "pattern": pattern,
             "excerpt": excerpt(path, a, b, pick), "original": original, "lines_now": lines_now,
             "lines_saved": lines_saved, "gap": gap, "status": status, "axis": axis, "cost": cost, "note": note}
        # `resolved`: what the hf-fast branch changed for this finding, and what
        # it measured; the entry's status stays the one the port recorded
        if resolved:
            e["resolved"] = resolved
        E.append(e)

    jk_n, jk_r, jk_p = stage(C, tz, "native", "jk"), stage(C, tz, "release", "jk"), stage(C, tz, "pyscf", "jk")
    add(1, "lib/hf/jk.viv", "jk_packed", "W2", "libcvhf nr_incore.c CVHFnrs8_incore_drv: the same 8-fold scatter in C over a flat `double *` with `i*n+j` offsets",
        53, 0, "G2-index-cost", "missing", "slower",
        "water cc-pVTZ J/K stage: native %s, release %s, PySCF %s%s. Micro (10 builds, 58 AOs): 48 ms a build in the old 3-D form, 31 ms after this port's 2-D rewrite, 25 and 20 ms in release; libcvhf 4.9 ms. The release column is what bounds checks cost. The rest is the double-to-int conversion of every index: `viv_ix` clamps the double into int range and converts it even in release, and replacing it by a plain cast takes the release J/K stage from 238 to 90 ms (results/index_experiment.json)." % (fmt(jk_n), fmt(jk_r), fmt(jk_p), star),
        "Each stored integral is read once and added in 8 orders, 20 array accesses with 2 or 3 indices each. The code is as short as the C; the cost is per access.", pick="K[")
    add(2, "lib/hf/ao.viv", "ao_eri", "W19", "PySCF: `mf._eri = mol.intor('int2e', aosym='s8')` holds benzene's 21.1 million values in core (169 MB) by default",
        12, 0, "G7-array-limit", "missing", "slower",
        "benzene cc-pVDZ: PySCF in core %s against PySCF direct %s; Viv can only run direct (native %s)%s." % (
            fmt((C.get(bz, {}).get("pyscf_incore") or {}).get("total")), fmt((C.get(bz, {}).get("pyscf") or {}).get("total")), fmt((C.get(bz, {}).get("native") or {}).get("total")), bzstar),
        "zeros() stops at 16,000,000 elements in every engine, so the packed ERIs of benzene cc-pVDZ (21.1 M) and (H2O)8 6-31G* (54.5 M) cannot be stored and the run must be direct.")
    add(3, "lib/hf/md_direct.viv", "mdjk_direct_nch", "W20", "libcvhf nr_direct.c with `CVHFnr_optimizer`: Schwarz and density screening, and `get_veff` builds from `dm - dm_last`, so late iterations skip most quartets",
        27, 0, "algorithm", "legacy", "slower",
        "benzene cc-pVDZ direct, per J/K build: PySCF %s; Viv native %s%s." % (
            fmt(((C.get(bz, {}).get("pyscf") or {}).get("stages", {}).get("jk") or 0) / max(1, ((C.get(bz, {}).get("pyscf") or {}).get("cycles") or 0) + 2) or None),
            fmt(((C.get(bz, {}).get("native") or {}).get("stages", {}).get("jk") or 0) / max(1, (C.get(bz, {}).get("native") or {}).get("iterations") or 1) or None), bzstar),
        "Expressible in Viv today: J/K of D - D_prev added to the previous J/K, and a quartet skipped when Q_ab Q_cd max|dD| is below the threshold. It changes the rounding of the direct path, so it was left for a change that is allowed to move the last digits.",
        resolved=RESOLVED["pyscf-003"])
    add(4, "lib/hf/md_eri.viv", "mderi_block", "W21", "libcint: generated per-angular-momentum C, Rys quadrature, spherical output",
        58, 0, "G1-blas-matmul", "missing", "slower",
        "`@` (viv_matmul, cache-blocked since 456c969, still no BLAS) is 7% of the native samples of the benzene direct run and 5% of water cc-pVTZ (results/profile/summary.json).",
        "The block is written as two matrix products, E_cd @ BR and E_ab @ Z^T, so it inherits the unblocked `@`. The products are small (tens by hundreds), where blocking helps less than for large ones.", pick="@")
    add(5, "lib/hf/transform.viv", "tr4_quartet", "W22", "libcint applies the Cartesian-to-spherical matrix inside each shell quartet before the integrals are written (CINTc2s_*)",
        101, 0, "algorithm", "legacy", "slower",
        "water cc-pVTZ `ao_eri/transform`: 199 ms before this port's identity skip, 140 ms after (native, measured at load about 4); PySCF has no such stage.",
        "Viv computes Cartesian blocks into the packed array, then gathers each quartet back out through es_index8, transforms it and scatters it into a second packed array. Transforming the block inside mderi_block, before it is stored, would remove the gather and the second array.", pick="es_index8",
        resolved=RESOLVED["pyscf-005"])
    add(6, "lib/hf/eigen.viv", "eig_tql2", "W23", "scipy.linalg.eigh(F, S) -> LAPACK dsygvd (OpenBLAS)",
        77, 0, "lapack-eig", "missing", "slower",
        "water cc-pVTZ diagonalization: native %s, PySCF %s (PySCF also solves the generalized problem each time)%s." % (fmt(stage(C, tz, "native", "diagonalize")), fmt(stage(C, tz, "pyscf", "diagonalize")), star),
        "Householder tridiagonalization and implicit QL (EISPACK tred2/tql2), written in Viv; O(n^3) with no blocking.")
    add(7, "lib/hf/scf_loop.viv", "scf_commutator", "W2", "PySCF CDIIS.update: `reduce(numpy.dot, (C.T, s, d, f, C))`, BLAS dgemm",
        24, 0, "G1-blas-matmul", "missing", "slower",
        "water cc-pVTZ DIIS stage (commutator and extrapolation): native %s, PySCF %s%s. PySCF runs this stage in Python, so the gate asks Viv to be faster here." % (fmt(stage(C, tz, "native", "diis")), fmt(stage(C, tz, "pyscf", "diis")), star),
        "Six or so m x m products per iteration through Viv's unblocked `@` against BLAS.",
        resolved=RESOLVED["pyscf-007"])
    add(8, "lib/hf/opts.viv", "opt_array", "W1", "PySCF reads `mf.diis_space`, `mf.conv_tol` from the object",
        25, 23, "records-through-apply", "missing", "longer",
        "25 lines", "Registry members are called through `apply`, which passes numbers, functions and arrays, so the options record is copied into a column array with named offsets for them.")
    add(9, "lib/hf/opts.viv", "ScfOptions", "W18", "PySCF: strings and objects (`mf.init_guess = '1e'`, `mf.DIIS = scf.CDIIS`)",
        20, 0, "enums-as-numbers", "legacy", "less-clear",
        "`method: Num`, `guess: Num`, `jk: Num` hold member numbers (`GS_CORE = 1`, `INT_JK_DIRECT = 3`) where an enum now exists; `round(o.jk) == INT_JK_DIRECT` in scf.viv.",
        "Written before int-enum merged; Int and enums with match exist now.",
        resolved=RESOLVED["pyscf-009"])
    add(10, "lib/hf/accelerate.viv", "accel_push", "W1", "pyscf/lib/diis.py keeps Python lists of vectors and a dict of errors",
        10, 6, "struct-as-number-array", "partial", "less-clear",
        "The ring of Fock matrices and errors is one `[2 nd + 1, k, m, m]` array with a 4-slot `meta` number array (count, head, capacity, steps).",
        "Records exist; a record of arrays plus Int fields would name the four counters. Arrays of arrays (a list of matrices) do not.",
        resolved=RESOLVED["pyscf-010"])
    add(11, "lib/hf/md_eri.viv", "mderi_store", "W1", "PySCF passes `atm`, `bas`, `env` int/double arrays to libcint, the same flat layout",
        3, 0, "struct-as-number-array", "legacy", "less-clear",
        "The shell-pair store is one flat number array read at offsets (`ST[o1 + 2]`), as libcint's env is; md_eri.viv says why: it predates records.",
        "Not longer than the C original; less clear than a record per shell pair would be.")
    add(12, "lib/hf/jk_direct.viv", "int_jk_direct", "W1", "PySCF's `_vhf.direct(dms, mol._atm, mol._bas, mol._env)`",
        33, 25, "records-through-apply", "missing", "longer",
        "33 lines; the blob is packed by int_basis_blob and unpacked here",
        "A direct J/K member gets the basis through the same array slots as the packed integrals, so atoms, shells, primitives and the c2s matrix are flattened into one 'blob' and read back by offsets.")
    add(13, "lib/hf/jk.viv", "jk_packed", "W24", "C: `int` loop counters",
        53, 0, "G2-int-loops", "partial", "slower",
        "Declaring the counters Int (`n: Int = int(sh[1])`, `p: Int = 0`) does not give integer loops: in the generated C the loop counters stay `double` and every Int add becomes a checked `viv_i_add` call (read in the C, not timed; reverted). What an integer index would recover is measured in results/index_experiment.json: the whole water cc-pVTZ run in release goes from 557 to 281 ms, (H2O)4 6-31G* from 1.72 to 0.95 s.",
        "The integer index rule exists; its C lowering is not an integer loop yet.", pick="for")
    add(14, "lib/hf/tests/transform.viv", "tr4_low_l_gap", "W25", "Python: `for sh in (water631gs, ccpvdz, svp): check(sh)`",
        21, 4, "values-in-containers", "partial", "longer",
        "4 lines: a list of three shell tables cannot be written (`expected a number for a vector element`), so the test calls a helper three times",
        "Also: top-level names in test blocks share one namespace with every module, so a test variable needs a module prefix (`tr4_t...`) and reusing a name with another type stops the compile (namespaces).")
    add(15, "lib/hf/scf_loop.viv", "scf_iterate", "W26", "PySCF: `logger.timer(mf, 'cycle=...')` from `time.perf_counter`",
        65, 0, "G7-clock", "missing", "less-clear",
        "No clock builtin: the stage times had to come from a patch of the generated C (study/ports/harness/port.py, STAGE_CLOCK_C) hooking the step events; wasm and the interpreter get only process times.",
        "The steps are there; a clock would let the program report its own stage times in every engine.")
    add(16, "lib/hf/guess.viv", "gs_core", "W27", "PySCF's default guess, `minao`: superposed atomic densities from its ANO tables",
        4, 0, "atomic-data", "missing", "less-clear",
        "Both sides had to be run from the core guess; a user who copies PySCF's defaults gets a different starting point and, for open shells, possibly another state.",
        "Needs atomic densities per element and basis.")
    add(17, "lib/hf/scf_loop.viv", "scf_iterate", "W28", "the pseudocode lens of the loop",
        65, 0, "tooling", "partial", "less-clear",
        "`viv view lib/hf/scf_loop.viv --lens pseudocode --def scf_iterate` prints the loop header twice: `for it from 1 to setup.options.max_iter + 1` and again `for it` under the repeat line (results/scf_iterate.pseudocode.txt, lines 22 and 25).",
        "A rendering problem of the pseudocode view, not of lib/hf.", pick="for it")
    add(18, "lib/hf/jk.viv", "jk_build_methods", "W30", "Python: `for f in (update, rebuild): f(dm, dm_last, vhf_last)`",
        3, 0, "tooling", "partial", "longer",
        "In lib/hf/tests/jk_direct.viv, `apply(jk_build_methods()[1], ...)` was refused: \"`int_jkw` is a number but is assigned a array\", whatever the variable's name; the same call compiles in a test program that uses only lib/hf/jk, lib/hf/integrals and lib/hf/scf_loop. The test calls the two members by name instead.",
        "The registry's result type is withheld until a late fixpoint; something in the larger program fixes it as a number first.")
    add(19, "lib/hf/rohf.viv", "rohf_t_regime", "W31", "the intent lens of a test helper",
        19, 0, "tooling", "partial", "less-clear",
        "`viv lens check lib/hf/rohf.viv` panicked (lower.rs:1905, index out of bounds) while a def of the module was stale: looking for the version whose body the old stamp names, it lowered an older module together with the current ones, and the program held two `rohf_effective` defs (3 and 4 parameters). Stamping the def made the panic go away; nothing in lib/hf caused it.",
        "A stale entry should report, not crash; found while ScfOptions.method was an enum for a few edits (reverted, entry 20).")
    add(20, "lib/hf/opts.viv", "ScfOptions", "W18", "PySCF: `mf.init_guess = '1e'`",
        20, 0, "tooling", "partial", "less-clear",
        "`method: ScfMethod` (an enum `{ Auto, Rhf, Uhf, Rohf }` with `SCF_RHF = ScfMethod.Rhf`) compiled and passed the tests in all three engines, but `viv arch check` (lib/hf/layers.arch) stopped on it: \"unknown type `ScfMethod`\" (the arch lens lowers programs without the enum declarations), and an enum cannot be an array element (scf/coverage.viv and lib/hf/rohf.viv keep their methods in arrays) or an engine input (engines/hf-energy takes numbers). The conversion was reverted; `method` stays a number.",
        "Waits for the arch lens to lower enums and for enums as array elements. The registry fields (`guess`, `jk`, ...) index registries and are swept as numbers, so they are numbers by design; `round(o.jk) == INT_JK_DIRECT` became `int_jk_is_direct(o.jk)`.")
    add(21, "lib/hf/md_eri.viv", "mderi_groups", "W29", "libcint computes the primitive integrals of a general contraction once for all its contractions (`CINTg0_*` over `nctr`)",
        28, 0, "algorithm", "legacy", "slower",
        "Not in the port's ledger: found while timing the direct path. Benzene cc-pVDZ in segmented shells has 9.24e7 primitive quartets; with each carbon's two s contractions (same eight exponents) as one group, 2.29e7, and the Hermite R work drops from 5.38e8 to 2.00e8 terms (counted from the basis table).",
        "Expressible in Viv today: the E rows of a group pair carry each shell pair's coefficients, and the R tables are shared.",
        resolved=RESOLVED["pyscf-021"])
    add(22, "lib/hf/accelerate.viv", "accel_inner", "W32", "NumPy: `numpy.dot(xi.ravel(), xj.ravel())`",
        3, 0, "G10-wasm-bridge", "missing", "slower",
        "In wasm, `ravel` of a 49-element array and `dot` of two of them each took about 0.2 ms a call (800 calls: 0.23 s against 0.08 s for the loop without them; `sum(a * b)` and `@` on the same arrays cost nothing measurable). The first version of the kept DIIS inner products used `dot(ravel(...), ravel(...))` and added 0.17 s to every wasm SCF run of water (0.10 s to 0.27 s); `sum(E * slab(hist, slot))` restored it.",
        "Native code does not show it (the same calls cost microseconds). Found by the wasm column of the provisional timing.")
    out = {"port": "pyscf", "schema": "port friction ledger", "measured_from": which, "findings": E}
    json.dump(out, open(os.path.join(HERE, "FRICTION.json"), "w"), indent=1)
    open(os.path.join(HERE, "FRICTION.json"), "a").write("\n")
    print("wrote FRICTION.json: %d entries (%s)" % (len(E), which))


if __name__ == "__main__":
    main()

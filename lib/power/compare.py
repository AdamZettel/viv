# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""lib/power against its references; writes lib/power/compare.json.

    VIV=<viv binary> python3 lib/power/compare.py [--big path/to/case9241pegase.m]

1. MATPOWER's own solved t_case9_pf (lib/t/soln9_pf.mat and soln9_dcpf.mat in
   the MATPOWER repository): every bus, generator and branch result column.
2. Every case by NR, FDXB and DC against the independent NumPy/SciPy power
   flow of lib/power/oracle/pf_scipy.py (FDXB is held to the NR answer: the
   same equations at the same tolerance).
3. The mismatch at Viv's solution, recomputed here in NumPy.
4. The sparse LU alone: each case's Newton Jacobian solved by splu and by
   scipy.sparse.linalg.splu, with the fill of each.
5. Timings, best of several runs, Viv (compiled, through vipyr) and SciPy.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE / "oracle"))
import pf_scipy  # noqa: E402
from vipyr import _power  # noqa: E402

CASES = ["case9", "case14", "case30", "case118", "case300", "case2383wp"]
KEYS = ["vm", "va", "pg", "qg", "pf", "qf", "pt", "qt"]


def best(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, min(ts)


def maxdiff(a, b):
    return {k: float(np.max(abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in KEYS}


def residual(c, r):
    f, t, g = pf_scipy._internal(c)
    Y, _, _ = pf_scipy.make_ybus(c, f, t)
    V = r["vm"] * np.exp(1j * np.radians(r["va"]))
    S, _ = pf_scipy.sbus_v0(c, g)
    ref, pv, pq = pf_scipy.bus_types(c, g)
    mis = V * np.conj(Y @ V) - S
    return float(max(np.max(abs(mis[np.r_[pv, pq]].real)), np.max(abs(mis[pq].imag))))


def matpower_t_case9():
    c = _power.read_case(HERE / "cases" / "t_case9_pf.m")
    rows = []
    for method, sol in (("nr", "soln9_pf.mat"), ("fdxb", "soln9_pf.mat"), ("dc", "soln9_dcpf.mat")):
        m = scipy.io.loadmat(HERE / "cases" / sol)
        r = _power._runpf(c, method)
        ref = {"vm": m["bus_soln"][:, 7], "va": m["bus_soln"][:, 8], "pg": m["gen_soln"][:, 1],
               "qg": m["gen_soln"][:, 2], "pf": m["branch_soln"][:, 13], "qf": m["branch_soln"][:, 14],
               "pt": m["branch_soln"][:, 15], "qt": m["branch_soln"][:, 16]}
        rows.append(dict(method=method, iterations=r["iterations"], **maxdiff(r, ref)))
    return rows


def lu_row(name, c, api):
    colptr, rowind, vals = api.jacobian_at(c["baseMVA"], c["bus"], c["gen"], c["branch"])
    n = len(colptr) - 1
    J = sp.csc_matrix((vals, rowind, colptr), (n, n))
    b = np.random.default_rng(0).standard_normal(n)
    (x, nnz_lu, q), t_viv = best(lambda: api.lu_check(n, colptr, rowind, vals, b, 0.001), 5)
    lu, t_sp = best(lambda: spla.splu(J, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.001), 5)
    xs = lu.solve(b)
    lu2 = spla.splu(J)
    return dict(case=name, n=n, nnz=int(J.nnz), nnz_lu=int(nnz_lu), nnz_superlu_mmd=int(lu.L.nnz + lu.U.nnz - n),
                nnz_superlu_colamd=int(lu2.L.nnz + lu2.U.nnz - n),
                dx_rel=float(np.max(abs(x - xs)) / np.max(abs(xs))),
                resid_rel=float(np.max(abs(J @ x - b)) / np.max(abs(b))),
                t_viv=t_viv, t_superlu=t_sp)


def markdown(out):
    L = ["# lib/power against its references", "",
         "Written by `lib/power/compare.py` (do not edit). Differences are the largest over all buses, "
         "generators or branches: Vm in per unit, Va in degrees, P and Q in MW and MVAr.", "",
         "## MATPOWER's own solution of t_case9_pf", "",
         "Reference: `lib/t/soln9_pf.mat` (AC) and `soln9_dcpf.mat` (DC) from the MATPOWER repository, commit "
         "80b6e16. FDXB is held to the mismatch tolerance (1e-8 per unit), not to rounding.", "",
         "| method | iterations | Vm | Va | Pg | Qg | Pf | Qf | Pt | Qt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["matpower_t_case9"]:
        L.append("| %s | %d | " % (r["method"], r["iterations"]) + " | ".join("%.1e" % r[k] for k in KEYS) + " |")
    L += ["", "## Every case against the independent SciPy power flow", "",
          "Reference: `lib/power/oracle/pf_scipy.py` (MATPOWER's matrix formulas in scipy.sparse, solved by "
          "SuperLU). NR and FDXB are compared with its Newton solution, DC with its DC solution. `mismatch` is "
          "the largest power mismatch at Viv's answer, recomputed in NumPy (per unit). Times are the best of "
          "several runs on an Apple M2 Pro with other jobs running (load about 10).", "",
          "| case | buses | method | iter (Viv / SciPy) | Vm | Va | Pg | Qg | Pf | Qf | mismatch | Viv ms | SciPy ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["cases"]:
        L.append("| %s | %d | %s | %d / %d | %s | %s | %.2f | %s |" % (
            r["case"], r["nb"], r["method"], r["iterations"], r["ref_iterations"],
            " | ".join("%.1e" % r[k] for k in ("vm", "va", "pg", "qg", "pf", "qf")),
            "%.1e" % r["mismatch"] if "mismatch" in r else "", 1e3 * r["t_viv"],
            "%.2f" % (1e3 * r["t_ref"]) if r["method"] != "fdxb" else ""))
    L += ["", "## The sparse LU alone", "",
          "Each case's Newton Jacobian at the starting voltages, solved by `splu` (minimum-degree order, "
          "threshold 0.001) and by `scipy.sparse.linalg.splu` (SuperLU, MMD_AT_PLUS_A order, the same "
          "threshold). Fill is the entries of L and U, the diagonal counted once; SuperLU's default COLAMD "
          "order is shown too. Times include the ordering. SuperLU's time on case9241pegase varied from 22 ms to "
          "2.5 s between runs on the loaded machine; Viv's stayed between 38 and 96 ms.", "",
          "| case | n | nnz(J) | nnz(L+U) Viv | SuperLU MMD | SuperLU COLAMD | max dx / max x | residual | Viv ms | SuperLU ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["lu"]:
        L.append("| %s | %d | %d | %d | %d | %d | %.1e | %.1e | %.2f | %.2f |" % (
            r["case"], r["n"], r["nnz"], r["nnz_lu"], r["nnz_superlu_mmd"], r["nnz_superlu_colamd"], r["dx_rel"],
            r["resid_rel"], 1e3 * r["t_viv"], 1e3 * r["t_superlu"]))
    L += ["", "## The voltages stored in the case files", "",
          "Not a reference: each file stores the voltages it was distributed with (case9 a flat start; the IEEE "
          "cases the solution of their original data files, rounded; the others the voltages their authors stored), "
          "so they differ from MATPOWER's solution by up to these amounts.",
          "", "| case | max dVm | max dVa (deg) |", "|---|---|---|"]
    for r in out["stored"]:
        L.append("| %s | %.2e | %.2e |" % (r["case"], r["dvm"], r["dva"]))
    return "\n".join(L) + "\n"


def main():
    big = sys.argv[sys.argv.index("--big") + 1] if "--big" in sys.argv else None
    api = _power._api()
    out = {"matpower_t_case9": matpower_t_case9(), "cases": [], "lu": [], "stored": []}
    for name in CASES + ([big] if big else []):
        c, label = _power.load_case(name)
        reps = 3 if c["bus"].shape[0] > 5000 else 10
        ref_nr, t_ref_nr = best(lambda: pf_scipy.runpf_nr(c), reps)
        ref_dc, t_ref_dc = best(lambda: pf_scipy.runpf_dc(c), reps)
        for method in ("nr", "fdxb", "dc"):
            r, t = best(lambda: _power._runpf(c, method), reps)
            ref = ref_dc if method == "dc" else ref_nr
            row = dict(case=label, nb=int(c["bus"].shape[0]), method=method, success=r["success"],
                       iterations=r["iterations"], t_viv=t, ref_iterations=ref["iterations"],
                       t_ref=t_ref_dc if method == "dc" else t_ref_nr, **maxdiff(r, ref))
            if method != "dc":
                row["mismatch"] = residual(c, r)
            out["cases"].append(row)
            if method == "nr":
                #: the voltages stored in the case file, where the file holds a solved state
                out["stored"].append(dict(case=label, dvm=float(np.max(abs(r["vm"] - c["bus"][:, 7]))),
                                          dva=float(np.max(abs(r["va"] - c["bus"][:, 8])))))
        out["lu"].append(lu_row(label, c, api))
        print(label, "done", flush=True)
    (HERE / "compare.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "COMPARISON.md").write_text(markdown(out))
    print(json.dumps(out["matpower_t_case9"], indent=1))
    for r in out["cases"]:
        print("%-15s %-4s it %2d/%2d  dVm %.1e dVa %.1e dPg %.1e dQg %.1e dPf %.1e dQf %.1e  viv %.4f s  ref %.4f s"
              % (r["case"], r["method"], r["iterations"], r["ref_iterations"], r["vm"], r["va"], r["pg"], r["qg"],
                 r["pf"], r["qf"], r["t_viv"], r["t_ref"]))
    for r in out["lu"]:
        print(r)
    for r in out["stored"]:
        print(r)


if __name__ == "__main__":
    main()

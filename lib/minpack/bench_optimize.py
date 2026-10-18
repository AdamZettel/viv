# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.optimize against NIST's certified values and against SciPy.

    python3 lib/minpack/bench_optimize.py [--quick]     (from the repository root)

1. NIST StRD nonlinear regression, all 27 problems from both certified
   starts. The 26 with one predictor go through vp.optimize.curve_fit (the
   models of lib/minpack/nist_strd.py, translated by py2viv), with the exact
   Jacobian (grad) and with forward differences, and through SciPy's
   curve_fit (method 'lm', forward differences); Nelson (two predictors)
   goes through least_squares. Each row: the certified digits matched (NIST's
   log relative error, LRE, of the worst parameter and of the worst standard
   deviation), the evaluation counts, MINPACK's info, and warm wall times.
2. The 24 Moré-Garbow-Hillstrom cases of MINPACK's test driver
   (lib/minpack/mgh_ls.viv) through vp.optimize.least_squares, with
   SSQJAC's Jacobian and with grad's, against SciPy's
   least_squares(method='lm') given the same Jacobian in NumPy
   (lib/minpack/oracle/scipy_check.py).
3. The square MGH problems through vp.optimize.root against SciPy's
   root(method='lm').

Writes lib/minpack/bench_optimize.json (with the machine and its load) and
prints the tables. Exits 1 when a fit that SciPy's comparable call
completes fails here, or when vp.optimize.curve_fit with the exact Jacobian
matches fewer certified digits than NIST's own bar for a careful code (4)
on a problem SciPy also solves.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "oracle"))
sys.path.insert(0, str(ROOT / "python"))

import scipy  # noqa: E402
from scipy.optimize import curve_fit, least_squares, root  # noqa: E402

import nist_strd as N  # noqa: E402
import vipyr as vp  # noqa: E402
from vipyr._optimize import covariance  # noqa: E402

MIN_LRE = 4.0
REPEAT = 5


def best(fn, repeat=REPEAT):
    """The smallest wall time of `repeat` calls (warm), and the last result."""
    t, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        t = min(t, time.perf_counter() - t0)
    return t, out


def scipy_cf(p, start):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov, info, msg, ier = curve_fit(N.MODELS[p.name], p.x, p.y, start, full_output=True)
    return popt, np.sqrt(np.diag(pcov)), info["nfev"], ier


def vv_cf(p, start, jac):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = vp.optimize.curve_fit(N.MODELS[p.name], p.x, p.y, start, jac=jac)
    return r.popt, r.perr, r.nfev, r.njev, r.info


def nelson(p, start, jac):
    x1, x2 = p.x
    if jac == "scipy":
        r = least_squares(N.nelson_res, start, args=(x1, x2, p.y), method="lm")
        return r.x, r.jac, r.fun, r.nfev, r.njev, r.status
    r = vp.optimize.least_squares(N.nelson_res, start, args=(x1, x2, p.y), jac=jac)
    return r.x, r.jac, r.fun, r.nfev, r.njev, r.status


def run_nist(names, repeat):
    rows, bad = [], []
    for name in names:
        p = N.load(name)
        for s, start in enumerate(p.starts, 1):
            row = {"problem": name, "difficulty": p.difficulty, "start": s, "m": len(p.y), "n": len(start)}
            for label, jac in (("vp", None), ("vp_fd", "fd"), ("scipy", "scipy")):
                try:
                    if name == "Nelson":
                        fn = (lambda jac=jac: nelson(p, start, jac))
                        nelson(p, start, jac)
                        t, (x, J, f, nfev, njev, st) = best(fn, repeat)
                        sd = np.sqrt(np.diag(covariance(J, float(f @ f), len(f), len(x), False)))
                        info = st
                    elif jac == "scipy":
                        t, (x, sd, nfev, info) = best(lambda: scipy_cf(p, start), repeat)
                        njev = None
                    else:
                        vv_cf(p, start, jac)          # compile once, outside the timing
                        t, (x, sd, nfev, njev, info) = best(lambda jac=jac: vv_cf(p, start, jac), repeat)
                    row[label] = {"ok": True, "lre": round(float(N.lre(x, p.certified).min()), 1),
                                  "lre_sd": round(float(N.lre(sd, p.cert_sd).min()), 1), "nfev": int(nfev),
                                  "njev": None if njev is None else int(njev), "info": int(info), "s": t}
                except (RuntimeError, ValueError, ArithmeticError) as e:
                    row[label] = {"ok": False, "error": str(e).splitlines()[0][:120]}
            rows.append(row)
            sp, me = row["scipy"], row["vp"]
            if sp["ok"] and sp["lre"] >= MIN_LRE and (not me["ok"] or me["lre"] < MIN_LRE):
                bad.append("%s start %d" % (name, s))
            print(fmt_nist(row), flush=True)
    return rows, bad


def fmt_nist(r):
    def cell(c):
        if not c["ok"]:
            return "%-34s" % ("FAILED " + c["error"][:27])
        return "%4.1f %4.1f %4d %4s %d %7.2fms" % (c["lre"], c["lre_sd"], c["nfev"],
                                                    "-" if c["njev"] is None else c["njev"], c["info"], 1e3 * c["s"])
    return "%-9s %d  %s | %s | %s" % (r["problem"], r["start"], cell(r["vp"]), cell(r["vp_fd"]), cell(r["scipy"]))


MGH_CASES = [(4, 2, 2), (5, 3, 3), (6, 4, 4), (7, 2, 2), (8, 3, 15), (9, 4, 11)]
MGH_CASES = [(p, n, m, f) for p, n, m in MGH_CASES for f in (1.0, 10.0, 100.0)] + \
    [(10, 3, 16, 1.0), (10, 3, 16, 10.0), (12, 3, 10, 1.0), (13, 2, 10, 1.0), (17, 5, 33, 1.0), (18, 11, 65, 1.0)]
MGH_NAMES = {4: "Rosenbrock", 5: "helical valley", 6: "Powell singular", 7: "Freudenstein-Roth", 8: "Bard",
             9: "Kowalik-Osborne", 10: "Meyer", 12: "Box 3-D", 13: "Jennrich-Sampson", 17: "Osborne 1",
             18: "Osborne 2"}


def run_mgh(repeat):
    from scipy_check import f_and_j
    k = vp.load(HERE / "mgh_ls.viv")
    src = str(HERE / "mgh_ls.viv") + ":mgh_res"
    rows = []
    for nprob, n, m, factor in MGH_CASES:
        f, j, _ = f_and_j(nprob, m)
        x0 = np.asarray(k.mgh_x0(float(nprob), n, factor))
        prob = np.array([float(nprob), float(m)])
        row = {"problem": MGH_NAMES[nprob], "nprob": nprob, "n": n, "m": m, "factor": factor}
        for label, call in (
                ("vv_ssqjac", lambda: vp.optimize.least_squares(src, x0, args=(prob,), jac="mgh_jac")),
                ("vv_grad", lambda: vp.optimize.least_squares(src, x0, args=(prob,))),
                ("scipy", lambda: least_squares(f, x0, jac=j, method="lm"))):
            call()
            t, r = best(call, repeat)
            row[label] = {"x": [float(v) for v in r.x], "cost": float(r.cost), "nfev": int(r.nfev),
                          "njev": int(r.njev), "status": int(r.status), "s": t}
        sp = row["scipy"]
        for label in ("vv_ssqjac", "vv_grad"):
            c = row[label]
            c["same_counts"] = c["nfev"] == sp["nfev"] and c["njev"] == sp["njev"] and c["status"] == sp["status"]
            c["max_rel_dx"] = float(np.linalg.norm(np.array(c["x"]) - sp["x"]) / max(np.linalg.norm(sp["x"]), 1e-300))
        rows.append(row)
        print("%-18s %5g  ssqjac %4d %4d %s %8.2e   grad %4d %4d %s %8.2e   scipy %4d %4d st %d  cost %.6e"
              % (row["problem"], factor, row["vv_ssqjac"]["nfev"], row["vv_ssqjac"]["njev"],
                 "=" if row["vv_ssqjac"]["same_counts"] else "!", row["vv_ssqjac"]["max_rel_dx"],
                 row["vv_grad"]["nfev"], row["vv_grad"]["njev"], "=" if row["vv_grad"]["same_counts"] else "!",
                 row["vv_grad"]["max_rel_dx"], sp["nfev"], sp["njev"], sp["status"], sp["cost"]), flush=True)
    return rows


def run_root(repeat):
    from scipy_check import f_and_j
    k = vp.load(HERE / "mgh_ls.viv")
    src = str(HERE / "mgh_ls.viv") + ":mgh_res"
    rows = []
    for nprob, n in ((4, 2), (5, 3), (6, 4), (7, 2)):
        f, j, _ = f_and_j(nprob, n)
        x0 = np.asarray(k.mgh_x0(float(nprob), n, 1.0))
        prob = np.array([float(nprob), float(n)])
        mine = lambda: vp.optimize.root(src, x0, args=(prob,), jac="mgh_jac")
        theirs = lambda: root(f, x0, jac=j, method="lm")
        mine()
        tm, a = best(mine, repeat)
        ts, b = best(theirs, repeat)
        row = {"problem": MGH_NAMES[nprob], "vp": {"x": a.x.tolist(), "nfev": a.nfev, "njev": a.njev,
                                                  "status": a.status, "s": tm},
               "scipy": {"x": b.x.tolist(), "nfev": int(b.nfev), "njev": int(b.njev), "status": int(b.status),
                         "s": ts}}
        rows.append(row)
        print("%-18s vp %4d %4d info %d |F| %.1e   scipy %4d %4d info %d |F| %.1e"
              % (row["problem"], a.nfev, a.njev, a.status, np.linalg.norm(a.fun), b.nfev, b.njev, b.status,
                 np.linalg.norm(b.fun)), flush=True)
    return rows


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    except OSError:
        cpu = ""
    return (cpu.strip() or platform.processor() or platform.machine()) + ", " + platform.system()


def main():
    quick = "--quick" in sys.argv
    repeat = 1 if quick else REPEAT
    names = ["Misra1a", "Thurber", "MGH09", "Eckerle4", "Rat43", "Bennett5"] if quick else N.NAMES
    load0 = os.getloadavg()[0]
    print("NIST StRD: problem start | vp (grad): LRE params, LRE sd, nfev, njev, info, time | vp (fd) | scipy (fd)")
    nist, bad = run_nist(names, repeat)
    print("\nMGH through least_squares (nfev, njev, = when SciPy's counts and status match, |x - x_scipy| / |x_scipy|)")
    mgh = run_mgh(repeat)
    print("\nMGH square systems through root(method='lm')")
    roots = run_root(repeat)
    out = {"machine": machine(), "load_average_start": round(load0, 2), "load_average_end":
           round(os.getloadavg()[0], 2), "scipy": scipy.__version__, "numpy": np.__version__,
           "repeat": repeat, "nist": nist, "mgh": mgh, "root": roots}
    if not quick:
        (HERE / "bench_optimize.json").write_text(json.dumps(out, indent=1) + "\n")
        print("\nwrote lib/minpack/bench_optimize.json")
    if bad:
        print("FAILED (scipy completes, vp.optimize does not reach %g digits): %s" % (MIN_LRE, ", ".join(bad)))
        sys.exit(1)


if __name__ == "__main__":
    main()

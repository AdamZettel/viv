# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Runhai Ouyang (SISSO, github.com/rouyang2017/SISSO)
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Hold the Viv port of SISSO to the Fortran (SISSO 3.5) on the same data.

    SISSO_BIN=/path/to/SISSO python3 study/ports/sisso/compare_fortran.py [--cases a,b] [--record] [--runs DIR]

For each case it writes SISSO.in and train.dat, runs the Fortran (one
process) once per descriptor dimension 1..D (SISSO.out prints only the last
dimension's model), and runs python/vipyr/sisso.fit on the same numbers. It
requires the same feature-space size of every Phi_k at every dimension, the
same SIS subspace (names, in order), the same descriptors, and coefficients,
intercepts and RMSE equal to the Fortran's printed ten digits (1e-9
relative). Timing: the Fortran's wall time for the D run against vipyr's
(library built and warm), best of three each.

--record writes study/ports/sisso/fortran_reference.json, which
python/tests/test_sisso.py reads, so the tests need no Fortran.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from vipyr import sisso  # noqa: E402

REL = 1e-9


def example():
    d = np.array([[3.0862, 0.8626, 0.7043, 0.6312], [2.8854, 0.7260, 0.7818, 0.6119], [0.6907, 0.4943, 0.0044, 0.4420],
                  [0.9902, 0.0106, 0.0399, 0.9877], [0.7242, 0.0970, 0.3199, 0.5504]])
    cols = {"property": d[:, 0], "feature1": d[:, 1], "feature2": d[:, 2], "feature3": d[:, 3]}
    return dict(df=cols, target="property", features=["feature1", "feature2", "feature3"],
                groups=[["feature1", "feature2"], ["feature3"]], ops="(+)(-)(*)(/)", fcomplexity=3, dim=2,
                nf_sis=50000)


def synthetic():
    """y = 3 x1 x2 / x3 - 1.5 x4^2 + 0.5, plus 1e-2 noise; x1, x2 in m, x3 in s."""
    rng = np.random.default_rng(20181018)
    n = 50
    x = np.round(rng.uniform(0.5, 2.5, size=(5, n)), 4)
    noise = np.round(rng.normal(0.0, 0.01, size=n), 5)
    y = np.round(3.0 * x[0] * x[1] / x[2] - 1.5 * x[3] ** 2 + 0.5 + noise, 6)
    cols = {"y": y}
    for k in range(5):
        cols["x%d" % (k + 1)] = x[k]
    return dict(df=cols, target="y", features=["x1", "x2", "x3", "x4", "x5"], groups=[["x1", "x2"], ["x3"]],
                ops="(+)(-)(*)(/)(^2)(sqrt)(exp)", fcomplexity=3, dim=2, nf_sis=50)


def timing():
    """A larger space: 7 features, 11 operators, rung 2, three dimensions."""
    rng = np.random.default_rng(83802)
    n = 200
    x = np.round(rng.uniform(0.2, 3.0, size=(7, n)), 4)
    y = np.round(2.0 * np.log(x[0]) * x[1] + 0.5 * np.sqrt(x[2] * x[3]) - 0.3 * x[4] ** 2 / x[5]
                 + rng.normal(0.0, 0.02, size=n), 6)
    cols = {"y": y}
    for k in range(7):
        cols["x%d" % (k + 1)] = x[k]
    return dict(df=cols, target="y", features=["x%d" % (k + 1) for k in range(7)],
                groups=[["x1", "x2"], ["x3", "x4"], ["x5"], ["x6", "x7"]],
                ops="(+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(|-|)", fcomplexity=3, dim=3, nf_sis=60)


def deep():
    """Rung 3 (complexity 4): deeper names for the A*B/B checks, and a pruned SIS buffer."""
    rng = np.random.default_rng(2018)
    n = 30
    x = np.round(rng.uniform(0.5, 2.0, size=(3, n)), 4)
    y = np.round((x[0] + x[1]) ** 2 / (x[2] * x[0]) + rng.normal(0.0, 0.01, size=n), 6)
    cols = {"y": y, "a": x[0], "b": x[1], "c": x[2]}
    return dict(df=cols, target="y", features=["a", "b", "c"], groups=[["a", "b"], ["c"]],
                ops="(+)(-)(*)(/)(^2)", fcomplexity=4, dim=2, nf_sis=30)


def fcheavy():
    """Feature construction dominates: 10 features, 13 operators, rung 2."""
    rng = np.random.default_rng(1802)
    n = 100
    x = np.round(rng.uniform(0.3, 2.5, size=(10, n)), 4)
    y = np.round(1.7 * np.exp(-x[0]) * x[1] - 0.4 * np.abs(x[2] - x[3]) / x[4] + rng.normal(0.0, 0.01, size=n), 6)
    cols = {"y": y}
    for k in range(10):
        cols["x%d" % k] = x[k]
    return dict(df=cols, target="y", features=["x%d" % k for k in range(10)],
                groups=[["x0", "x1"], ["x2", "x4"], ["x5", "x6"]],
                ops="(+)(-)(*)(/)(exp)(exp-)(log)(^-1)(^2)(^3)(sqrt)(cbrt)(|-|)", fcomplexity=3, dim=2, nf_sis=100)


CASES = {"example": example, "synthetic": synthetic, "timing": timing, "deep": deep, "fcheavy": fcheavy}


def write_inputs(case, run_dir, dim):
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    feats = case["features"]
    n = len(case["df"][case["target"]])
    with open(os.path.join(run_dir, "train.dat"), "w") as f:
        f.write("materials %s %s\n" % (case["target"], " ".join(feats)))
        for s in range(n):
            f.write("s%d %s %s\n" % (s + 1, repr(float(case["df"][case["target"]][s])),
                                     " ".join(repr(float(case["df"][c][s])) for c in feats)))
    funit = "".join("(%d:%d)" % (feats.index(g[0]) + 1, feats.index(g[-1]) + 1) for g in case["groups"])
    with open(os.path.join(run_dir, "SISSO.in"), "w") as f:
        f.write("ptype=1\nntask=1\nscmt=.false.\ndesc_dim=%d\nnsample=%d\nrestart=0\nfstore=1\nnsf=%d\n"
                "ops='%s'\nfcomplexity=%d\nfunit=%s\nfmax_min=1e-3\nfmax_max=1e5\nnf_sis=%d\nmethod_so='L0'\n"
                "fit_intercept=.true.\nmetric='RMSE'\nnmodel=100\n"
                % (dim, n, len(feats), case["ops"], case["fcomplexity"], funit, case["nf_sis"]))


def run_fortran(binary, run_dir):
    env = dict(os.environ, TMPDIR=os.environ.get("SISSO_TMPDIR", os.path.join(run_dir, "tmp")))
    os.makedirs(env["TMPDIR"], exist_ok=True)
    t = time.time()
    r = subprocess.run("ulimit -s 65520; exec %s" % binary, shell=True, cwd=run_dir, env=env,
                       capture_output=True, text=True)
    wall = time.time() - t
    if r.returncode != 0 or "Have a nice day" not in open(os.path.join(run_dir, "SISSO.out")).read():
        raise RuntimeError("SISSO failed in %s:\n%s" % (run_dir, r.stdout[-2000:] + r.stderr[-2000:]))
    return wall


def parse_out(run_dir):
    out = open(os.path.join(run_dir, "SISSO.out")).read()
    blocks = re.split(r"\nDimension:\s+(\d+)\n", out)[1:]
    phi, sis = [], []
    for k in range(0, len(blocks), 2):
        b = blocks[k + 1]
        sizes = [int(v) for v in re.findall(r"Total number of features in the space Phi\d+:\s+(\d+)", b)]
        m = re.search(r"Size of the SIS-selected subspace from Phi\d+:\s+(\d+)", b)
        phi.append(sizes)
        sis.append(int(m.group(1)) if m else 0)
    desc = re.findall(r"d\d{3} = (\S+)\s+feature_ID:(\d+)", out)
    last = out[out.rindex("descriptor/model"):]
    coef = [float(v) for v in re.search(r"coeff\.\(ci\):(.*)", last).group(1).split()]
    c0 = float(re.search(r"c0:\s+(\S+)", last).group(1))
    rmse, maxae = (float(v) for v in re.search(r"RMSE and MaxAE:\s+(\S+)\s+(\S+)", last).group(1, 2))
    names = [line.split()[0] for line in open(os.path.join(run_dir, "SIS_subspaces", "Uspace.expressions"))]
    return dict(phi=phi, sis=sis, descriptors=[d for d, _ in desc], ids=[int(i) for _, i in desc], coef=coef, c0=c0,
                rmse=rmse, maxae=maxae, subspace=names)


def case_args(case):
    # a group is SISSO's funit range (n1:n2): every feature from its first to its last
    feats = case["features"]
    units = {f: "u%d" % k for k, g in enumerate(case["groups"])
             for f in feats[feats.index(g[0]):feats.index(g[-1]) + 1]}
    return dict(target=case["target"], features=feats, units=units, ops=case["ops"],
                fcomplexity=case["fcomplexity"], dim=case["dim"], nf_sis=case["nf_sis"])


def write_cases(names):
    """study/ports/sisso/sisso.cases.json: the program's parameters per case, for tools/export/check.py."""
    out = {}
    for name in names:
        case = CASES[name]()
        p, _ = sisso.params(case["df"], **case_args(case))
        out[name] = {k: (np.asarray(v).tolist() if not isinstance(v, (int, float)) else v) for k, v in p.items()}
    path = os.path.join(HERE, "sisso.cases.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", os.path.relpath(path, ROOT))


def fit_viv(case):
    return sisso.fit(case["df"], **case_args(case))


def close(a, b):
    return abs(a - b) <= REL * max(abs(a), abs(b), 1e-300) + 1e-12


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--runs", default=os.environ.get("SISSO_RUNS", "/tmp/sisso_runs"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--write-cases", default="", help="write sisso.cases.json for these cases and stop")
    a = ap.parse_args()
    if a.write_cases:
        write_cases(a.write_cases.split(","))
        return
    binary = os.environ.get("SISSO_BIN")
    if not binary:
        sys.exit("set SISSO_BIN to the Fortran SISSO executable")
    ref = {}
    ok = True
    for name in a.cases.split(","):
        case = CASES[name]()
        D = case["dim"]
        fortran = []
        for d in range(1, D + 1):
            rd = os.path.join(a.runs, "%s_D%d" % (name, d))
            write_inputs(case, rd, d)
            run_fortran(binary, rd)
            fortran.append(parse_out(rd))
        t_f = min(run_fortran(binary, os.path.join(a.runs, "%s_D%d" % (name, D))) for _ in range(a.repeat))
        r = fit_viv(case)
        t_v = []
        for _ in range(a.repeat):
            t = time.time()
            r = fit_viv(case)
            t_v.append(time.time() - t)
        t_v = min(t_v)
        full = fortran[-1]
        checks = []
        viv_phi = [p for p in r.phi_sizes if p[0] != -1]
        f_phi = [p for p in full["phi"] if p]
        checks.append(("Phi sizes", viv_phi == f_phi, "%s vs %s" % (viv_phi, f_phi)))
        checks.append(("SIS sizes", [s for s in r.sis_sizes if s] == [s for s in full["sis"] if s],
                       "%s vs %s" % (r.sis_sizes, full["sis"])))
        checks.append(("subspace names in order", r.subspace == full["subspace"],
                       "%d names" % len(full["subspace"])))
        for d in range(1, D + 1):
            f, m = fortran[d - 1], r.models[d - 1]
            same_desc = m.descriptors == f["descriptors"] and [i + 1 for i in m.indices] == f["ids"]
            checks.append(("%dD descriptors" % d, same_desc, "%s vs %s" % (m.descriptors, f["descriptors"])))
            nums = all(close(x, y) for x, y in zip(m.coefficients + [m.intercept, m.rmse, m.maxae],
                                                   f["coef"] + [f["c0"], f["rmse"], f["maxae"]]))
            worst = max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(m.coefficients + [m.intercept, m.rmse],
                                                                         f["coef"] + [f["c0"], f["rmse"]]))
            checks.append(("%dD coefficients, c0, RMSE, MaxAE" % d, nums, "worst relative %.1e" % worst))
        print("== %s: Fortran %.3f s, Viv %.3f s (best of %d)" % (name, t_f, t_v, a.repeat))
        for what, good, detail in checks:
            ok &= good
            print("  %-36s %s  %s" % (what, "same" if good else "DIFFERENT", detail))
        ref[name] = dict(fortran=fortran, seconds_fortran=t_f, seconds_viv=t_v)
    if a.record:
        path = os.path.join(HERE, "fortran_reference.json")
        with open(path, "w") as f:
            json.dump(ref, f, indent=1)
            f.write("\n")
        print("wrote", os.path.relpath(path, ROOT))
    print("ALL SAME" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

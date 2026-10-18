# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2017 Lawrence Livermore National Security and San Diego State University Research Foundation
# SPDX-FileCopyrightText: 2026 The Viv authors
"""lib/nuclear/shell against BIGSTICK: energies, spins, basis dimensions, times.

    python3 lib/nuclear/shell/bigstick_compare.py --bigstick PATH/bigstick.x [--cases ne20,mg24] [--record]

For each case it runs BIGSTICK (github.com/cwjsdsu/BigstickPublick, MIT
license) on the same .sps and .int files the Viv program ships, and the Viv
program through vipyr twice: the exact build (the interpreter's bits) and
the fast build (fused multiply-add, the platform libm). Each is run once to
warm up and timed on a second run. It prints one row per case and, with
--record, writes the BIGSTICK numbers to bigstick_reference.json beside this
file, which test_shellmodel.py reads, so the Python test does not need a
Fortran build.

BIGSTICK keeps its Hamiltonian's matrix elements in single precision (REAL(4)
in its jump storage), and its default build keeps the Lanczos vectors in
single precision too. The oracle build sets lanc_prec = 8 (Lanczos vectors in
double precision; one declaration in boutputlib.f90 had to follow it) and
prints energies with 10 decimals and spins with 7. Its energies then carry
the rounding of single-precision matrix elements, about 1e-7 relative, so
the energies agree to a few 1e-5 MeV and not better. A full double-precision
build (-fdefault-real-8 and REAL(4) promoted to REAL(8)) did not run: its
Lanczos setup stops with a bus error, so it is not used.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "python"))

# name: (interaction, sps, int, z, n, core A, A0, x, nstates)
CASES = {
    "ne20": ("usdb", "sd", "usdb", 2, 2, 16, 18.0, 0.3, 10),
    "na22": ("usdb", "sd", "usdb", 3, 3, 16, 18.0, 0.3, 10),
    "mg24": ("usdb", "sd", "usdb", 4, 4, 16, 18.0, 0.3, 10),
    "mg25": ("usdb", "sd", "usdb", 4, 5, 16, 18.0, 0.3, 10),
    "si28": ("usdb", "sd", "usdb", 6, 6, 16, 18.0, 0.3, 10),
    "c12": ("ckpot", "p", "ckpot", 4, 4, 4, 1.0, 0.0, 10),
    "b10": ("ckpot", "p", "ckpot", 3, 3, 4, 1.0, 0.0, 10),
}

ITERATIONS = 1000  # BIGSTICK 'ld' (its standard convergence test) may take up to this many steps


def bigstick_input(name, c):
    _, sps, tb, z, n, core, a0, x, nk = c
    m2 = (z + n) % 2
    return "\n".join([
        "n", name, sps, f"{z} {n}", f"{m2}", "n", tb,
        f"1.0 {a0} {core + z + n} {x}", "end", "ld", f"{nk} {ITERATIONS}", ""])


def run_bigstick(exe, name, c):
    """Energies, spins, dimension and wall time of a BIGSTICK run, warm."""
    work = Path(tempfile.mkdtemp(prefix=f"bs_{name}_"))
    for f in (f"{c[1]}.sps", f"{c[2]}.int"):
        (work / f).write_text((HERE / "data" / f).read_text())
    (work / "in").write_text(bigstick_input(name, c))
    secs = []
    for _ in range(2):
        t = time.perf_counter()
        with open(work / "in") as fin:
            log = subprocess.run([exe], stdin=fin, cwd=work, capture_output=True, text=True).stdout
        secs.append(time.perf_counter() - t)
    dim = int(next(l for l in log.splitlines() if "Total basis =" in l).split("=")[1])
    lines = (work / f"{name}.res").read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.strip().startswith("State"))
    e, j = [], []
    for l in lines[k + 1:]:
        f = l.split()
        if len(f) < 4 or not f[0].isdigit():
            break
        e.append(float(f[1]))
        j.append(float(f[3]))
    return {"dim": dim, "energies": e, "j": j, "seconds": secs[-1]}


def run_viv(mod, c):
    kw = dict(interaction=c[0], z=c[3], n=c[4], nstates=c[8], tol=1e-8)
    mod.run(**kw)
    t = time.perf_counter()
    r = mod.run(**kw)
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigstick", default=os.environ.get("BIGSTICK"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    import vipyr
    exact = vipyr.load(HERE / "shellmodel.viv", name="shellmodel_exact")
    fast = vipyr.load(HERE / "shellmodel.viv", fast=True, name="shellmodel_fast")
    ref_path = HERE / "bigstick_reference.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    print(f"{'case':6} {'dim':>7} {'E1 (Viv)':>14} {'max|dE|':>9} {'max|dJ|':>9} {'iters':>6} {'BIGSTICK s':>10} {'exact s':>8} {'fast s':>8}")
    for name in a.cases.split(","):
        c = CASES[name]
        if a.bigstick:
            b = run_bigstick(a.bigstick, name, c)
            ref[name] = b
        else:
            b = ref[name]
        r, te = run_viv(exact, c)
        rf, tf = run_viv(fast, c)
        n = min(len(b["energies"]), len(r["energies"]))
        de = max(abs(b["energies"][i] - r["energies"][i]) for i in range(n))
        dj = max(abs(b["j"][i] - r["j"][i]) for i in range(n))
        dfe = max(abs(rf["energies"][i] - r["energies"][i]) for i in range(n))
        assert b["dim"] == r["dim"], (name, b["dim"], r["dim"])
        print(f"{name:6} {r['dim']:7d} {r['energies'][0]:14.8f} {de:9.1e} {dj:9.1e} {r['iterations']:6d} "
              f"{b['seconds']:10.3f} {te:8.3f} {tf:8.3f}   fast-exact {dfe:.1e}")
    if a.record:
        ref_path.write_text(json.dumps(ref, indent=1) + "\n")


if __name__ == "__main__":
    main()

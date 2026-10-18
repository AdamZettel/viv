# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""lib/skyrme/spherical.viv (through the exported skyrmehf) against the numpy
reference lib/skyrme/refs/skyrme_hf.py, nucleus by nucleus and force by force.

    viv export lib/skyrme/skyrmehf.export.toml -o target/export/skyrmehf
    python3 lib/skyrme/refs/compare.py [--spacing 0.1,0.05] [--json out.json]

The reference is the personal site's solver with its constants set to the
ones lib/skyrme/skyrme uses (skyrme_rpa's): hbar^2/2m from hbar c =
197.327053 MeV fm and the mean nucleon mass, times 1 - 1/A in the
single-particle equation and the energy alike; e^2 = 1.43986 MeV fm; the
charge radius from r_p^2 + 1.5 (0.65 fm)^2. Its forces are written out
below; SIII goes through its genuine three-body term 1/4 t3 rho_n rho_p rho,
which Viv writes as the density-dependent term with x3 = 1, alpha = 1.

What differs between the two is the method, not the functional: the
reference diagonalizes each (l, j) block densely (numpy.linalg.eigh) and
fills shells in the spectroscopic order it is given; Viv takes the occupied
orbitals by energy and finds them by bisection and inverse iteration. Both
iterate to an energy change below 1e-10 MeV here, so the differences are
the solvers' round-off and the stopping point.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
# one thread: the reference's matrices are small, and a threaded BLAS spends
# its time on contention (and loads every core of a shared machine)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(ROOT, "target", "export", "skyrmehf"))
import numpy as np  # noqa: E402
import skyrme_hf as ref  # noqa: E402
import skyrmehf  # noqa: E402

HBARC = 197.327053
M_NUC = 0.5 * (938.27231 + 939.56563)
E2 = 1.43986
FOLD = 0.65
MIX = 0.5
E_TOL = 1.0e-10
MAX_ITER = 600

NUCLEI = {"16O": (8, 8), "40Ca": (20, 20), "48Ca": (20, 28), "56Ni": (28, 28),
          "90Zr": (40, 50), "132Sn": (50, 82), "208Pb": (82, 126)}

FORCES = {
    "SLy4": ref.Skyrme("SLy4", t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0, x0=0.834, x1=-0.344, x2=-1.0,
                       x3=1.354, W0=123.0, gamma=1 / 6, three_body=False, j2_terms=False),
    "SLy5": ref.Skyrme("SLy5", t0=-2484.88, t1=483.13, t2=-549.40, t3=13763.0, x0=0.778, x1=-0.328, x2=-1.0,
                       x3=1.267, W0=126.0, gamma=1 / 6, three_body=False, j2_terms=True),
    "SIII": ref.Skyrme("SIII", t0=-1128.75, t1=395.0, t2=-95.0, t3=14000.0, x0=0.45, W0=120.0,
                       three_body=True, j2_terms=True),
    "SkM*": ref.Skyrme("SkM*", t0=-2645.0, t1=410.0, t2=-135.0, t3=15595.0, x0=0.09, x1=0.0, x2=0.0, x3=0.0,
                       W0=130.0, gamma=1 / 6, three_body=False, j2_terms=False),
}


def box_of(a):
    """skyrmehf.viv's default box."""
    return max(15.0, 1.2 * a ** (1.0 / 3.0) + 11.5)


def shells(count):
    """The first orbitals of the reference's spectroscopic order holding `count`."""
    out, held = [], 0
    for n, l, j2 in ref._FILL["s126"]:
        if held == count:
            break
        out.append(ref.Orbital(n, l, j2))
        held += j2 + 1
    assert held == count, count
    return out


def run_ref(name, force, h, box):
    z, n = NUCLEI[name]
    a = z + n
    ref.HBAR2_2M = HBARC * HBARC / (2.0 * M_NUC) * (1.0 - 1.0 / a)
    ref.E2 = E2
    nuc = ref.Nucleus(name, z, n, shells(z), shells(n))
    t = time.time()
    res = ref.run_hf(FORCES[force], nuc, grid=ref.Grid(rmax=box, h=h), maxiter=MAX_ITER, mix=MIX, tol=E_TOL,
                     com_correction=False, verbose=False)
    dt = time.time() - t
    levels = {}
    for tag, lev in (("n", res["levels_n"]), ("p", res["levels_p"])):
        for o, e in lev:
            levels[tag + " " + o.label()] = float(e)
    return dict(e_per_a=-res["BE_per_A"], r_p=res["rms_p"], r_n=res["rms_n"],
                r_ch=math.sqrt(res["rms_p"] ** 2 + 1.5 * FOLD ** 2), levels=levels, iterations=res["iters"],
                seconds=dt)


def run_viv(name, force, h, box):
    t = time.time()
    # linear mixing, as the numpy solver mixes: the same iteration on both sides
    r = skyrmehf.run(nucleus=name, force=force, spacing=h, box=box, mix=MIX, mixing="linear", e_tol=E_TOL,
                     max_iter=MAX_ITER)
    dt = time.time() - t
    assert r["converged"], (name, force, h)
    levels = dict(zip(r["labels"].split(","), (float(x) for x in r["levels"])))
    return dict(e_per_a=r["e_per_a"], r_p=r["r_p"], r_n=r["r_n"], r_ch=r["r_ch"], levels=levels,
                iterations=r["iterations"], refills=r["refills"], seconds=dt,
                spectrum=dict(zip(r["spectrum_labels"].split(","), (float(x) for x in r["spectrum"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", default="0.1,0.05")
    ap.add_argument("--nuclei", default=",".join(NUCLEI))
    ap.add_argument("--forces", default=",".join(FORCES))
    ap.add_argument("--json")
    args = ap.parse_args()
    out = []
    for force in args.forces.split(","):
        for name in args.nuclei.split(","):
            z, n = NUCLEI[name]
            box = box_of(z + n)
            for h in (float(x) for x in args.spacing.split(",")):
                v = run_viv(name, force, h, box)
                p = run_ref(name, force, h, box)
                assert set(v["levels"]) == set(p["levels"]), (name, force, sorted(v["levels"]), sorted(p["levels"]))
                dlev = max(abs(v["levels"][k] - p["levels"][k]) for k in v["levels"])
                row = dict(force=force, nucleus=name, h=h, box=box, viv=v, ref=p,
                           d_e_per_a=abs(v["e_per_a"] - p["e_per_a"]),
                           d_r=max(abs(v[k] - p[k]) for k in ("r_p", "r_n", "r_ch")), d_levels=dlev)
                out.append(row)
                print(f"{force:5s} {name:6s} h={h:<5} E/A {v['e_per_a']:.8f} ref {p['e_per_a']:.8f} "
                      f"|dE/A| {row['d_e_per_a']:.1e} |dr| {row['d_r']:.1e} |dlev| {dlev:.1e} "
                      f"r_ch {v['r_ch']:.5f} it {v['iterations']}/{p['iterations']} refills {v['refills']} "
                      f"t {v['seconds']:.3f}/{p['seconds']:.2f} s", flush=True)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

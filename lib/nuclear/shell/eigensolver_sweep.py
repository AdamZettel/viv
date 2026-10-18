# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The shell model's eigensolver (the lowest_eigenpairs contract), swept: Lanczos and Householder + QL on the same nuclei.

    python3 lib/nuclear/shell/eigensolver_sweep.py [--dense-max 7000] [--record]

For each case it runs lib/nuclear/shell/shellmodel.viv through vipyr (the
exact build) with eigensolver="lanczos" and,
when the basis has at most --dense-max states, "householder_ql", each once to warm
up the build and
once timed. It prints the basis dimension, each solver's wall time for the
whole run (the basis, the Hamiltonian's pieces, the solve, J and the
occupations), the largest energy difference between the two solvers, and the
largest difference from BIGSTICK where bigstick_reference.json has the case.
With --record it writes lowest_eigenpairs_sweep.json beside this file,
which the code map and the gallery read.

The two solvers share `apply_h`: householder_ql forms H column by column
from it. Their
agreement checks the eigensolvers, not the Hamiltonian; the comparison with
BIGSTICK checks the Hamiltonian.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
import vipyr  # noqa: E402

REF = json.loads((HERE / "bigstick_reference.json").read_text())
# name: (interaction, valence protons, valence neutrons)
CASES = {
    "c12": ("ckpot", 4, 4),
    "b10": ("ckpot", 3, 3),
    "o20": ("usdb", 0, 4),
    "f20": ("usdb", 1, 3),
    "ne20": ("usdb", 2, 2),
    "ne21": ("usdb", 2, 3),
    "ne22": ("usdb", 2, 4),
    "na22": ("usdb", 3, 3),
    "mg24": ("usdb", 4, 4),
}
NSTATES = 10
TOL = 1e-8


def timed(sm, **kw):
    sm.run(**kw)
    t = time.perf_counter()
    r = sm.run(**kw)
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-max", type=int, default=7000)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    sm = vipyr.load(HERE / "shellmodel.viv")
    rows = []
    print("%-6s %7s %10s %10s %12s %12s" % ("case", "dim", "lanczos s", "QL s", "|dE| max", "vs BIGSTICK"))
    for name in a.cases.split(","):
        inter, z, n = CASES[name]
        kw = dict(interaction=inter, z=z, n=n, nstates=NSTATES, tol=TOL)
        lz, t_l = timed(sm, eigensolver="lanczos", **kw)
        row = {"case": name, "interaction": inter, "z": z, "n": n, "dim": int(lz["dim"]),
               "lanczos": {"seconds": t_l, "products": int(lz["iterations"])}}
        k = len(lz["energies"])
        if name in REF:
            ref = np.array(REF[name]["energies"][:k])
            row["lanczos"]["vs_bigstick"] = float(np.max(np.abs(lz["energies"] - ref)))
        if lz["dim"] <= a.dense_max:
            dn, t_d = timed(sm, eigensolver="householder_ql", **kw)
            row["householder_ql"] = {"seconds": t_d, "products": int(dn["iterations"])}
            row["agreement"] = float(np.max(np.abs(lz["energies"] - dn["energies"])))
            if name in REF:
                row["householder_ql"]["vs_bigstick"] = float(np.max(np.abs(dn["energies"] - ref)))
        rows.append(row)
        d = row.get("householder_ql", {})
        print("%-6s %7d %10.3f %10s %12s %12s" % (
            name, row["dim"], t_l, "%.3f" % d["seconds"] if d else "-",
            "%.1e" % row["agreement"] if "agreement" in row else "-",
            "%.1e" % row["lanczos"]["vs_bigstick"] if "vs_bigstick" in row["lanczos"] else "-"), flush=True)
    if a.record:
        out = {
            "contract": "lowest_eigenpairs",
            "program": "lib/nuclear/shell/shellmodel.viv",
            "nstates": NSTATES, "tol": TOL,
            "seconds": "wall time of the whole run, exact build, second of two runs",
            "load": os.getloadavg()[0],
            "reference": "BIGSTICK 8.0.0, lib/nuclear/shell/bigstick_reference.json",
            "rows": rows,
        }
        (HERE / "lowest_eigenpairs_sweep.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Power flow convergence and timing from 9 to 9241 buses
======================================================

Newton-Raphson against the fast-decoupled method on MATPOWER cases from 9 to
2383 buses: the mismatch left after each iteration, the iterations each
needs, and the time per solve measured against an independent SciPy power
flow, up to the 9241-bus PEGASE model of the European grid.
"""

# %%
# ## The mismatch, iteration by iteration
#
# `max_it=k` stops the solver after k iterations, and `mismatch` is the
# largest power imbalance left at any bus, in per unit of the 100 MVA base.
# Newton's method roughly squares the error each step; the fast-decoupled
# method, whose matrices stay fixed, gains a constant factor per step. Each
# line ends at the iteration where the mismatch first falls below the
# tolerance, 1e-8.

import json

import numpy as np
import vipyr as vp
from vipyr.figure import figure

cases = ["case30", "case118", "case300", "case2383wp"]
hist = {}
for name in cases:
    for method, top in (("nr", 6), ("fdxb", 18)):
        ms = []
        for k in range(top + 1):
            r = vp.power.runpf(name, method=method, max_it=k)
            ms.append(r.mismatch)
            if r.success:
                break
        hist[name, method] = ms
for name in cases:
    print("%-10s nr  " % name, " ".join("%.0e" % m for m in hist[name, "nr"]))
print("case2383wp fdxb", " ".join("%.0e" % m for m in hist["case2383wp", "fdxb"]))

fig = figure("Largest mismatch after k iterations", "iteration k", "mismatch (per unit)")
fig.yscale("log")
for name in ("case118", "case2383wp"):
    for method in ("nr", "fdxb"):
        m = np.array(hist[name, method])
        fig.plot(np.arange(len(m)), m, "%s %s" % (name, method))
fig.hline("tolerance", 1e-8)

# %%
# ## Iterations and time per solve
#
# The table comes from `lib/power/compare.py`, which runs every case by both
# libraries and keeps the best of several runs; it was measured on an Apple
# M2 Pro under load from other jobs, so the times are indicative.
# Viv is the compiled solver called through vipyr; the SciPy reference
# builds the same sparse matrices with scipy.sparse and solves with SuperLU.
# case9241pegase is not shipped with vipyr (it is 1.5 MB); the comparison
# downloads it from the MATPOWER repository.

bench = json.load(open("lib/power/compare.json"))
rows = [r for r in bench["cases"]]
print("%-15s %6s  %-4s %5s  %10s  %10s" % ("case", "buses", "", "iter", "Viv (ms)", "SciPy (ms)"))
for r in rows:
    print("%-15s %6d  %-4s %5d  %10.2f  %10s" % (r["case"], r["nb"], r["method"], r["iterations"], 1e3 * r["t_viv"],
                                                 "%.2f" % (1e3 * r["t_ref"]) if r["method"] != "fdxb" else ""))

fig = figure("Time per power flow", "buses", "seconds")
fig.xscale("log")
fig.yscale("log")
for method, label in (("nr", "Viv NR"), ("fdxb", "Viv FDXB"), ("dc", "Viv DC")):
    sel = [r for r in rows if r["method"] == method]
    fig.plot([r["nb"] for r in sel], [r["t_viv"] for r in sel], label)
sel = [r for r in rows if r["method"] == "nr"]
fig.plot([r["nb"] for r in sel], [r["t_ref"] for r in sel], "SciPy NR")

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Special functions against mpmath, SciPy and R
=============================================

Every distribution in `vp.stats` reduces to a handful of special functions:
log-gamma, the error function family, and the regularized incomplete gamma
and beta functions with their inverses. `lib/stats/accuracy.py` evaluates
each on a grid that runs into its tails, against mpmath at 50 digits, and
evaluates SciPy 1.12 and R 4.3.3 on the same grid. The error of a value is
counted in units in the last place (ulps) of the correctly rounded result.
"""

# %%
# ## A few values, checked now
#
# The far lower tail of the normal distribution, the 97.5% point of
# Student's t with 10 degrees of freedom, and an upper incomplete gamma
# value of order 1e-15, each against mpmath at 50 digits.

import json

import mpmath as mp
import numpy as np
import scipy.special as sc
import vipyr as vp
from vipyr.figure import figure

mp.mp.dps = 50
checks = [
    ("ndtr(-37.5)", vp.stats.ndtr(-37.5), sc.ndtr(-37.5), mp.ncdf(-37.5)),
    (
        "gammaincc(50, 120)",
        vp.stats.gammaincc(50.0, 120.0),
        sc.gammaincc(50.0, 120.0),
        mp.gammainc(50, 120, mp.inf, regularized=True),
    ),
    (
        "betainc(0.5, 30, 0.2)",
        vp.stats.betainc(0.5, 30.0, 0.2),
        sc.betainc(0.5, 30.0, 0.2),
        mp.betainc(0.5, 30, 0, 0.2, regularized=True),
    ),
]
print("%-22s %24s %10s %10s" % ("", "vp.stats", "ulps", "SciPy ulps"))
for name, v, s, ref in checks:
    sp = np.spacing(float(ref))
    print(
        "%-22s %24.17g %10.1f %10.1f"
        % (name, v, float(abs(v - ref)) / sp, float(abs(s - ref)) / sp)
    )

# %%
# ## The whole grid
#
# The worst error of each function over its grid (read from
# `lib/stats/accuracy.json`, measured with the commands in that file's
# header). R has no error-function family of its own; its column is empty
# there. The incomplete gamma and beta functions lose digits in their deep
# tails, where the result is exp of an exponent of order hundreds: the
# exponent's own rounding, a few hundred ulps, is the error, the same
# mechanism in all three libraries.

acc = json.load(open("lib/stats/accuracy.json"))["functions"]


def worst(entry):
    return "-" if entry is None else "%.3g" % entry["max_ulps"]


print("%-13s %6s %10s %10s %10s" % ("", "points", "vp.stats", "SciPy", "R"))
for name, row in acc.items():
    print(
        "%-13s %6d %10s %10s %10s"
        % (
            name,
            row["points"],
            worst(row["vipyr"]),
            worst(row["scipy"]),
            worst(row["R"]),
        )
    )

# %%
# ## The map
#
# The worst error per function (1 is one ulp; an exact result is drawn at
# 0.5), one marker per library, functions sorted by vp.stats's error; a
# missing marker is a function the library does not have.

names = sorted(acc, key=lambda n: (acc[n]["vipyr"] or {"max_ulps": 0})["max_ulps"])
fig = figure("Worst error against mpmath", "worst ulps", "")
for who, mark in [("vipyr", "s1 circle"), ("scipy", "s2 square"), ("R", "s3 cross")]:
    xs, ys = [], []
    for i, n in enumerate(names):
        e = acc[n][who]
        if e is not None:
            xs.append(max(e["max_ulps"], 0.5))
            ys.append(i)
    lab = {"vipyr": "vp.stats"}.get(who, who)
    fig.scatter(xs, ys, label=lab)
    fig.style(lab, mark)
for i, n in enumerate(names):
    fig.ytick(n, i)
fig.xscale("log")
fig

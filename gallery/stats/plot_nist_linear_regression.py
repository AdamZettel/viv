# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
NIST's certified linear regressions, digit by digit
===================================================

NIST's Statistical Reference Datasets include eleven linear least-squares
problems with coefficients, standard errors, residual standard deviation and
R^2 certified to 15 digits, from the trivial (Norris, a line) to the
notorious (Filip, a degree-10 polynomial; Wampler4-5, degree 5 with large
residuals). `vp.stats.ols` fits each with every solver of its slot, and the
table gives the certified digits of the worst coefficient (NIST's log
relative error, LRE) beside statsmodels 0.14 and R 4.3.3's lm.
"""

# %%
# ## One fit, checked now
#
# Longley's macroeconomic data: six correlated predictors and an intercept.

import json
import sys

import numpy as np
import vipyr as vp
from vipyr.figure import figure

sys.path.insert(0, "lib/stats")
import nist_lls as N  # noqa: E402

y, X, cert = N.read("Longley")
r = vp.stats.ols(y, X, names=["const"] + ["x%d" % k for k in range(1, 7)])
print(r.summary())
print(
    "certified digits: coefficients %.1f, standard errors %.1f, R^2 %.1f"
    % (
        N.lre(r.coef, cert["coef"]).min(),
        N.lre(r.bse, cert["se"]).min(),
        N.lre(r.rsquared, cert["r2"]),
    )
)

# %%
# ## All eleven datasets
#
# Measured by `python3 lib/stats/nist_lls.py` (`lib/stats/nist_lls.json`).
# "qr" is the default: pivoted Householder QR on unit-norm columns and two
# steps of Bjorck's refinement of the augmented system, with both of its
# residuals summed exactly. "attainable" is the exact least-squares solution
# of the design matrix as stored in doubles (mpmath, 60 digits): for Filip,
# x^10 rounded to double has already lost the digits past 7.6, and the QR
# reaches that solution to 14.4 digits.

table = json.load(open("lib/stats/nist_lls.json"))["datasets"]
cols = [
    "vipyr qr",
    "vipyr qr, no refinement",
    "vipyr cholesky",
    "vipyr svd",
    "statsmodels",
    "R lm",
    "attainable",
]
short = ["qr", "qr-1", "chol", "svd", "statsm", "R lm", "attain"]
print("%-9s" % "" + "".join("%8s" % s for s in short))
for name, row in table.items():
    print("%-9s" % name + "".join("%8.1f" % row[c]["coef"] for c in cols))

# %%
# ## The picture
#
# Certified digits of the worst coefficient, per dataset.

names = list(table)
fig = figure("Certified digits of the worst coefficient", "dataset", "LRE")
for k, (c, lab) in enumerate(
    [
        ("vipyr qr", "vp.stats qr"),
        ("statsmodels", "statsmodels"),
        ("R lm", "R lm"),
    ]
):
    fig.bars(
        list(range(len(names))),
        [table[n][c]["coef"] for n in names],
        label=lab,
    )
for i, n in enumerate(names):
    fig.xtick(n, i)
fig

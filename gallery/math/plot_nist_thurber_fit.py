# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Thurber's electron mobility, a NIST certified fit
=================================================

NIST's Statistical Reference Datasets give nonlinear regression problems
with parameters certified to eleven digits. Thurber (higher difficulty) is
a rational function of degree three over three, seven parameters, fitted
to 37 measurements of electron mobility against the log of density.
`vp.optimize.curve_fit` fits it from both of NIST's starting points with
MINPACK's Levenberg-Marquardt and an exact Jacobian, and the table compares
every parameter and standard deviation with the certified value.
"""

# %%
# ## The model and the data
#
# The model is NIST's, written for SciPy in `lib/minpack/nist_strd.py`:
#
#     def rational3(x, b1, b2, b3, b4, b5, b6, b7):
#         return (b1 + b2 * x + b3 * x ** 2 + b4 * x ** 3) / (1.0 + b5 * x + b6 * x ** 2 + b7 * x ** 3)
#
# py2viv translates it to Viv once; the driver differentiates it with
# reverse-mode `grad` at each point, so the Jacobian has no step-size error.

import json
import sys

import numpy as np
import vipyr as vp
from vipyr.figure import figure

sys.path.insert(0, "lib/minpack")
import nist_strd as N  # noqa: E402

p = N.load("Thurber")
print(
    "Thurber: %d observations, %d parameters, %s difficulty"
    % (len(p.y), len(p.certified), p.difficulty)
)
fits = [
    vp.optimize.curve_fit(N.rational3, p.x, p.y, start) for start in p.starts
]
for k, r in enumerate(fits, 1):
    print(
        "start %d: %3d evaluations, %3d Jacobians, MINPACK info %d, residual "
        "sum of squares %.10e" % (k, r.nfev, r.njev, r.info, r.rss)
    )
print("certified residual sum of squares          %.10e" % p.rss)

# %%
# ## Against the certified values
#
# The digits that agree are NIST's log relative error,
# LRE = -log10(|value - certified| / |certified|). With leastsq's defaults
# lmder stops when the sum of squares falls by less than 1.5e-8 relative
# (MINPACK info 1). Thurber's parameters are strongly correlated, so a sum
# of squares right to nine digits leaves them right to about five. NIST
# counts four or more digits as a successful fit.

r = fits[0]
lre, lre_sd = N.lre(r.popt, p.certified), N.lre(r.perr, p.cert_sd)
print(
    "     %22s %22s %5s   %16s %16s %5s"
    % ("certified", "vp.optimize", "LRE", "certified sd", "vp sd", "LRE")
)
for i in range(len(p.certified)):
    print(
        "b%d  %22.10e %22.10e %5.1f   %16.10e %16.10e %5.1f"
        % (
            i + 1,
            p.certified[i],
            r.popt[i],
            lre[i],
            p.cert_sd[i],
            r.perr[i],
            lre_sd[i],
        )
    )

# %%
# ## Tighter tolerances, more digits
#
# `ftol` and `xtol` are leastsq's. Each two decades smaller buys about one
# certified digit, for a handful of extra evaluations.

for tol in (1e-10, 1e-12, 1e-14):
    for k, start in enumerate(p.starts, 1):
        t = vp.optimize.curve_fit(
            N.rational3, p.x, p.y, start, ftol=tol, xtol=tol
        )
        print(
            "ftol = xtol = %.0e, start %d: %2d evaluations, LRE of "
            "parameters %.1f, of standard deviations %.1f"
            % (
                tol,
                k,
                t.nfev,
                N.lre(t.popt, p.certified).min(),
                N.lre(t.perr, p.cert_sd).min(),
            )
        )

# %%
# ## Against SciPy
#
# SciPy's `curve_fit` calls the same Fortran (lmdif, with differences), so
# with `jac="fd"` the two take the same steps. The counts and the
# certified digits of each are from `lib/minpack/bench_optimize.py`, over
# all 27 NIST problems; the times are for one warm fit and depend on the
# machine and its load, recorded with them.

bench = json.load(open("lib/minpack/bench_optimize.json"))
print(
    "measured on",
    bench["machine"],
    "| load average",
    bench["load_average_start"],
    "| SciPy",
    bench["scipy"],
)
print(
    "%-9s %5s  %22s  %22s  %22s"
    % (
        "problem",
        "start",
        "vp (grad): LRE nfev ms",
        "vp (fd): LRE nfev ms",
        "SciPy (fd): LRE nfev ms",
    )
)


def cell(c):
    return (
        "%4.1f %5d %7.2f" % (c["lre"], c["nfev"], 1e3 * c["s"])
        if c["ok"]
        else "%22s" % "did not converge"
    )


for row in bench["nist"]:
    if row["difficulty"] == "higher":
        print(
            "%-9s %5d  %22s  %22s  %22s"
            % (
                row["problem"],
                row["start"],
                cell(row["vp"]),
                cell(row["vp_fd"]),
                cell(row["scipy"]),
            )
        )

# %%
# ## The fit and its residuals
#
# The curve through the data, and the residuals y - f(x) against x. NIST's
# residual standard deviation for Thurber is 13.71; the fit's is printed
# below it.

xs = np.linspace(p.x.min(), p.x.max(), 200)
fig = figure(
    "Thurber: electron mobility against log density", "log density", "mobility"
)
fig.scatter(p.x, p.y, "NIST data")
fig.plot(xs, N.rational3(xs, *r.popt), "vp.optimize fit")
fig.style("vp.optimize fit", "s2 nomarker")

res = p.y - N.rational3(p.x, *r.popt)
print(
    "residual standard deviation %.4f (NIST, certified: 1.3714600784E+01)"
    % np.sqrt(res @ res / (len(p.y) - len(p.certified)))
)
fig2 = figure("Residuals of the fit", "log density", "y - f(x)")
fig2.scatter(p.x, res, "residual")
fig2.hline("zero", 0.0)

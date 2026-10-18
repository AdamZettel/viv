# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""NIST StRD's linear least-squares datasets: certified digits by package.

    python3 lib/stats/nist_lls.py        # writes lib/stats/nist_lls.json

Each dataset (lib/stats/data/<name>.dat, from
https://www.itl.nist.gov/div898/strd/lls/lls.shtml) is fitted by
vp.stats.ols with each solver of its slot, by statsmodels 0.14's OLS
(pinv, its default) and by R 4.3.3's lm (QR, LINPACK dqrdc2, tol 1e-7). The
log relative error LRE = -log10(|value - certified| / |certified|) (or
-log10 |value| when the certified value is 0), capped at 15 (NIST
certifies 15 digits), is taken for every coefficient, every standard
error, the residual standard deviation and R^2; the table keeps the
smallest of each group. A dropped coefficient (R's NA) has LRE 0.
"attainable" is the LRE of the exact least-squares solution of the design
matrix as stored in doubles: Filip's x^10 has already lost the digits the
certified values need (the exact solution of the stored matrix agrees with
NIST to 7.6 digits only).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
CAP = 15.0
DATASETS = [
    "Norris",
    "Pontius",
    "NoInt1",
    "NoInt2",
    "Filip",
    "Longley",
    "Wampler1",
    "Wampler2",
    "Wampler3",
    "Wampler4",
    "Wampler5",
]
# polynomial degree in x, and whether the model has an intercept
POLY = {
    "Norris": 1,
    "Pontius": 2,
    "NoInt1": 1,
    "NoInt2": 1,
    "Filip": 10,
    "Wampler1": 5,
    "Wampler2": 5,
    "Wampler3": 5,
    "Wampler4": 5,
    "Wampler5": 5,
}
NO_INTERCEPT = {"NoInt1", "NoInt2"}


def read(name):
    """(y, X, certified) from a StRD file: X with the model's columns."""
    lines = (HERE / "data" / (name + ".dat")).read_text().splitlines()
    head = "\n".join(lines[:10])
    d0, d1 = map(
        int, re.search(r"Data\s+\(lines (\d+) to (\d+)\)", head).groups()
    )
    rows = np.array(
        [[float(t) for t in ln.split()] for ln in lines[d0 - 1 : d1]]
    )
    cert = {"coef": [], "se": []}
    for ln in lines:
        m = re.match(r"\s+B(\d+)\s+(\S+)\s+(\S+)", ln)
        if m:
            cert["coef"].append(float(m.group(2)))
            cert["se"].append(float(m.group(3)))
        m = re.match(r"\s+Standard Deviation\s+(\S+)", ln)
        if m:
            cert["resid_sd"] = float(m.group(1))
        m = re.match(r"\s+R-Squared\s+(\S+)", ln)
        if m:
            cert["r2"] = float(m.group(1))
    y = rows[:, 0]
    if name == "Longley":
        X = np.column_stack([np.ones(len(y)), rows[:, 1:]])
    else:
        x = rows[:, 1]
        cols = [x**k for k in range(POLY[name] + 1)]
        X = np.column_stack(cols[1:] if name in NO_INTERCEPT else cols)
    return y, X, cert


def lre(v, c):
    v, c = np.asarray(v, dtype=float), np.asarray(c, dtype=float)
    out = np.zeros(v.shape)
    for i, (a, b) in enumerate(zip(v.ravel(), c.ravel())):
        if not np.isfinite(a):
            out.flat[i] = 0.0
            continue
        err = abs(a - b) / abs(b) if b != 0 else abs(a)
        out.flat[i] = CAP if err == 0 else min(CAP, max(0.0, -np.log10(err)))
    return out


def score(coef, se, sd, r2, cert):
    return dict(
        coef=float(lre(coef, cert["coef"]).min()),
        se=float(lre(se, cert["se"]).min()),
        resid_sd=float(lre(sd, cert["resid_sd"])),
        r2=float(lre(r2, cert["r2"])),
    )


def r_lm(y, X):
    """R's lm(y ~ 0 + X): coefficients, standard errors, sigma, R^2."""
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(
            os.path.join(d, "x.txt"), np.column_stack([y, X]), fmt="%.17g"
        )
        code = (
            "m <- as.matrix(read.table('%s/x.txt')); y <- m[,1]; X <- m[,-1, drop=FALSE]; "
            "f <- lm(y ~ 0 + X); s <- summary(f); co <- coef(f); se <- rep(NA, length(co)); "
            "se[!is.na(co)] <- s$coefficients[, 2]; "
            "writeLines(sprintf('%%.17g', c(co, se, s$sigma, s$r.squared)), '%s/o.txt')"
            % (d, d)
        )
        subprocess.run(
            ["Rscript", "-e", code], check=True, capture_output=True
        )
        v = np.array(
            [
                float(t) if t != "NA" else np.nan
                for t in open(os.path.join(d, "o.txt")).read().split()
            ]
        )
    k = X.shape[1]
    return v[:k], v[k : 2 * k], v[2 * k], v[2 * k + 1]


def main():
    import statsmodels.api as sm
    import vipyr as vp

    out = {
        "about": __doc__.split("\n\n")[1].replace("\n", " "),
        "datasets": {},
    }
    for name in DATASETS:
        y, X, cert = read(name)
        row = {}
        for label, kw in [
            ("vipyr qr", dict(solver="qr")),
            ("vipyr qr, no refinement", dict(solver="qr", refine=False)),
            ("vipyr cholesky", dict(solver="cholesky")),
            ("vipyr svd", dict(solver="svd", tol=1e-15)),
        ]:
            r = vp.stats.ols(y, X, **kw)
            row[label] = score(r.coef, r.bse, r.resid_sd, r.rsquared, cert)
        f = sm.OLS(y, X).fit()
        row["statsmodels"] = score(
            f.params, f.bse, np.sqrt(f.scale), f.rsquared, cert
        )
        if name in NO_INTERCEPT:
            # statsmodels' rsquared is uncentered without a constant, as NIST's
            pass
        co, se, sd, r2 = r_lm(y, X)
        row["R lm"] = score(co, se, sd, r2, cert)
        # the least-squares solution of the design matrix as stored in
        # doubles (x^k rounded), by mpmath at 60 digits: the digits any
        # double-precision solver can reach, and how close vipyr comes to it
        mp.mp.dps = 60
        exact = np.array(
            [
                float(v)
                for v in mp.qr_solve(
                    mp.matrix(X.tolist()), mp.matrix(y.tolist())
                )[0]
            ]
        )
        row["attainable"] = dict(coef=float(lre(exact, cert["coef"]).min()))
        r = vp.stats.ols(y, X)
        row["vipyr qr vs the exact solution"] = dict(
            coef=float(lre(r.coef, exact).min())
        )
        out["datasets"][name] = row
        print(
            "%-9s " % name
            + "  ".join(
                "%s %4.1f"
                % (
                    k.split()[0] + ("" if k.count(" ") == 0 else "*"),
                    v["coef"],
                )
                for k, v in row.items()
            ),
            flush=True,
        )
    (HERE / "nist_lls.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

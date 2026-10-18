# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.stats.glm against R's glm, on R's own datasets.

    python3 lib/stats/glm_check.py         # writes lib/stats/glm_check.json

Rscript builds each model's design matrix (model.matrix), fits it with glm
at R's default epsilon (1e-8) and at epsilon 1e-14 (maxit 100), and hands
the data and both fits over as JSON; vp.stats.glm fits the same matrices
at the same two settings. The table keeps, per case, the largest relative
difference of the coefficients, the standard errors and the deviance, and
whether the two raise the same warnings. R 4.3.3.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

R_CODE = r"""
suppressMessages(library(jsonlite))
clot <- data.frame(u = c(5, 10, 15, 20, 30, 40, 60, 80, 100),
                   lot1 = c(118, 58, 42, 35, 27, 25, 21, 19, 18))
sep <- data.frame(x = c(1, 2, 3, 4, 5, 6), y = c(0, 0, 0, 1, 1, 1))
es <- transform(esoph, n = ncases + ncontrols)
cases <- list(
  list(name = "mtcars am ~ wt + hp, binomial logit", f = am ~ wt + hp, d = mtcars,
       fam = binomial("logit"), fam_py = "binomial", link = "logit"),
  list(name = "mtcars vs ~ mpg, binomial probit", f = vs ~ mpg, d = mtcars,
       fam = binomial("probit"), fam_py = "binomial", link = "probit"),
  list(name = "esoph ncases/n ~ agegp + alcgp, binomial with trials", f = ncases / n ~ agegp + alcgp, d = es,
       fam = binomial("logit"), fam_py = "binomial", link = "logit", w = "n"),
  list(name = "warpbreaks breaks ~ wool + tension, poisson log", f = breaks ~ wool + tension, d = warpbreaks,
       fam = poisson("log"), fam_py = "poisson", link = "log"),
  list(name = "InsectSprays count ~ spray, poisson log", f = count ~ spray, d = InsectSprays,
       fam = poisson("log"), fam_py = "poisson", link = "log"),
  list(name = "clotting lot1 ~ log(u), Gamma inverse", f = lot1 ~ log(u), d = clot,
       fam = Gamma("inverse"), fam_py = "gamma", link = "inverse"),
  list(name = "clotting lot1 ~ log(u), Gamma log", f = lot1 ~ log(u), d = clot,
       fam = Gamma("log"), fam_py = "gamma", link = "log"),
  list(name = "separated: y = (x > 3), binomial logit", f = y ~ x, d = sep,
       fam = binomial("logit"), fam_py = "binomial", link = "logit")
)
out <- lapply(cases, function(k) {
  w <- if (is.null(k$w)) rep(1, nrow(k$d)) else k$d[[k$w]]
  d <- k$d
  d$.w <- w
  mf <- model.frame(k$f, k$d)
  X <- model.matrix(k$f, k$d)
  y <- model.response(mf)
  fits <- lapply(list(c(1e-8, 25), c(1e-14, 100)), function(ctl) {
    warn <- character(0)
    g <- withCallingHandlers(
      glm(k$f, family = k$fam, data = d, weights = .w, control = glm.control(epsilon = ctl[1], maxit = ctl[2])),
      warning = function(e) { warn <<- c(warn, conditionMessage(e)); invokeRestart("muffleWarning") })
    s <- summary(g)
    list(coef = unname(coef(g)), se = unname(s$coefficients[, 2]), deviance = g$deviance,
         null = g$null.deviance, aic = g$aic, iter = g$iter, dispersion = s$dispersion, warnings = warn)
  })
  list(name = k$name, family = k$fam_py, link = k$link, X = unname(X), y = unname(as.numeric(y)), w = w,
       names = colnames(X), default = fits[[1]], tight = fits[[2]])
})
cat(toJSON(out, digits = I(17), auto_unbox = TRUE))
"""


def rel(a, b):
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(
        np.asarray(b, float)
    )
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def main():
    import vipyr as vp

    r = subprocess.run(
        ["Rscript", "-e", R_CODE], check=True, capture_output=True, text=True
    )
    cases = json.loads(r.stdout)
    rows = []
    for c in cases:
        X, y, w = (
            np.array(c["X"], float),
            np.array(c["y"], float),
            np.array(c["w"], float),
        )
        if X.ndim == 1:
            X = X[:, None]
        row = {"case": c["name"]}
        for label, eps, maxit in [
            ("default", 1e-8, 25),
            ("tight", 1e-14, 100),
        ]:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                g = vp.stats.glm(
                    y,
                    X,
                    family=c["family"],
                    link=c["link"],
                    weights=w,
                    epsilon=eps,
                    maxit=maxit,
                )
            ref = c[label]
            if isinstance(ref["warnings"], str):
                ref["warnings"] = [ref["warnings"]]
            row[label] = dict(
                coef=rel(g.coef, ref["coef"]),
                se=rel(g.bse, ref["se"]),
                deviance=rel(g.deviance, ref["deviance"]),
                aic=rel(g.aic, ref["aic"]),
                iterations=[g.iterations, ref["iter"]],
                warnings_same=sorted(g.warnings) == sorted(ref["warnings"]),
                warnings=list(g.warnings),
                r_warnings=list(ref["warnings"]),
            )
        rows.append(row)
        t = row["tight"]
        print(
            "%-58s coef %.1e  se %.1e  dev %.1e  aic %.1e  iter %s  warnings %s"
            % (
                c["name"],
                t["coef"],
                t["se"],
                t["deviance"],
                t["aic"],
                t["iterations"],
                "same"
                if t["warnings_same"]
                else "%s vs R %s" % (t["warnings"], t["r_warnings"]),
            ),
            flush=True,
        )
    (HERE / "glm_check.json").write_text(
        json.dumps(
            {
                "about": __doc__.split("\n\n")[1].replace("\n", " "),
                "cases": rows,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

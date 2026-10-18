# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A scikit-learn pipeline as one C file
=====================================

Fit a scikit-learn pipeline (scaling, one-hot encoding and gradient-boosted
trees) on a synthetic loan-default table, and compile it with
`vipyr.compile.compile_model` into a single C99 file that needs only libc.
The compiler checks the C against scikit-learn on every held-out row before
it returns, and embeds a few hundred rows with scikit-learn's outputs so the
file can recheck itself on another machine (`./model --selftest`).
"""

# %%
# ## The model
#
# 6,000 synthetic applicants: income, debt-to-income ratio, credit history
# length, number of recent inquiries and a home-ownership category. The label
# comes from a logistic model with an interaction, so trees have something to
# find. The pipeline is a `ColumnTransformer` (`StandardScaler` on the
# numbers, `OneHotEncoder` on the category) and a
# `GradientBoostingClassifier` of 120 trees of depth 3.

import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from vipyr.compile import compile_model, load_model
from vipyr.figure import figure

rng = np.random.default_rng(4)
n = 6000
df = pd.DataFrame(
    {
        "income": np.round(rng.lognormal(10.8, 0.5, n), 0),
        "dti": np.round(rng.beta(2.0, 5.0, n), 4),
        "history_years": np.round(rng.gamma(3.0, 3.0, n), 1),
        "inquiries": rng.poisson(1.2, n).astype(float),
        "home": rng.choice(
            ["rent", "own", "mortgage"], n, p=[0.45, 0.15, 0.40]
        ),
    }
)
logit = (
    -2.0
    + 4.0 * df.dti
    - 0.6 * np.log(df.income / 50000)
    - 0.08 * df.history_years
    + 0.35 * df.inquiries
    + 1.5 * df.dti * (df.home == "rent")
    - 0.4 * (df.home == "own")
)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
train, test = df.iloc[:4500], df.iloc[4500:].reset_index(drop=True)
num = ["income", "dti", "history_years", "inquiries"]
pipe = Pipeline(
    [
        (
            "prep",
            ColumnTransformer(
                [
                    ("num", StandardScaler(), num),
                    ("cat", OneHotEncoder(), ["home"]),
                ]
            ),
        ),
        (
            "gbm",
            GradientBoostingClassifier(
                n_estimators=120, max_depth=3, random_state=0
            ),
        ),
    ]
).fit(train, y[:4500])
print(
    "default rate %.3f; held-out accuracy %.3f"
    % (y.mean(), pipe.score(test, y[4500:]))
)

# %%
# ## Compile and check
#
# `compile_model` writes the pipeline as readable Viv (`loans.viv`), exports
# it to C with `viv export`, builds it, and compares its outputs with
# scikit-learn's on all 1,500 held-out rows. It returns a report.

work = Path(tempfile.mkdtemp())
rep = compile_model(pipe, test, work / "loans.c", log=lambda *a: None)
print("C file: %d lines; estimator %s" % (rep["lines"], rep["estimator"]))
print(
    "rows verified: %d, outputs per row: %d"
    % (rep["rows_verified"], rep["outputs_per_row"])
)
print(
    "largest difference from scikit-learn: %.1e (%d ulp), labels identical: %s"
    % (rep["max_abs_diff"], rep["max_ulps"], rep["labels_identical"])
)
print("self-test:", rep["builds"][0]["selftest"])
print("the first lines of the generated Viv:")
print("\n".join((work / "loans.viv").read_text().splitlines()[:12]))

# %%
# ## The compiled model's predictions against scikit-learn's
#
# `load_model` builds the C as a shared library and calls it on a DataFrame.
# The scatter plot is the default probability from the C file against
# scikit-learn's for every second held-out row: they lie on the diagonal.
# Of the 1,500 rows, one differs at all, by one unit in the last place,
# within the tolerance the compiler allows.

m = load_model(work / "loans.c")
p_c = m.predict_proba(test)[:, 1]
p_sk = pipe.predict_proba(test)[:, 1]
d = p_c - p_sk
print(
    "max |p_C - p_sklearn| = %.1e over %d rows; %d rows differ at all"
    % (np.abs(d).max(), len(d), int(np.sum(d != 0)))
)
fig = figure(
    "Probability of default: compiled C against scikit-learn",
    "scikit-learn",
    "C file",
)
fig.scatter(p_sk[::2], p_c[::2], "held-out rows")
fig.plot([0, 1], [0, 1], "y = x")
fig.style("y = x", "s2 dash thin nomarker")

# %%
# ## Latency
#
# Measured by `examples/deploy/deploy.py` on its own pipeline (150 trees of
# depth 3, a Viv feature function and 30,000 rows), and committed in
# `examples/deploy/results.json`. Each tree is written as nested `if`s with
# its thresholds as constants, so the C compiler sees straight-line
# comparisons and no table lookups. The C library, called through ctypes on
# a DataFrame and computing the features too, scores the batch faster than
# scikit-learn's `predict_proba` on the frame that already holds the
# features, and one row in microseconds where scikit-learn's per-call
# overhead takes hundreds.

res = json.load(open("examples/deploy/results.json"))
sp = res["speed"]
for k in sorted(sp):
    if k.endswith("_rows_per_s"):
        print("%-48s %12.0f rows/s" % (k[: -len("_rows_per_s")], sp[k]))
print(
    "one row: C %.0f us, scikit-learn %.0f us"
    % (sp["single_row_c_us"], sp["single_row_sklearn_us"])
)
print(
    "rows: %d; compile and check, total %.1f s"
    % (sp["rows"], res["compile_total_s"])
)

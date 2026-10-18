# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The vipyr twin of lstsq_qr_vs_normal_equations.viv: the same fits by
vp.stats.ols(solver="qr") and vp.stats.ols(solver="cholesky"), the
condition numbers by NumPy's SVD, the files read in Python.

###
python: ~/.venvs/power/bin/python
###
"""
import re

import numpy as np
import vipyr as vp
from claimlib import show, test

DATASETS = ["Norris", "Pontius", "Longley", "Wampler1", "Wampler2", "Wampler3", "Wampler4", "Wampler5", "Filip"]
CAP = 15.0


def read_strd(name):
    lines = open(f"lib/stats/data/{name}.dat").read().split("\n")
    start = next(i for i, ln in enumerate(lines) if re.match(r"^Data:\s+y\b", ln))
    cert = [float(ln.split()[1]) for ln in lines[:start] if re.match(r"^\s+B\d+\s", ln) and len(ln.split()) >= 3]
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[start + 1:] if ln.strip()])
    return np.array(cert), rows


def coefs(r):
    return np.asarray(r.coef, float)


def lre(b, c):
    e = np.abs(b - c) / np.abs(c)
    e = e[e > 0]
    return max(0.0, min(CAP, float((-np.log10(e)).min()))) if e.size else CAP


conds, dq, dn = [], [], []
for name in DATASETS:
    cert, rows = read_strd(name)
    p = len(cert)
    y = rows[:, 0]
    if name == "Longley":
        X = np.column_stack([np.ones(len(y)), rows[:, 1:]])
    else:
        X = np.ones((len(y), p))
        for j in range(1, p):  # powers by repeated multiplication, as the Viv program builds them
            X[:, j] = X[:, j - 1] * rows[:, 1]
    s = np.linalg.svd(X / np.linalg.norm(X, axis=0), compute_uv=False)
    conds.append(np.log10(s.max() / s.min()))
    dq.append(lre(coefs(vp.stats.ols(y, X, solver="qr")), cert))
    dn.append(lre(np.nan_to_num(coefs(vp.stats.ols(y, X, solver="cholesky"))), cert))
conds, dq, dn = map(np.array, (conds, dq, dn))
show("digits_qr", "[" + ", ".join(f"{v:.6g}" for v in dq) + "]")
show("min_digits_qr", dq.min())
show("cond_min", 10 ** conds.min())
show("cond_max", 10 ** conds.max())


@test("QR gets at least 7.5 digits on every dataset, and at least 16 - log10 kappa - 3 on each")
def _():
    assert dq.min() > 7.5 and (dq > 16 - conds - 3).all(), dq


@test("Cholesky on the normal equations keeps between 16 - 2 log10 kappa - 4 and 16 - 2 log10 kappa + 2 digits")
def _():
    assert (dn > 16 - 2 * conds - 4).all() and (dn <= np.maximum(0, 16 - 2 * conds + 2) + 1e-9).all(), dn


@test("none on Filip")
def _():
    assert dn[-1] == 0.0, dn[-1]


@test("QR is ahead by at least 2 digits on every dataset")
def _():
    assert (dq - dn > 2).all(), dq - dn

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Randomized trace estimation and low-rank kernel approximation
=============================================================

Estimate the trace of a matrix you can only multiply by, and build a
low-rank factor of a kernel matrix from a few of its columns. Hutch++
(`vp.randla.trace_hutchpp`, Meyer, Musco, Musco and Woodruff, 2021) spends a
third of its matrix products on a sketch of the top of the spectrum and the
rest on Hutchinson's estimator for what is left. Its error is guaranteed to
fall at least like 1/m in the number of products m, where Hutchinson's falls
like 1/sqrt(m).
Randomly pivoted Cholesky (`vp.randla.rp_cholesky`, Chen, Epperly, Tropp and
Webber, 2023) picks columns with probability proportional to the diagonal
of what is still unexplained.
"""

# %%
# ## A matrix with a decaying spectrum
#
# A 400 x 400 symmetric matrix with eigenvalues j^-2 in a random orthonormal
# basis: most of its trace sits in a few large eigenvalues, the case where
# Hutch++ helps most. Hutchinson's estimator is written here in NumPy with
# Rademacher probes, tr(A) ~ mean of g^T A g. Hutchinson's error follows
# 1/sqrt(m). Hutch++ beats its 1/m guarantee here: with eigenvalues falling
# like j^-2, the part its sketch misses shrinks as the sketch grows, and the
# fitted slope is about -2.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

rng = np.random.default_rng(0)
n = 400
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
lam = np.arange(1, n + 1) ** -2.0
A = (Q * lam) @ Q.T
exact = lam.sum()
print("exact trace %.10f" % exact)


def hutchinson(A, m, seed):
    g = np.random.default_rng(seed).choice([-1.0, 1.0], size=(A.shape[0], m))
    return np.sum(g * (A @ g)) / m


ms = [12, 24, 48, 96, 192, 384]
seeds = range(20)
err_pp, err_h = [], []
for m in ms:
    e_pp = [
        abs(vp.randla.trace_hutchpp(A=A, m=m, seed=s).trace - exact) / exact
        for s in seeds
    ]
    e_h = [abs(hutchinson(A, m, s) - exact) / exact for s in seeds]
    err_pp.append(np.mean(e_pp))
    err_h.append(np.mean(e_h))
    print(
        "m = %3d   Hutch++ mean relative error %.2e   Hutchinson %.2e"
        % (m, err_pp[-1], err_h[-1])
    )
for name, e in (("Hutch++", err_pp), ("Hutchinson", err_h)):
    print(
        "%-10s fitted slope of log error against log m: %.2f"
        % (name, np.polyfit(np.log(ms), np.log(e), 1)[0])
    )

fig = figure(
    "Trace estimation error, mean of 20 seeds",
    "matrix-vector products m",
    "relative error",
)
fig.plot(ms, err_pp, "Hutch++ (vipyr)")
fig.plot(ms, err_h, "Hutchinson (NumPy)")
fig.plot(ms, [err_pp[0] * ms[0] / m for m in ms], "1/m")
fig.plot(ms, [err_h[0] * np.sqrt(ms[0] / m) for m in ms], "$1/\\sqrt{m}$")
fig.style("1/m", "s1 dash thin nomarker")
fig.style("$1/\\sqrt{m}$", "s2 dash thin nomarker")
fig.xscale("log")
fig.yscale("log")

# %%
# ## A low-rank factor of a Gaussian kernel
#
# The kernel matrix K_ij = exp(-|x_i - x_j|^2 / (2 h^2)) of 800 points in the
# unit square, h = 0.15. `rp_cholesky` returns F with K ~ F F^T after k
# pivots. The best possible rank-k error in the trace norm is the sum of the
# eigenvalues past the k-th, which needs the full eigendecomposition.
# RPCholesky reads only k columns of K and its diagonal. Its error stays
# within a factor of 2 to 7 of the best while both fall by four orders of
# magnitude between k = 10 and k = 120.

pts = np.random.default_rng(1).random((800, 2))
d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
K = np.exp(-d2 / (2 * 0.15**2))
ev = np.sort(np.linalg.eigvalsh(K))[::-1]
ks = [10, 20, 40, 60, 80, 120]
rp, best = [], []
for k in ks:
    F = vp.randla.rp_cholesky(A=K, k=k, seed=7).F
    rp.append((np.trace(K) - np.sum(F * F)) / np.trace(K))
    best.append(ev[k:].sum() / np.trace(K))
    print(
        "k = %3d   RPCholesky trace error %.2e   best rank-k %.2e   ratio %.2f"
        % (k, rp[-1], best[-1], rp[-1] / best[-1])
    )

fig = figure(
    "Gaussian kernel, 800 points: trace error of the rank-k factor",
    "rank k",
    "$\\mathrm{tr}(K - FF^\\top)/\\mathrm{tr}(K)$",
)
fig.plot(ks, rp, "RPCholesky (vipyr)")
fig.plot(ks, best, "best rank k (eigendecomposition)")
fig.style("best rank k (eigendecomposition)", "s2 dash")
fig.yscale("log")

# %%
# ## How it's checked
#
# `lib/randla/check_ref.py` runs the same two experiments in NumPy: Hutch++'s
# fitted slope must be steeper than -0.9 on the j^-2 spectrum, and
# RPCholesky's trace error at k = 40 must be below the best rank-20 error on
# a 500-point kernel.

print(
    "check():",
    vp.randla.trace_hutchpp(A=A, m=48, seed=0).check().passed,
    vp.randla.rp_cholesky(A=K[:300, :300], k=40, seed=7).check().passed,
)

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""lib/randla against NumPy, SciPy and scikit-learn, through vipyr.

    VIV=target/release/viv python3 lib/randla/check_ref.py [--quick]

Each check prints one row and the script exits nonzero if any fails. The
Viv side is lib/randla/api.viv compiled to C by vipyr (exact mode: no
fused multiply-add, Viv's own libm), so these are the same bits the
interpreter and wasm print.

1. The generator: Philox4x32-10 written again here in NumPy; the Rademacher
   operator must match bit for bit, the Gaussian one to 4 ulps (Box-Muller
   goes through log, cos and sin, where Viv's libm and the platform's may
   round differently).
2. rSVD against the exact SVD (numpy.linalg.svd) on decaying spectra: the
   Frobenius error against the Eckart-Young optimum, and against
   sklearn.utils.extmath.randomized_svd with the same oversampling and power
   iterations (accuracy and time).
3. Least squares against numpy.linalg.lstsq on an ill-conditioned problem:
   the residual norms to 1e-13 relative, the solutions to the perturbation
   bound at unit roundoff.
4. Hutch++ and Hutchinson against the exact trace, error against the number
   of products.
5. RPCholesky's trace error against the best rank-k trace error.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# one thread for NumPy's BLAS: the Viv side runs on one core, and a shared
# machine makes threaded timings noise
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))
import vipyr  # noqa: E402

QUICK = "--quick" in sys.argv
FAILS = []


def row(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} {detail}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------ Philox in NumPy

M32 = np.uint64(0xFFFFFFFF)


def philox4x32(c0, c1, c2, c3, k0, k1):
    """Random123's philox4x32-10 on arrays of counters (uint64 holding 32-bit words)."""
    x0, x1, x2, x3 = (np.asarray(c, dtype=np.uint64) for c in (c0, c1, c2, c3))
    ka, kb = np.uint64(k0), np.uint64(k1)
    for r in range(10):
        if r > 0:
            ka = (ka + np.uint64(0x9E3779B9)) & M32
            kb = (kb + np.uint64(0xBB67AE85)) & M32
        p0 = np.uint64(0xD2511F53) * x0
        p1 = np.uint64(0xCD9E8D57) * x2
        x0, x1, x2, x3 = ((p1 >> np.uint64(32)) ^ x1 ^ ka, p1 & M32, (p0 >> np.uint64(32)) ^ x3 ^ kb, p0 & M32)
    return x0, x1, x2, x3


def block(seed, stream, index, sub=0):
    index = np.asarray(index, dtype=np.uint64)
    return philox4x32(index & M32, index >> np.uint64(32), np.full_like(index, stream % 2**32),
                      np.full_like(index, sub), seed % 2**32, seed // 2**32)


def unit_open(a, b):
    k = (a >> np.uint64(6)).astype(np.float64) * 67108864.0 + (b >> np.uint64(6)).astype(np.float64)
    return (k + 0.5) / 4503599627370496.0


def np_gaussian(rows, cols, seed, stream):
    total = rows * cols
    w0, w1, w2, w3 = block(seed, stream, np.arange((total + 1) // 2))
    u1, u2 = unit_open(w0, w1), unit_open(w2, w3)
    rad = np.sqrt(-2.0 * np.log(u1))
    g = np.empty(2 * len(u1))
    g[0::2] = rad * np.cos(2 * np.pi * u2)
    g[1::2] = rad * np.sin(2 * np.pi * u2)
    return g[:total].reshape(rows, cols)


def np_rademacher(rows, cols, seed, stream):
    total = rows * cols
    w = np.stack(block(seed, stream, np.arange((total + 3) // 4)), axis=1).ravel()[:total]
    return np.where(w >= np.uint64(2**31), -1.0, 1.0).reshape(rows, cols)


def with_spectrum(m, n, s, rng):
    U, _ = np.linalg.qr(rng.standard_normal((m, len(s))))
    V, _ = np.linalg.qr(rng.standard_normal((n, len(s))))
    return (U * s) @ V.T


def best_of(f, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        t.append(time.perf_counter() - t0)
    return min(t), out


def main():
    from sklearn.utils.extmath import randomized_svd as sk_rsvd

    api = vipyr.load(HERE / "api.viv")
    fast = vipyr.load(HERE / "api.viv", fast=True)
    rng = np.random.default_rng(20261011)

    # 1. the generator
    R = api.rademacher_op(37, 29, 123456789012, 3)
    row("Philox Rademacher, bit for bit", np.array_equal(R, np_rademacher(37, 29, 123456789012, 3)), "37x29")
    G = api.gaussian_op(301, 7, 42, 5)
    Gp = np_gaussian(301, 7, 42, 5)
    ulps = np.max(np.abs(G - Gp) / np.spacing(np.abs(Gp)))
    row("Philox Gaussian, Box-Muller", ulps <= 4, f"max {ulps:.0f} ulp, {int(np.sum(G != Gp))} of {G.size} differ")

    # 2. rSVD
    print("\nrSVD (k + 10 columns, 2 power iterations; time = best of 3)")
    print(f"  {'m x n':>11} {'k':>3} {'decay':>9} {'viv/opt':>8} {'skl/opt':>8} {'viv ms':>8} {'fast ms':>8} {'skl ms':>7}")
    cases = [(500, 300, 10, "poly1"), (1000, 600, 20, "poly2"), (2000, 1000, 20, "exp")]
    if not QUICK:
        cases.append((4000, 2000, 50, "poly1"))
    table = []
    for m, n, k, decay in cases:
        r = min(m, n)
        j = np.arange(1, r + 1, dtype=float)
        s = {"poly1": 1.0 / j, "poly2": 1.0 / j**2, "exp": np.exp(-j / 5.0)}[decay]
        A = with_spectrum(m, n, s, rng)
        opt = np.linalg.norm(s[k:])
        tv, (U, sv, Vt) = best_of(lambda: api.randomized_svd(A, k, 10, 2, 1), 3)
        tf, (Uf, sf, Vtf) = best_of(lambda: fast.randomized_svd(A, k, 10, 2, 1), 3)
        ts, (Us, ss, Vts) = best_of(lambda: sk_rsvd(A, k, n_oversamples=10, n_iter=2,
                                                    power_iteration_normalizer="QR", random_state=1), 3)
        ev = np.linalg.norm(A - (U * sv) @ Vt) / opt
        ef = np.linalg.norm(A - (Uf * sf) @ Vtf) / opt
        es = np.linalg.norm(A - (Us * ss) @ Vts) / opt
        table.append((m, n, k, decay, ev, es, tv, tf, ts))
        print(f"  {m:5d}x{n:<5d} {k:3d} {decay:>9} {ev:8.5f} {es:8.5f} {1e3*tv:8.1f} {1e3*tf:8.1f} {1e3*ts:7.1f}")
        row(f"rSVD {m}x{n} k={k} {decay}: error / Eckart-Young", ev < 1.05 and ef < 1.05 and abs(ev - ef) < 1e-6,
            f"viv {ev:.5f}, fast {ef:.5f}, sklearn {es:.5f}")
        sing = np.linalg.svd(A, compute_uv=False)[:k]
        row(f"rSVD {m}x{n}: leading singular values", np.max(np.abs(sv - sing) / sing[0]) < 1e-3,
            f"max rel {np.max(np.abs(sv - sing) / sing[0]):.1e}")

    # 3. least squares
    print()
    m, n = (3000, 60) if not QUICK else (1500, 40)
    s = np.logspace(0, -6, n)
    A = with_spectrum(m, n, s, rng)
    b = A @ rng.standard_normal(n) + 1e-3 * rng.standard_normal(m)
    tl, (xl, *_) = best_of(lambda: np.linalg.lstsq(A, b, rcond=None), 3)
    rl = np.linalg.norm(b - A @ xl)
    kappa = s[0] / s[-1]
    bound = 10 * 1.11e-16 * (kappa + kappa**2 * rl / (s[0] * np.linalg.norm(xl)))
    for name in ("sap_srht", "sap_saso", "sap_gaussian"):
        t, (x, it) = best_of(lambda: getattr(api, name)(A, b, 3), 3)
        dr = abs(np.linalg.norm(b - A @ x) - rl) / rl
        dx = np.linalg.norm(x - xl) / np.linalg.norm(xl)
        row(f"{name} {m}x{n}, cond 1e6, vs lstsq", dr < 1e-13 and dx < bound,
            f"|r| rel {dr:.1e}, x rel {dx:.1e} (bound {bound:.0e}), {int(it)} its, {1e3*t:.0f} ms vs lstsq {1e3*tl:.0f} ms")

    # 4. trace estimators
    print("\ntrace estimators, n = 400, eigenvalues j^-2 (mean relative error over 8 seeds)")
    n = 400
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.arange(1, n + 1, dtype=float) ** -2.0
    A = (Q * lam) @ Q.T
    t = lam.sum()
    print(f"  {'products':>8} {'Hutchinson':>11} {'Hutch++':>10}")
    errs = {}
    for mv in (12, 48, 192):
        e1 = np.mean([abs(api.trace_hutchinson(A, mv, sd) - t) / t for sd in range(8)])
        e2 = np.mean([abs(api.trace_hutchpp(A, mv, sd) - t) / t for sd in range(8)])
        errs[mv] = (e1, e2)
        print(f"  {mv:8d} {e1:11.2e} {e2:10.2e}")
    s1 = np.log(errs[12][0] / errs[192][0]) / np.log(16)
    s2 = np.log(errs[12][1] / errs[192][1]) / np.log(16)
    row("Hutchinson error ~ m^-1/2", 0.25 < s1 < 0.8, f"fitted slope {s1:.2f}")
    row("Hutch++ error falls faster than m^-1 / 2", s2 > 0.9, f"fitted slope {s2:.2f}")

    # 5. RPCholesky on a Gaussian kernel
    x = np.sort(rng.uniform(0, 1, 500))
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / 0.05**2)
    ev = np.linalg.eigvalsh(K)[::-1]
    print("\nRPCholesky, Gaussian kernel on 500 points (bandwidth 0.05)")
    print(f"  {'k':>4} {'tr(K - F F^T)':>14} {'best rank-k':>12} {'ratio':>6}")
    ok = True
    for k in (10, 20, 40):
        F = api.rp_cholesky(K, k, 7)
        err = np.trace(K) - np.sum(F * F)
        best = ev[k:].sum()
        print(f"  {k:4d} {err:14.4e} {best:12.4e} {err / best:6.2f}")
        # Chen et al. (2022): the trace error is within a modest factor of the best rank-k one
        # once k exceeds the rank of interest by a margin; we hold k=40 against the best rank 20
        if k == 40:
            ok = err <= ev[20:].sum()
    row("RPCholesky k=40 beats the best rank-20 trace error", ok, "")

    print(f"\n{len(FAILS)} failed" if FAILS else "\nall checks pass")
    return 1 if FAILS else 0


if __name__ == "__main__":
    os.environ.setdefault("VIV", str(REPO / "target" / "release" / "viv"))
    sys.exit(main())

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""randomized_svd(df_or_array, k) from Python, with lib/randla doing the work.

    VIV=target/release/viv python3 lib/randla/example_rsvd.py

Takes a NumPy array or a pandas/polars DataFrame of numeric columns, and
times it against scikit-learn's randomized_svd with the same settings (k +
10 columns, 2 power iterations, QR between them). Both run on one core.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
import vipyr  # noqa: E402

_exact = vipyr.load(HERE / "api.viv")
_fast = None


def randomized_svd(df_or_array, k, oversample=10, power=2, seed=0, fast=False):
    """The rank-k SVD of a matrix or a frame of numeric columns: U, s, Vt."""
    global _fast
    if hasattr(df_or_array, "to_numpy"):          # pandas or polars
        A = np.ascontiguousarray(df_or_array.to_numpy(), dtype=np.float64)
    else:
        A = np.ascontiguousarray(df_or_array, dtype=np.float64)
    if fast and _fast is None:
        _fast = vipyr.load(HERE / "api.viv", fast=True)
    lib = _fast if fast else _exact
    return lib.randomized_svd(A, k, oversample, power, seed)


def best_ms(f, reps=5):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return 1e3 * min(t)


def main():
    import pandas as pd
    from sklearn.utils.extmath import randomized_svd as sk

    rng = np.random.default_rng(0)
    m, n, k = 3000, 400, 15
    # a frame with a decaying spectrum: 400 columns driven by 15 strong factors plus noise
    F = rng.standard_normal((m, k)) * (1.0 / np.arange(1, k + 1))
    A = F @ rng.standard_normal((k, n)) + 1e-3 * rng.standard_normal((m, n))
    df = pd.DataFrame(A, columns=[f"c{j}" for j in range(n)])
    U, s, Vt = randomized_svd(df, k)
    sv = np.linalg.svd(A, compute_uv=False)
    opt = np.sqrt(np.sum(sv[k:] ** 2))
    err = np.linalg.norm(A - (U * s) @ Vt)
    Us, ss, Vts = sk(A, k, n_oversamples=10, n_iter=2, power_iteration_normalizer="QR", random_state=0)
    errs = np.linalg.norm(A - (Us * ss) @ Vts)
    print(f"{m} x {n} frame, k = {k}")
    print(f"  error / Eckart-Young optimum: viv {err / opt:.6f}, sklearn {errs / opt:.6f}")
    print(f"  top singular values, viv - exact: {np.max(np.abs(s - sv[:k])):.1e}")
    print(f"  time: viv exact {best_ms(lambda: randomized_svd(A, k)):.1f} ms, "
          f"viv fast {best_ms(lambda: randomized_svd(A, k, fast=True)):.1f} ms, "
          f"sklearn {best_ms(lambda: sk(A, k, n_oversamples=10, n_iter=2, power_iteration_normalizer='QR', random_state=0)):.1f} ms")


if __name__ == "__main__":
    os.environ.setdefault("VIV", str(HERE.parents[1] / "target" / "release" / "viv"))
    main()

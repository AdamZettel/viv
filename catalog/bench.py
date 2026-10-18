# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Timing for three catalog entries: Viv, the polars version and the pandas
reference.

    VIV=../target/release/viv python3 bench.py           # 1e7 rows
    VIV=../target/release/viv python3 bench.py 1000000   # fewer

Viv is timed two ways: the kernel on a frame already in canonical order
(`module(...).kernel(df)`, frame conversion included) and the catalog
wrapper, which also sorts the rows and puts the result back. Warm median of
3 runs, one core for Viv. The pandas reference is timed once on 1e5 rows and
scaled to the full size (it is a Python loop per row, so its time is close
to linear in the rows when each user's history is short). The polars
versions are multicore and are checked against the reference on 1e5 rows.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import catalog  # noqa: E402
from vipyr.pit import _equal  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 7
N_REF = 10 ** 5


def timed(f, *a, reps=3):
    f(*a)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*a)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def differ(a, b):
    out = {}
    for c in a.columns:
        x, y = np.asarray(a[c].to_numpy()), np.asarray(b[c].to_numpy())
        if x.dtype.kind == "M" or y.dtype.kind == "M":
            x, y = x.astype("datetime64[ns]").view(np.int64), y.astype("datetime64[ns]").view(np.int64)
        out[c] = int((~_equal(x, y)).sum())
    return out


def main():
    print("load %.1f" % os.getloadavg()[0])
    small = catalog.make_events(N_REF, seed=3)
    small_l = small[["user", "ts"]]
    small_r = catalog.make_quotes(small_l, N_REF // 2, seed=4)
    big = catalog.make_events(N, seed=5)[["user", "ts", "amount", "merchant"]]
    a = {c: catalog.column_array(big, c)[0] for c in big.columns}
    big = big.iloc[np.lexsort((a["merchant"], a["amount"], a["ts"], a["user"]))].reset_index(drop=True)
    big_l = big[["user", "ts"]]
    big_r = catalog.make_quotes(big_l, N // 2, seed=6)
    big_pl, big_l_pl, big_r_pl = pl.from_pandas(big), pl.from_pandas(big_l), pl.from_pandas(big_r)
    print("%d rows, %d users; reference timed on %d rows" % (len(big), big["user"].nunique(), len(small)))

    cases = [
        ("sessions",
         lambda: catalog.module("sessions").sessions(big, gap=catalog._ns("30min")),
         lambda: catalog.sessions(big),
         lambda: catalog.sessions_reference(small),
         lambda: catalog.sessions_polars(big_pl),
         lambda: differ(catalog.sessions_polars(pl.from_pandas(small)), catalog.sessions_reference(small))),
        ("velocity",
         lambda: catalog.module("velocity").velocity(big),
         lambda: catalog.velocity(big),
         lambda: catalog.velocity_reference(small),
         lambda: catalog.velocity_polars(big_pl),
         lambda: differ(catalog.velocity_polars(pl.from_pandas(small)),
                        catalog.velocity_reference(small)[["n_1h", "n_24h", "sum_24h"]])),
        ("asof",
         None,
         lambda: catalog.asof(big_l, big_r),
         lambda: catalog.asof_reference(small_l, small_r),
         lambda: catalog.asof_polars(big_l_pl, big_r_pl),
         lambda: differ(catalog.asof_polars(small_l, small_r), catalog.asof_reference(small_l, small_r))),
    ]
    print("| entry | Viv kernel, sorted frame | Viv, catalog wrapper | polars | pandas reference (scaled) "
          "| polars vs reference, rows that differ (1e5) |")
    print("|---|---|---|---|---|---|")
    for name, kern, wrap, ref, polars, diff in cases:
        t_k = timed(kern) if kern else float("nan")
        t_w = timed(wrap)
        t0 = time.perf_counter()
        ref()
        t_r = (time.perf_counter() - t0) * len(big) / len(small)
        t_p = timed(polars)
        print("| %s | %s | %.2f s | %.2f s | %.0f s | %s |" % (
            name, "%.2f s" % t_k if kern else "n/a", t_w, t_p, t_r, diff()))
    print("load %.1f" % os.getloadavg()[0])


if __name__ == "__main__":
    main()

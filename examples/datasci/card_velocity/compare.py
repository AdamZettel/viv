# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Viv, the pandas reference and the assistant-style polars version, side by side.

    python3 compare.py               # 1e6 rows for parity, 1e7 for timing
    python3 compare.py --quick       # 1e5 and 1e6

1. Parity: Viv against the pandas groupby().apply() reference, every row.
2. The polars version against the same reference, and the point-in-time
   check run on both Viv and polars.
3. Timing (warm median of 5; the Viv time includes the frame conversions).
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

import velocity as v  # noqa: E402
import vipyr  # noqa: E402

QUICK = "--quick" in sys.argv
N_PARITY = 10 ** 5 if QUICK else 10 ** 6
N_TIME = 10 ** 6 if QUICK else 10 ** 7
OUTCOMES = ["declined", "chargeback"]


def timed(f, *a, reps=5):
    f(*a)                                   # warm: compile cache, page faults
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*a)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    df = v.make_transactions(N_PARITY, seed=1)
    print("%d transactions, %d cards" % (len(df), df["card"].nunique()))

    t0 = time.perf_counter()
    ref = v.pandas_reference(df)
    t_ref = time.perf_counter() - t0
    viv = v.viv_features(df)
    same = {c: int((viv[c].to_numpy() == ref[c].to_numpy()).sum()) for c in ref.columns}
    print("\n1. Viv vs the pandas reference: equal rows %s of %d; status counts ok/watch/locked %s"
          % (same, len(df), np.bincount(ref["status"], minlength=3).tolist()))
    assert all(x == len(df) for x in same.values()), "Viv and the reference disagree"

    llm = v.polars_llm(pl.from_pandas(df)).to_pandas()
    diff = {c: int((llm[c].to_numpy() != ref[c].to_numpy()).sum()) for c in ref.columns}
    print("\n2. polars (assistant-style) vs the reference: rows that differ %s" % diff)
    i = int(np.flatnonzero(llm["n_prev_1h"].to_numpy() != ref["n_prev_1h"].to_numpy())[0])
    print("   e.g. row %d, card %d at %s: reference n_prev_1h=%d, polars %d"
          % (i, df["card"][i], df["ts"][i], ref["n_prev_1h"][i], llm["n_prev_1h"][i]))

    small = df.iloc[: min(len(df), 200_000)]
    rep = vipyr.check_point_in_time(v.viv_features, small, by="card", ts="ts", outcomes=OUTCOMES)
    print("   point-in-time check, Viv: passed (%d cuts, %d row-features compared)"
          % (len(rep["cuts"]), rep["rows_checked"] * len(rep["features"])))
    try:
        vipyr.check_point_in_time(v.polars_llm, pl.from_pandas(small), by="card", ts="ts", outcomes=OUTCOMES)
        print("   point-in-time check, polars: passed")
    except vipyr.PointInTimeError as e:
        print("   point-in-time check, polars: FAILED\n     %s" % e)

    load = os.getloadavg()[0]
    print("\n3. Timing (load %.1f%s), warm median of 5" % (load, "" if load < 3 else ", ABOVE 3: numbers unreliable"))
    print("   pandas reference, %d rows: %.2f s (one run)" % (len(df), t_ref))
    big = v.make_transactions(N_TIME, seed=2)
    big_pl = pl.from_pandas(big)
    print("   %d rows:" % len(big))
    print("   Viv, pandas frame in and out:   %.3f s" % timed(v.viv_features, big))
    print("   Viv, polars frame in and out:   %.3f s" % timed(v.viv_features, big_pl))
    print("   polars (assistant-style, wrong): %.3f s" % timed(v.polars_llm, big_pl))


if __name__ == "__main__":
    main()

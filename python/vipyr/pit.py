# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The point-in-time check: a feature at time t must not read the future.

    vipyr.check_point_in_time(fn, df, by="card", ts="ts",
                                outcomes=["declined", "chargeback"])

`fn(frame)` computes features, one row per input row, in the input's row
order (a pandas result with the input's index is aligned by index). The check
picks cut times t among the rows' timestamps and, for each, runs `fn` twice
more on altered copies of `df`:

  future   every row with ts > t gets a new later timestamp, and its other
           columns (all but `by`) are permuted among those rows; columns
           named in `outcomes` that hold 0/1 are also flipped.
  instant  the `outcomes` of the rows with ts == t are flipped: the outcome of
           a transaction (declined, charged back) is not known when it is
           scored, at that instant or for its neighbors at the same instant.

Each altered frame is also shuffled row by row, so a function that depends
on row order rather than on (by, ts) fails too. Every feature at every row
with ts <= t must come out equal (NaN equal to NaN). A difference raises
PointInTimeError naming the feature, the row, the cut and which alteration
changed it; success returns a small report.
"""
from __future__ import annotations

import numpy as np

from .frames import frame_library

__all__ = ["check_point_in_time", "PointInTimeError"]


class PointInTimeError(AssertionError):
    """A feature at time t changed when only rows after t (or the outcomes
    at t) changed: it reads the future."""


def _col(df, c):
    return df[c].to_numpy() if frame_library(df) == "pandas" else df.get_column(c).to_numpy()


def _rebuild(df, cols, order):
    """A frame like `df` with columns `cols` (numpy, in df's row order),
    rows taken in `order`."""
    if frame_library(df) == "pandas":
        import pandas as pd
        out = pd.DataFrame({c: cols[c][order] for c in df.columns}, index=df.index[order])
        return out.astype(df.dtypes.to_dict())
    import polars as pl
    return pl.DataFrame([pl.Series(c, cols[c][order]).cast(df.schema[c]) for c in df.columns])


def _features(fn, frame, original, order):
    """fn(frame) as {name: array in `original`'s row order}."""
    res = fn(frame)
    lib = frame_library(res)
    inv = np.empty(len(order), np.int64)
    inv[order] = np.arange(len(order))
    if lib == "pandas":
        if res.index.equals(frame.index) and original.index.is_unique:
            res = res.reindex(original.index)
            return {c: res[c].to_numpy() for c in res.columns}
        return {c: res[c].to_numpy()[inv] for c in res.columns}
    if lib == "polars":
        return {c: res.get_column(c).to_numpy()[inv] for c in res.columns}
    if hasattr(res, "to_numpy") and hasattr(res, "name"):       # a Series
        return {str(res.name): np.asarray(res.to_numpy())[inv]}
    return {k: np.asarray(v)[inv] for k, v in dict(res).items()}


def _flip(v):
    """0/1 and bool columns flipped; anything else left for the permutation."""
    if v.dtype == bool:
        return ~v
    if v.dtype.kind in "iu" and v.size and np.isin(v, (0, 1)).all():
        return 1 - v
    return v


def _later(t, t_all, k, rng):
    """k random timestamps after t, spread over the data's time span, in
    t_all's dtype (numbers or datetime64/timedelta64)."""
    if t_all.dtype.kind in "mM":
        iv = t_all.view(np.int64)
        t0 = np.asarray(t).view(np.int64)
        span = max(2, int(iv.max() - iv.min()))
        return (t0 + rng.integers(1, span + 1, size=k)).view(t_all.dtype)
    if t_all.dtype.kind == "f":
        span = max(1.0, float(t_all.max() - t_all.min()))
        return t + rng.uniform(span * 1e-9, span, size=k)
    span = max(2, int(t_all.max() - t_all.min()))
    return (t + rng.integers(1, span + 1, size=k)).astype(t_all.dtype)


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        a, b = a.astype(float), b.astype(float)
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def check_point_in_time(fn, df, by, ts, outcomes=(), cuts=5, seed=0):
    """Check that `fn`'s features at time t read no row after t and no
    outcome at t (see the module docstring). `df` is a pandas or polars
    frame; `by` the entity key column (or a list of them), `ts` the
    timestamp column (numbers or datetimes), `outcomes` the columns only
    known after a row's own instant. Returns {"cuts": [...], "rows_checked":
    n, "features": [...]}; raises PointInTimeError on a leak."""
    if frame_library(df) is None:
        raise TypeError("check_point_in_time takes a pandas or polars DataFrame")
    rng = np.random.default_rng(seed)
    keys = [by] if isinstance(by, str) else list(by)
    outcomes = list(outcomes)
    cols = {c: _col(df, c) for c in df.columns}
    t_all = cols[ts]
    n = len(t_all)
    ident = np.arange(n)
    base = _features(fn, df, df, ident)
    if any(len(v) != n for v in base.values()):
        raise ValueError("check_point_in_time: fn must return one row per input row")
    lo, hi = np.quantile(np.arange(n), [0.2, 0.8]).astype(int)
    sorted_t = np.sort(t_all)
    picks = sorted_t[rng.integers(lo, hi + 1, size=cuts)]
    report = {"cuts": [], "rows_checked": 0, "features": sorted(base)}
    for t in picks:
        past = t_all <= t
        future = ~past
        now = t_all == t
        trials = []
        if future.any():
            alt = {c: v.copy() for c, v in cols.items()}
            idx = np.flatnonzero(future)
            for c in df.columns:
                if c in keys or c == ts:
                    continue
                alt[c][idx] = alt[c][idx][rng.permutation(len(idx))]
                if c in outcomes:
                    alt[c][idx] = _flip(alt[c][idx])
            alt[ts][idx] = _later(t, t_all, len(idx), rng)
            trials.append(("future", "rows after t were changed", alt))
        if outcomes and now.any():
            alt = {c: v.copy() for c, v in cols.items()}
            for c in outcomes:
                alt[c][now] = _flip(alt[c][now])
            trials.append(("instant", "the outcomes (%s) at t were flipped" % ", ".join(outcomes), alt))
        for kind, what, alt in trials:
            order = rng.permutation(n)
            got = _features(fn, _rebuild(df, alt, order), df, order)
            for name, ref in base.items():
                if name not in got:
                    raise PointInTimeError("feature %r is missing when %s" % (name, what))
                ok = _equal(ref, got[name])
                bad = np.flatnonzero(past & ~ok)
                if bad.size:
                    i = bad[0]
                    raise PointInTimeError(
                        "feature %r reads the future: at row %d (%s=%r, %s=%s) it was %r and became %r when %s "
                        "(cut t=%s); %d of %d rows with %s <= t changed"
                        % (name, i, keys[0], cols[keys[0]][i], ts, t_all[i], ref[i], got[name][i], what, t,
                           bad.size, int(past.sum()), ts))
        report["cuts"].append(t)
        report["rows_checked"] += int(past.sum())
    return report

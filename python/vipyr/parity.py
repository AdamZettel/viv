# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The train/serve parity check: a feature computed in a training batch must
equal the same feature computed the way a serving system computes it.

    vipyr.check_parity(fn, df, by="user", ts="ts", outcomes=["declined"],
                         lookback=pd.Timedelta("24h"), requests=300)

Training runs `fn` once on the whole frame. Serving is replayed request by
request: for a request (one key at one instant t) the serving store holds

  - the key's rows with ts < t, or with t - lookback <= ts < t when a
    lookback is given (what a store that keeps that much history holds);
  - the request's own rows, the key's rows at ts == t, with their `outcomes`
    hidden: 0/1 columns flipped, other numbers set to 0. A transaction's
    outcome is not known when it is scored.

and `serve(frame)` (by default `fn` itself) computes the request rows'
features from that frame alone. Each must equal the training value BIT FOR
BIT (NaN equal to NaN): no tolerance.

What it finds that the point-in-time check does not:

  - a feature that needs more history than the serving store keeps (a
    "24-hour" feature that silently reads 25 hours, a lifetime flag served
    from a 90-day store);
  - a feature that reads other keys' rows (a global counter, a rank across
    users): serving sees one key;
  - a serving implementation that is different code from the training one
    (`serve=`): a window edge closed on the other side, a tie broken another
    way.

`batches="1D"` replays daily batch scoring instead: for each day, `fn` runs on
every row with ts before the day's end, and that day's rows must match
training. Raises ParityError naming the feature, the key, the instant and
both values; success returns a small report.
"""
from __future__ import annotations

import numpy as np

from .frames import frame_library
from .pit import _col, _equal, _features

__all__ = ["check_parity", "ParityError"]


class ParityError(AssertionError):
    """A feature computed as serving computes it differs from training."""


def _hide(v):
    """An outcome column as serving sees it before the outcome is known."""
    if v.dtype == bool:
        return ~v
    if v.dtype.kind in "iu" and v.size and np.isin(v, (0, 1)).all():
        return 1 - v
    if v.dtype.kind in "iuf":
        return np.zeros_like(v)
    return v


def _take(df, rows, cols=None):
    """Rows `rows` of df (in that order), with some columns replaced."""
    if frame_library(df) == "pandas":
        out = df.iloc[rows].copy()
        for c, v in (cols or {}).items():
            out[c] = v
        return out
    import polars as pl
    out = df[rows]
    if cols:
        out = out.with_columns([pl.Series(c, v).cast(df.schema[c]) for c, v in cols.items()])
    return out


def _as_ns(x):
    """A lookback or batch length as int64 nanoseconds (or a plain number
    for numeric timestamps)."""
    if x is None:
        return None
    if hasattr(x, "value"):                       # pandas Timedelta
        return int(x.value)
    if isinstance(x, np.timedelta64):
        return int(x.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(x, str):
        import pandas as pd
        return int(pd.Timedelta(x).value)
    if hasattr(x, "total_seconds"):               # datetime.timedelta
        return int(round(x.total_seconds() * 1e9))
    return x


def _keys(df, by):
    keys = [by] if isinstance(by, str) else list(by)
    cols = [np.asarray(_col(df, k)) for k in keys]
    if len(cols) == 1:
        return keys, cols[0]
    # several key columns: one int64 id per distinct combination
    _, ids = np.unique(np.stack([c.astype(np.int64) for c in cols], axis=1), axis=0, return_inverse=True)
    return keys, ids.reshape(-1)


def check_parity(fn, df, by, ts, outcomes=(), lookback=None, requests=300, serve=None,
                 batches=None, seed=0):
    """Replay serving and compare with training (see the module docstring).
    `requests` is how many (key, instant) requests to replay, picked at random
    (None for all of them); `lookback` (a Timedelta, a string like "24h", or a
    number for numeric timestamps) is how much history the serving store keeps
    (None: all of it). Returns {"requests": n, "rows_checked": n, "features":
    [...]}; raises ParityError on the first difference."""
    if frame_library(df) is None:
        raise TypeError("check_parity takes a pandas or polars DataFrame")
    serve = serve or fn
    n = len(df)
    ident = np.arange(n)
    train = _features(fn, df, df, ident)
    if any(len(v) != n for v in train.values()):
        raise ValueError("check_parity: fn must return one row per input row")
    t_all = np.asarray(_col(df, ts))
    t_int = t_all.view(np.int64) if t_all.dtype.kind in "mM" else t_all
    keys, key = _keys(df, by)
    outcomes = list(outcomes)
    lb = _as_ns(lookback)
    report = {"requests": 0, "rows_checked": 0, "features": sorted(train)}
    if batches is not None:
        return _batches(fn, df, train, t_int, _as_ns(batches), report)

    order = np.lexsort((t_int, key))
    k_s, t_s = key[order], t_int[order]
    starts = np.flatnonzero(np.r_[True, (k_s[1:] != k_s[:-1]) | (t_s[1:] != t_s[:-1])])
    ends = np.r_[starts[1:], n]
    group_first = np.flatnonzero(np.r_[True, k_s[1:] != k_s[:-1]])
    first_of = group_first[np.searchsorted(group_first, starts, "right") - 1]
    pick = np.arange(len(starts))
    if requests is not None and requests < len(starts):
        pick = np.sort(np.random.default_rng(seed).choice(len(starts), requests, replace=False))
    hidden = {c: _hide(np.asarray(_col(df, c))) for c in outcomes}
    for p in pick:
        a, b, g0 = starts[p], ends[p], first_of[p]
        t = t_s[a]
        lo = g0 if lb is None else g0 + np.searchsorted(t_s[g0:a], t - lb, "left")
        hist, now = order[lo:a], order[a:b]
        rows = np.concatenate([hist, now])
        cols = {c: np.concatenate([np.asarray(_col(df, c))[hist], hidden[c][now]]) for c in outcomes}
        frame = _take(df, rows, cols)
        got = _features(serve, frame, frame, np.arange(len(rows)))
        for name, ref in train.items():
            if name not in got:
                raise ParityError("feature %r is missing from the serving result" % name)
            g = np.asarray(got[name])[len(hist):]
            ok = _equal(ref[now], g)
            if not ok.all():
                bad = now[~ok]
                what = "serving from %d rows of history%s" % (
                    len(hist), "" if lb is None else " (lookback %s)" % lookback)
                raise ParityError(
                    "feature %r: training gave %r, serving gave %r at row %d (%s=%r, %s=%s) when %s, "
                    "the request's outcomes hidden; %d of %d request rows differ"
                    % (name, ref[bad[0]], g[~ok][0], bad[0], keys[0], np.asarray(_col(df, keys[0]))[bad[0]],
                       ts, t_all[bad[0]], what, int((~ok).sum()), len(now)))
        report["requests"] += 1
        report["rows_checked"] += len(now)
    return report


def _batches(fn, df, train, t_int, step, report):
    """Daily (or `step`-long) batch scoring: each batch's rows computed from
    every row before the batch's end."""
    order = np.argsort(t_int, kind="stable")
    t_s = t_int[order]
    edge = (t_s[0] // step) * step + step
    while True:
        a = np.searchsorted(t_s, edge - step, "left")
        b = np.searchsorted(t_s, edge, "left")
        if a >= len(t_s):
            break
        if b > a:
            rows = np.sort(order[:b])
            frame = _take(df, rows)
            got = _features(fn, frame, frame, np.arange(len(rows)))
            now = np.isin(rows, order[a:b])
            for name, ref in train.items():
                g = np.asarray(got[name])
                ok = _equal(ref[rows[now]], g[now])
                if not ok.all():
                    i = rows[now][~ok][0]
                    raise ParityError(
                        "feature %r: training gave %r, the batch ending %s gave %r at row %d; "
                        "%d of %d batch rows differ"
                        % (name, ref[i], edge, g[now][~ok][0], i, int((~ok).sum()), int(now.sum())))
            report["requests"] += 1
            report["rows_checked"] += int(now.sum())
        edge += step
    return report

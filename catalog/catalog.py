# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The feature catalog from Python: each entry's Viv function on a pandas or
polars frame, its pandas reference, a polars version where one exists, and
synthetic data to test it on.

    import catalog
    feats = catalog.sessions(df, gap="30min")      # df: user, ts
    e = catalog.ENTRIES["sessions"]                # fn, reference, by, ts, lookback, ...

Every Viv entry takes rows in any order. It sorts them into the canonical
order (key, ts, then the other input columns, so rows at the same instant
are always taken in the same order), calls the kernel and puts the result
back in the input's row order. A pandas result keeps the input's index.

The references are written for reading, not speed: a groupby().apply() with
a Python loop per row that states the definition directly.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import vipyr  # noqa: E402
from vipyr.frames import column_array, frame_library  # noqa: E402

NS = 10 ** 9
HOUR = 3600 * NS
DAY = 24 * HOUR
LN2 = 0.6931471805599453

_mods = {}


def module(name):
    if name not in _mods:
        _mods[name] = vipyr.load(HERE / (name + ".viv"))
    return _mods[name]


def _ns(x):
    return int(pd.Timedelta(x).value) if isinstance(x, str) else int(getattr(x, "value", x))


# ------------------------------------------------------------- the plumbing

def _arrays(df, cols):
    return {c: column_array(df, c)[0] for c in cols}


def _canonical(a, key, ts, rest):
    """The canonical row order: key, ts, then the other columns (None when
    the rows are in it already, which costs one pass instead of a sort)."""
    cols = [a[key], a[ts]] + [a[c] for c in rest]
    if len(cols[0]) < 2:
        return None
    ok = np.zeros(len(cols[0]) - 1, bool)
    eq = np.ones(len(cols[0]) - 1, bool)
    for x in cols:
        ok |= eq & (x[1:] > x[:-1])
        eq &= x[1:] == x[:-1]
    if (ok | eq).all():
        return None
    return np.lexsort(tuple(reversed(cols)))


def _frame(df, out, order=None):
    """Kernel outputs (in canonical order when `order` is given) as a frame
    of df's library, in df's row order."""
    cols = {}
    for c, v in out.items():
        if order is not None:
            back = np.empty_like(v)
            back[order] = v
            v = back
        cols[c] = v
    if frame_library(df) == "polars":
        import polars as pl
        return pl.DataFrame(cols)
    return pd.DataFrame(cols, index=df.index)


def _run(df, name, kernel, key, ts, rest, extra=None, **kw):
    """Call `kernel` of module `name` on df's columns in canonical order."""
    a = _arrays(df, [key, ts] + rest)
    a.update(extra or {})
    order = _canonical(a, key, ts, rest + sorted(extra or {}))
    args = [a[c] if order is None else a[c][order] for c in [key, ts] + rest + sorted(extra or {})]
    return _frame(df, getattr(module(name), kernel)(*args, **kw), order)


def _as_datetime(out, cols):
    for c in cols:
        if frame_library(out) == "polars":
            import polars as pl
            out = out.with_columns(pl.col(c).cast(pl.Datetime("ns")))
        else:
            out[c] = out[c].to_numpy().view("datetime64[ns]")
    return out


def _per_key(df, key, ts, rest, f):
    """A reference: rows in canonical order, f(group) per key, input order back."""
    a = _arrays(df, [key, ts] + rest)
    order = _canonical(a, key, ts, rest)
    if order is None:
        order = np.arange(len(a[key]))
    s = pd.DataFrame({c: a[c][order] for c in a}, index=np.arange(len(order)))
    parts = [f(g.reset_index(drop=True)).set_index(g.index) for _, g in s.groupby(key, sort=False)]
    out = pd.concat(parts).sort_index()
    back = pd.DataFrame(index=df.index if frame_library(df) == "pandas" else None)
    for c in out.columns:
        v = np.empty(len(order), out[c].to_numpy().dtype)
        v[order] = out[c].to_numpy()
        back[c] = v
    return back.reset_index(drop=True) if frame_library(df) != "pandas" else back


# ------------------------------------------------------------- synthetic data

EDGES = np.array([1800, 1801, 3600, 3601, 86400, 86401])   # seconds: the catalog's window edges, and one past


def make_events(n, users=None, days=120, seed=0):
    """About n events sorted by (user, ts), whole seconds over `days` days.
    Users have their own activity rates; a third of rows come in bursts after
    the previous row: 15% of those at the same second, 15% exactly at a window
    edge (30 minutes, 1 hour, 24 hours) or one second past it, the rest within
    30 minutes. One user in twenty is also active on 90% to 100% of days, so streaks
    run long enough to cross the 90-day window. Each row has an amount, a
    merchant, an event kind, a value and a reset flag."""
    rng = np.random.default_rng(seed)
    users = users or max(1, n // 60)
    user = np.sort(rng.integers(0, users, n))
    rate = rng.uniform(0.2, 1.0, users)[user]
    t = (rng.random(n) ** rate * days * 86400).astype(np.int64)
    burst = rng.random(n) < 0.35
    u = rng.random(n)
    step = np.where(u < 0.15, 0, np.where(u < 0.30, rng.choice(EDGES, n), rng.integers(1, 1800, n)))
    t = np.where(burst, np.roll(t, 1) + step, t)
    daily_u = np.arange(0, users, 20)
    dd = np.argwhere(rng.random((len(daily_u), days)) < rng.uniform(0.9, 1.0, (len(daily_u), 1)))
    user = np.r_[user, daily_u[dd[:, 0]]]
    t = np.r_[t, dd[:, 1] * 86400 + rng.integers(0, 86400, len(dd))]
    t = np.clip(t, 0, days * 86400 - 1)
    order = np.lexsort((t, user))
    m = len(t)
    kind = rng.choice(np.array(["view", "cart", "purchase"]), m, p=[0.7, 0.2, 0.1])
    return pd.DataFrame({
        "user": user[order].astype(np.int64),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(t[order], unit="s"),
        "amount": np.round(rng.gamma(2.0, 30.0, m), 2),
        "merchant": rng.integers(0, 400, m).astype(np.int64),
        "kind": kind,
        "value": np.round(rng.normal(50.0, 15.0, m), 3),
        "reset": rng.random(m) < 0.01,
    })


def make_accounts(n, seed=0):
    """About n bill and payment rows: monthly bills per account, paid on
    time, late, or (for some accounts) months late and then caught up."""
    rng = np.random.default_rng(seed)
    rows = []
    acct = 0
    while len(rows) < n:
        start = int(rng.integers(0, 60)) * DAY
        kind = rng.choice(3, p=[0.6, 0.25, 0.15])
        months = int(rng.integers(6, 25))
        for m in range(months):
            due = start + m * 30 * DAY + int(rng.integers(0, 3600)) * NS
            amount = int(rng.integers(50, 500)) * 100
            rows.append((acct, due, amount, 0))
            if kind == 0:
                delay = rng.integers(0, 25)
            elif kind == 1:
                delay = rng.integers(10, 50)
            else:
                delay = rng.integers(0, 25) if rng.random() < 0.6 else rng.integers(55, 140)
            share = 1.0 if rng.random() < 0.85 else 0.5
            if rng.random() < 0.97:
                rows.append((acct, due + int(delay) * DAY + int(rng.integers(0, 86400)) * NS, int(amount * share), 1))
        acct += 1
    a = np.array(rows, dtype=np.int64)
    order = np.lexsort((a[:, 1], a[:, 0]))
    a = a[order]
    return pd.DataFrame({
        "account": a[:, 0],
        "ts": pd.Timestamp("2025-01-01") + pd.to_timedelta(a[:, 1], unit="ns"),
        "billed": np.where(a[:, 3] == 0, a[:, 2], 0),
        "paid": np.where(a[:, 3] == 1, a[:, 2], 0),
    })


def make_quotes(left, n, seed=0):
    """A right table for the as-of join: n (user, ts, value) rows, a fifth of
    them stamped exactly at a left row's time and a tenth 1 or 60 seconds
    after one, some repeating an instant."""
    rng = np.random.default_rng(seed)
    lu = left["user"].to_numpy()
    lt = left["ts"].to_numpy().view(np.int64)
    pick = rng.integers(0, len(lu), n)
    t = lt[pick] - rng.integers(0, 3 * 86400, n) * NS
    u = rng.random(n)
    t = np.where(u < 0.2, lt[pick], np.where(u < 0.3, lt[pick] + rng.choice([1, 60], n) * NS, t))
    dup = rng.random(n) < 0.05
    t = np.where(dup, np.roll(t, 1), t)
    u = np.where(dup, np.roll(lu[pick], 1), lu[pick])
    order = np.lexsort((t, u))
    return pd.DataFrame({
        "user": u[order].astype(np.int64),
        "ts": t[order].view("datetime64[ns]"),
        "value": np.round(rng.normal(100.0, 10.0, n), 1),
    })


# ------------------------------------------------------------- 1. sessions

def sessions(df, gap="30min"):
    """Gap-based sessions per user (catalog/sessions.viv)."""
    out = _run(df, "sessions", "sessions", "user", "ts", [], gap=_ns(gap))
    return _as_datetime(out, ["session_start"])


def sessions_reference(df, gap="30min"):
    g_ns = _ns(gap)

    def f(g):
        t = g["ts"].to_numpy()
        sid = np.zeros(len(t), np.int64)
        for i in range(1, len(t)):
            sid[i] = sid[i - 1] + (1 if t[i] - t[i - 1] > g_ns else 0)
        start = np.array([t[sid == sid[i]].min() for i in range(len(t))])
        nprev = np.array([int(((sid == sid[i]) & (t < t[i])).sum()) for i in range(len(t))])
        secs = np.array([float(t[i] - start[i]) / float(NS) for i in range(len(t))])
        return pd.DataFrame({"session_start": start, "session_index": sid, "secs_since_start": secs,
                             "n_prev_in_session": nprev})
    return _as_datetime(_per_key(df, "user", "ts", [], f), ["session_start"])


def sessions_polars(df, gap="30min"):
    import polars as pl
    d = df if frame_library(df) == "polars" else pl.from_pandas(df)
    g_ns = _ns(gap)
    out = (
        d.select("user", "ts").with_row_index("_row").sort("user", "ts")
        .with_columns(_t=pl.col("ts").cast(pl.Int64))
        .with_columns(_new=(pl.col("_t").diff().over("user").fill_null(g_ns + 1) > g_ns).cast(pl.Int64))
        .with_columns(session_index=pl.col("_new").cum_sum().over("user") - 1)
        .with_columns(session_start=pl.col("_t").min().over("user", "session_index"))
        .with_columns(n_prev_in_session=pl.col("_t").rank("min").over("user", "session_index").cast(pl.Int64) - 1,
                      secs_since_start=(pl.col("_t") - pl.col("session_start")).cast(pl.Float64) / NS)
        .sort("_row")
    )
    return out.select(pl.col("session_start").cast(pl.Datetime("ns")), "session_index", "secs_since_start",
                      "n_prev_in_session")


# ------------------------------------------------------------- 2. velocity

def velocity(df):
    """Trailing-window counts, sum and distinct count (catalog/velocity.viv)."""
    return _run(df, "velocity", "velocity", "user", "ts", ["amount", "merchant"])


def velocity_reference(df):
    def f(g):
        t, amt, mer = g["ts"].to_numpy(), g["amount"].to_numpy(), g["merchant"].to_numpy()
        rows = []
        for i in range(len(t)):
            w1 = (t >= t[i] - HOUR) & (t < t[i])
            w24 = (t >= t[i] - DAY) & (t < t[i])
            total = 0.0
            for a in amt[w24]:
                total += a
            rows.append((int(w1.sum()), int(w24.sum()), total, len(set(mer[w24].tolist()))))
        return pd.DataFrame(rows, columns=["n_1h", "n_24h", "sum_24h", "distinct_24h"])
    return _per_key(df, "user", "ts", ["amount", "merchant"], f)


def velocity_polars(df):
    """polars rolling windows closed on the left, [t - W, t): the right edges.
    The sum is polars' running sum, so its last bits may differ from a sum
    taken afresh; there is no rolling distinct count by time."""
    import polars as pl
    d = df if frame_library(df) == "polars" else pl.from_pandas(df)
    one = pl.col("_one")
    out = (
        d.with_row_index("_row").sort("user", "ts", "amount", "merchant")
        .with_columns(_one=pl.lit(1, dtype=pl.Int64))
        .with_columns(
            n_1h=one.rolling_sum_by("ts", window_size="1h", closed="left").over("user"),
            n_24h=one.rolling_sum_by("ts", window_size="24h", closed="left").over("user"),
            sum_24h=pl.col("amount").rolling_sum_by("ts", window_size="24h", closed="left").over("user"),
        )
        .sort("_row")
    )
    return out.select(pl.col("n_1h").fill_null(0).cast(pl.Int64), pl.col("n_24h").fill_null(0).cast(pl.Int64),
                      pl.col("sum_24h").fill_null(0.0))


# ------------------------------------------------------------- 3. streaks

def streaks(df):
    """Consecutive active days (catalog/streaks.viv)."""
    return _run(df, "streaks", "streaks", "user", "ts", [])


def streaks_reference(df):
    def f(g):
        day = g["ts"].to_numpy() // DAY
        cur, best = [], []
        for d in day:
            active = set(day[day <= d].tolist())
            s = 0
            while d - s in active:
                s += 1
            longest = run = 0
            for x in range(d - 89, d + 1):
                run = run + 1 if x in active else 0
                longest = max(longest, run)
            cur.append(s)
            best.append(longest)
        return pd.DataFrame({"streak_days": cur, "longest_90d": best})
    return _per_key(df, "user", "ts", [], f)


# ------------------------------------------------------------- 4. EWMA

def ewma(df, half_life="1D"):
    """Gap-aware EWMA with resets (catalog/ewma.viv)."""
    return _run(df, "ewma", "ewma", "user", "ts", ["value", "reset"], half_life=_ns(half_life))


def ewma_reference(df, half_life="1D"):
    h = _ns(half_life)

    def f(g):
        t, x, r = g["ts"].to_numpy(), g["value"].to_numpy(), g["reset"].to_numpy()
        means, weights = [], []
        s = w = 0.0
        t_prev = t[0]
        i = 0
        while i < len(t):
            j = i
            while j < len(t) and t[j] == t[i]:
                j += 1
            fct = math.exp(-LN2 * float(t[i] - t_prev) / float(h))
            s, w, t_prev = s * fct, w * fct, t[i]
            means += [s / w if w > 0 else math.nan] * (j - i)
            weights += [w] * (j - i)
            if r[i:j].any():
                s = w = 0.0
            for k in range(i, j):
                s += x[k]
                w += 1.0
            i = j
        return pd.DataFrame({"ewma": means, "ewm_weight": weights})
    return _per_key(df, "user", "ts", ["value", "reset"], f)


# ------------------------------------------------------------- 5. time since last

def since_last(df, kind="purchase"):
    """Seconds since the last event of a kind, and of any kind (catalog/since_last.viv)."""
    hit = (column_array(df, "kind")[0] == _code(df, kind)).astype(np.int64)
    return _run(df, "since_last", "since_last", "user", "ts", [], extra={"hit": hit})


def _code(df, kind):
    vals, _, meta = column_array(df, "kind")
    cats = list(meta.categories) if meta is not None else []
    return cats.index(kind) if kind in cats else -1


def since_last_reference(df, kind="purchase"):
    hit = (np.asarray(df["kind"].to_numpy()) == kind).astype(np.int64)
    d = df[["user", "ts"]] if frame_library(df) == "pandas" else df.select("user", "ts").to_pandas()
    d = d.assign(hit=hit)

    def f(g):
        t, h = g["ts"].to_numpy(), g["hit"].to_numpy()
        a, b = [], []
        for i in range(len(t)):
            prior = t < t[i]
            a.append(float(t[i] - t[prior & (h == 1)].max()) / float(NS) if (prior & (h == 1)).any() else math.nan)
            b.append(float(t[i] - t[prior].max()) / float(NS) if prior.any() else math.nan)
        return pd.DataFrame({"secs_since_hit": a, "secs_since_prev": b})
    out = _per_key(d, "user", "ts", ["hit"], f)
    return out.reset_index(drop=True) if frame_library(df) != "pandas" else out


# ------------------------------------------------------------- 6. delinquency

def delinquency(df):
    """Days past due, worst in 12 months, ever cured after 60+ (catalog/delinquency.viv)."""
    return _run(df, "delinquency", "delinquency", "account", "ts", ["billed", "paid"])


def delinquency_reference(df):
    """Written from cumulative sums, not from a queue: under oldest-first
    payment with credit carried forward, bill k is closed at time s exactly
    when the payments up to s add up to at least the bills up to k."""
    def f(g):
        t, bil, pay = g["ts"].to_numpy(), g["billed"].to_numpy(), g["paid"].to_numpy()

        def dpd(s, upto_inclusive):
            known = (t <= s) if upto_inclusive else (t < s)
            bills = np.flatnonzero(known & (bil > 0))
            paid = pay[known].sum()
            open_ = [k for c, k in zip(np.cumsum(bil[bills]), bills) if c > paid]
            return 0 if not open_ else int((s - t[open_[0]]) // DAY)
        out = []
        for i in range(len(t)):
            now = dpd(t[i], False)
            instants = sorted(set(t[(t < t[i]) & (t >= t[i] - 365 * DAY)].tolist()))
            worst = max([now] + [dpd(p, False) for p in instants])
            cured = int(any(dpd(p, False) >= 60 and dpd(p, True) == 0 for p in set(t[t < t[i]].tolist())))
            out.append((now, worst, cured))
        return pd.DataFrame(out, columns=["dpd", "worst_dpd_12m", "ever_cured_60"])
    return _per_key(df, "account", "ts", ["billed", "paid"], f)


# ------------------------------------------------------------- 7. as-of join

def asof(left, right, strict=False):
    """Each left row's latest right value of the same user (catalog/asof.viv)."""
    la = _arrays(left, ["user", "ts"])
    ra = _arrays(right, ["user", "ts", "value"])
    lo = _canonical(la, "user", "ts", [])
    ro = _canonical(ra, "user", "ts", ["value"])
    L = {c: v if lo is None else v[lo] for c, v in la.items()}
    R = {c: v if ro is None else v[ro] for c, v in ra.items()}
    out = module("asof").asof(L, R, int(bool(strict)))
    return _frame(left, out, lo)


def asof_reference(left, right, strict=False):
    lu, lt = left["user"].to_numpy(), np.asarray(left["ts"].to_numpy()).view(np.int64)
    ru, rt, rv = right["user"].to_numpy(), np.asarray(right["ts"].to_numpy()).view(np.int64), right["value"].to_numpy()
    by_user = {u: np.flatnonzero(ru == u) for u in np.unique(ru)}
    val, age = [], []
    for u, t in zip(lu, lt):
        idx = by_user.get(u, np.array([], np.int64))
        ok = idx[(rt[idx] < t) if strict else (rt[idx] <= t)]
        if not len(ok):
            val.append(math.nan)
            age.append(math.nan)
            continue
        latest = rt[ok].max()
        val.append(rv[ok][rt[ok] == latest].max())
        age.append(float(t - latest) / float(NS))
    return pd.DataFrame({"value": val, "age_secs": age},
                        index=left.index if frame_library(left) == "pandas" else None)


def asof_polars(left, right, strict=False):
    """polars join_asof, backward, by user. Ties at the matching instant are
    resolved by polars' own rule (the last row), so the right table is
    sorted by value within an instant first to match the catalog's rule."""
    import polars as pl
    lf = left if frame_library(left) == "polars" else pl.from_pandas(left)
    rf = right if frame_library(right) == "polars" else pl.from_pandas(right)
    lf = lf.select("user", "ts").with_row_index("_row").sort("ts")
    rf = rf.select("user", "ts", "value").with_columns(_rts=pl.col("ts")).sort("ts", "value")
    out = lf.join_asof(rf, on="ts", by="user", strategy="backward", allow_exact_matches=not strict).sort("_row")
    return out.select("value", age_secs=(pl.col("ts") - pl.col("_rts")).dt.total_nanoseconds().cast(pl.Float64) / NS)


def asof_combined(left, right):
    """Left and right rows in one frame (side 0 left, 1 right), the shape the
    point-in-time and parity checks take: they then alter right rows after t
    as well as left ones."""
    l = pd.DataFrame({"user": left["user"].to_numpy(), "ts": left["ts"].to_numpy(), "side": 0, "value": 0.0})
    r = pd.DataFrame({"user": right["user"].to_numpy(), "ts": right["ts"].to_numpy(), "side": 1,
                      "value": right["value"].to_numpy()})
    return pd.concat([l, r], ignore_index=True)


def asof_on_combined(strict=False):
    """A feature function over a combined frame: the as-of value on left rows,
    NaN on right rows."""
    def fn(df):
        side = column_array(df, "side")[0]
        lmask, rmask = side == 0, side == 1
        a = _arrays(df, ["user", "ts", "value"])
        left = pd.DataFrame({"user": a["user"][lmask], "ts": a["ts"][lmask].view("datetime64[ns]")})
        right = pd.DataFrame({"user": a["user"][rmask], "ts": a["ts"][rmask].view("datetime64[ns]"),
                              "value": a["value"][rmask]})
        res = asof(left, right, strict)
        out = {c: np.full(len(side), np.nan) for c in res.columns}
        for c in res.columns:
            out[c][lmask] = res[c].to_numpy()
        return _frame(df, out)
    return fn


# ------------------------------------------------------------- 8. labels

def churn_label(df, horizon="14D"):
    """LABEL, not a feature: 1 when the user is silent for `horizon` after the
    row (catalog/labels.viv). It reads the future by design."""
    return _run(df, "labels", "churn", "user", "ts", [], horizon=_ns(horizon))


def churn_label_reference(df, horizon="14D"):
    h = _ns(horizon)
    end = np.asarray(df["ts"].to_numpy()).view(np.int64).max()

    def f(g):
        t = g["ts"].to_numpy()
        out = []
        for x in t:
            if ((t > x) & (t <= x + h)).any():
                out.append(0.0)
            else:
                out.append(1.0 if x + h <= end else math.nan)
        return pd.DataFrame({"churned": out})
    return _per_key(df, "user", "ts", [], f)


# ------------------------------------------------------------- the registry

@dataclass
class Entry:
    name: str
    fn: object                  # frame -> features, the Viv version
    reference: object           # frame -> features, the pandas reference
    make: object                # (n, seed) -> frame
    by: str
    ts: str = "ts"
    outcomes: list = field(default_factory=list)
    lookback: object = None     # serving history the feature needs; None = all of it
    kind: str = "feature"       # or "label"
    polars: object = None
    exact: bool = True          # False: the reference uses the platform's exp


def _events(n, seed):
    return make_events(n, seed=seed)


def _joined(n, seed):
    left = make_events(n, seed=seed)[["user", "ts"]]
    return asof_combined(left, make_quotes(left, n // 2, seed=seed + 1))


def _asof_ref_combined(strict):
    def ref(df):
        side = df["side"].to_numpy()
        left, right = df[side == 0], df[side == 1]
        res = asof_reference(left, right, strict)
        out = pd.DataFrame({c: np.full(len(df), np.nan) for c in res.columns}, index=df.index)
        out.loc[side == 0, :] = res.to_numpy()
        return out
    return ref


ENTRIES = {e.name: e for e in [
    Entry("sessions", sessions, sessions_reference, _events, "user", polars=sessions_polars),
    Entry("velocity", velocity, velocity_reference, _events, "user", lookback=pd.Timedelta("24h"),
          polars=velocity_polars),
    Entry("streaks", streaks, streaks_reference, _events, "user"),
    Entry("ewma", ewma, ewma_reference, _events, "user", outcomes=["value", "reset"], exact=False),
    Entry("since_last", since_last, since_last_reference, _events, "user"),
    Entry("delinquency", delinquency, delinquency_reference, lambda n, seed: make_accounts(n, seed), "account",
          outcomes=["billed", "paid"]),
    Entry("asof", asof_on_combined(False), _asof_ref_combined(False), _joined, "user"),
    Entry("asof_strict", asof_on_combined(True), _asof_ref_combined(True), _joined, "user", outcomes=["value"]),
    Entry("churn_label", churn_label, churn_label_reference, _events, "user", kind="label"),
]}

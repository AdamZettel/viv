# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Card velocity features three ways: Viv, the pandas groupby().apply()
reference, and a polars version written the way an assistant often writes it.

Every function takes a frame with columns card (int64), ts (datetime64),
declined (bool), chargeback (bool) and returns a frame with n_prev_1h and
status, one row per input row, in the input's row order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
import vipyr  # noqa: E402

HOUR = 3600 * 10 ** 9
DAY = 24 * HOUR
CLEAN = 7 * DAY
OK, WATCH, LOCKED = 0, 1, 2

_mod = None


def module():
    global _mod
    if _mod is None:
        _mod = vipyr.load(HERE / "features.viv")
    return _mod


# --------------------------------------------------------------------- data

def make_transactions(n, cards=None, seed=0):
    """n synthetic transactions, sorted by (card, ts), at whole seconds over
    30 days: about 50 per card, a third of them in bursts minutes apart (some
    at the same second), declined (30% in bursts, 3% otherwise), 0.3% charged back."""
    rng = np.random.default_rng(seed)
    cards = cards or max(1, n // 50)
    card = np.sort(rng.integers(0, cards, n))
    t = rng.integers(0, 30 * 86400, n)
    burst = rng.random(n) < 0.35
    # a burst row follows the row before it by 0 to 20 minutes
    t = np.where(burst, np.roll(t, 1) + rng.integers(0, 1200, n), t)
    declined = rng.random(n) < np.where(burst, 0.3, 0.03)
    chargeback = rng.random(n) < 0.003
    order = np.lexsort((t, card))
    return pd.DataFrame({
        "card": card[order].astype(np.int64),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(t[order], unit="s"),
        "amount": np.round(rng.gamma(2.0, 30.0, n), 2),
        "declined": declined[order],
        "chargeback": chargeback[order],
    })


# --------------------------------------------------------------------- Viv

def viv_features(df):
    """The Viv kernel on a pandas or polars frame. Sorted input goes straight
    in (card and ts zero copy for pandas int64/datetime64[ns]); otherwise
    the rows are sorted first and the result put back in input order."""
    k = module()
    card = np.asarray(df["card"].to_numpy(), dtype=np.int64)
    ts = vipyr.frames.column_array(df, "ts")[0]
    if card.size < 2 or (np.all(card[1:] >= card[:-1]) and np.all((card[1:] > card[:-1]) | (ts[1:] >= ts[:-1]))):
        return k.card_features(df)
    order = np.lexsort((ts, card))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    if vipyr.frames.frame_library(df) == "pandas":
        return k.card_features(df.iloc[order]).iloc[inv]
    import polars as pl
    res = k.card_features(df.select(pl.all().gather(order)))
    return res.select(pl.all().gather(inv))


# ---------------------------------------------------- the pandas reference

def _per_card(g):
    """One card, its rows in time order: the definition, written plainly."""
    t = g["ts"].to_numpy().view(np.int64)
    dec = g["declined"].to_numpy()
    cb = g["chargeback"].to_numpy()
    # rows with t - 1h <= t_j < t
    n_prev = np.searchsorted(t, t, "left") - np.searchsorted(t, t - HOUR, "left")
    status = np.zeros(len(t), np.int64)
    locked, last_bad, j = False, 0, 0          # rows before j are the past
    for i in range(len(t)):
        while j < len(t) and t[j] < t[i]:      # the past grows to t[i]
            if locked and t[j] - last_bad >= CLEAN:
                locked = False
            if cb[j]:
                locked, last_bad = True, t[j]
            elif dec[j] and locked:
                last_bad = t[j]
            j += 1
        if locked and t[i] - last_bad >= CLEAN:
            locked = False
        recent = dec[(t >= t[i] - DAY) & (t < t[i])].sum()
        status[i] = LOCKED if locked else WATCH if recent >= 3 else OK
    return pd.DataFrame({"n_prev_1h": n_prev, "status": status}, index=g.index)


def pandas_reference(df):
    """groupby().apply(): correct, and slow (a Python loop per row)."""
    s = df.sort_values(["card", "ts"], kind="stable")
    out = s.groupby("card", group_keys=False).apply(_per_card, include_groups=False)
    return out.reindex(df.index)


# ------------------------------------------------------ the polars version

def polars_llm(df):
    """What an assistant commonly writes: rolling windows over each card.
    It looks right and runs fast. It is wrong in three ways: `rolling_*_by`
    windows are closed on the right, so they hold the current row and every
    row at the same instant ("- 1" removes only one of them); the 24-hour
    decline count includes the current transaction's own outcome; and the
    lock is "a chargeback in the last 7 days", including this row's."""
    import polars as pl
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    out = (
        df.with_row_index("_row")
        .sort("card", "ts")
        .with_columns(pl.lit(1).alias("_one"))
        .with_columns(
            n_prev_1h=(pl.col("_one").rolling_sum_by("ts", window_size="1h").over("card") - 1),
            _dec24=pl.col("declined").cast(pl.Int64).rolling_sum_by("ts", window_size="24h").over("card"),
            _cb7=pl.col("chargeback").cast(pl.Int64).rolling_max_by("ts", window_size="7d").over("card"),
        )
        .with_columns(
            status=pl.when(pl.col("_cb7") > 0).then(LOCKED)
            .when(pl.col("_dec24") >= 3).then(WATCH)
            .otherwise(OK)
            .cast(pl.Int64)
        )
        .sort("_row")
    )
    return out.select(pl.col("n_prev_1h").cast(pl.Int64), "status")

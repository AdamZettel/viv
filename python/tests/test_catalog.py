# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The feature catalog (catalog/): every entry against its pandas reference,
the point-in-time check, and the train/serve parity check.

    VIV=target/release/viv python3 -m pytest -q python/tests/test_catalog.py

For each entry: the Viv function equals the reference on synthetic data
(bit for bit, except the EWMA, whose reference uses the platform's exp);
vipyr.check_point_in_time passes; vipyr.check_parity passes with the
lookback the entry declares. The label is refused by both checks. The
remaining tests show each check catching the mistake it exists for.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=".*bottleneck.*")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "catalog"))

import vipyr  # noqa: E402
import catalog  # noqa: E402
from vipyr.pit import _equal  # noqa: E402

N = 4000
FEATURES = [n for n, e in catalog.ENTRIES.items() if e.kind == "feature"]
DAY = pd.Timedelta("1D")


@pytest.fixture(scope="module")
def data():
    return {name: e.make(N, 7) for name, e in catalog.ENTRIES.items()}


def _cols(a, b):
    for c in b.columns:
        x, y = np.asarray(a[c].to_numpy()), np.asarray(b[c].to_numpy())
        if x.dtype.kind == "M" or y.dtype.kind == "M":
            x, y = x.astype("datetime64[ns]").view(np.int64), y.astype("datetime64[ns]").view(np.int64)
        yield c, x, y


# ------------------------------------------------------------ the entries

@pytest.mark.parametrize("name", list(catalog.ENTRIES))
def test_equals_reference(name, data):
    e, df = catalog.ENTRIES[name], data[name]
    got, ref = e.fn(df), e.reference(df)
    assert list(got.columns) == list(ref.columns)
    for c, x, y in _cols(got, ref):
        if e.exact:
            bad = np.flatnonzero(~_equal(x, y))
            assert bad.size == 0, "%s.%s: %d rows differ, first row %d: viv %r, reference %r" % (
                name, c, bad.size, bad[0], x[bad[0]], y[bad[0]])
        else:
            # Viv's exp and the platform's differ in the last bit now and then
            np.testing.assert_allclose(x, y, rtol=1e-13, atol=0, equal_nan=True)


@pytest.mark.parametrize("name", FEATURES)
def test_point_in_time(name, data):
    e = catalog.ENTRIES[name]
    rep = vipyr.check_point_in_time(e.fn, data[name], by=e.by, ts=e.ts, outcomes=e.outcomes, cuts=3)
    assert rep["rows_checked"] > 0


@pytest.mark.parametrize("name", FEATURES)
def test_parity(name, data):
    e = catalog.ENTRIES[name]
    rep = vipyr.check_parity(e.fn, data[name], by=e.by, ts=e.ts, outcomes=e.outcomes,
                               lookback=e.lookback, requests=120)
    assert rep["requests"] == 120


def test_parity_in_daily_batches(data):
    e = catalog.ENTRIES["velocity"]
    rep = vipyr.check_parity(e.fn, data["velocity"].iloc[:1500], by=e.by, ts=e.ts, batches="1D")
    assert rep["requests"] > 10


def test_polars_frames_give_the_same_features(data):
    df = data["velocity"]
    a = catalog.velocity(df)
    b = catalog.velocity(pl.from_pandas(df))
    assert isinstance(b, pl.DataFrame)
    for c in a.columns:
        assert (a[c].to_numpy() == b[c].to_numpy()).all()


def test_rows_in_any_order(data):
    df = data["sessions"]
    shuffled = df.sample(frac=1.0, random_state=3)
    a = catalog.sessions(df)
    b = catalog.sessions(shuffled).reindex(df.index)
    for c in a.columns:
        assert (a[c].to_numpy() == b[c].to_numpy()).all()


# ------------------------------------------------------------ the label

def test_label_is_refused_as_a_feature(data):
    e = catalog.ENTRIES["churn_label"]
    with pytest.raises(vipyr.PointInTimeError, match="churned"):
        vipyr.check_point_in_time(e.fn, data["churn_label"], by="user", ts="ts", cuts=2)
    with pytest.raises(vipyr.ParityError, match="churned"):
        vipyr.check_parity(e.fn, data["churn_label"], by="user", ts="ts", requests=200)


# ------------------------------------------------------------ what the checks catch

def test_window_edges_at_the_boundary():
    """Rows exactly 1 hour and 24 hours back are in the window, one more
    nanosecond and they are out; rows at the same instant are not in it."""
    t0 = pd.Timestamp("2026-01-02")
    df = pd.DataFrame({"user": [1] * 5,
                       "ts": [t0 - DAY, t0 - pd.Timedelta("1h"), t0, t0, t0 + pd.Timedelta("1ns")],
                       "amount": [1.0, 2.0, 4.0, 8.0, 16.0], "merchant": [1, 2, 3, 3, 4]})
    v = catalog.velocity(df)
    assert v["n_1h"].tolist() == [0, 0, 1, 1, 2]
    assert v["n_24h"].tolist() == [0, 1, 2, 2, 3]
    assert v["sum_24h"].tolist() == [0.0, 1.0, 3.0, 3.0, 14.0]
    assert v["distinct_24h"].tolist() == [0, 1, 2, 2, 2]
    assert (catalog.velocity_reference(df).to_numpy() == v.to_numpy()).all()


def test_parity_catches_a_serving_window_closed_on_the_wrong_side(data):
    """The assistant-style polars window (closed on the right, minus one)
    counts the other rows of the same instant. Served with it, the feature
    differs from Viv's training value."""
    def naive(df):
        d = pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df
        out = d.with_row_index("_r").sort("user", "ts").with_columns(_one=pl.lit(1)).with_columns(
            n_1h=pl.col("_one").rolling_sum_by("ts", window_size="1h").over("user") - 1).sort("_r")
        return pd.DataFrame({"n_1h": out["n_1h"].cast(pl.Int64).to_numpy()}, index=df.index)

    def viv_n_1h(df):
        return catalog.velocity(df)[["n_1h"]]

    with pytest.raises(vipyr.ParityError, match="n_1h"):
        vipyr.check_parity(viv_n_1h, data["velocity"], by="user", ts="ts", serve=naive, requests=None)


def test_parity_checks_the_declared_lookback():
    """A 100-day streak served from a 90-day store comes out short; the
    windowed longest streak needs only the window."""
    days = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(100), unit="D")
    df = pd.DataFrame({"user": 1, "ts": days})
    with pytest.raises(vipyr.ParityError, match="streak_days"):
        vipyr.check_parity(catalog.streaks, df, by="user", ts="ts", lookback="90D", requests=None)
    rep = vipyr.check_parity(lambda d: catalog.streaks(d)[["longest_90d"]], df, by="user", ts="ts",
                               lookback="90D", requests=None)
    assert rep["requests"] == 100


def test_point_in_time_catches_a_global_session_counter(data):
    """A session id numbered across all users changes when an earlier user
    gains sessions later on."""
    def global_id(df):
        s = catalog.sessions(df)
        a = catalog._arrays(df, ["user", "ts"])
        order = np.lexsort((a["ts"], a["user"]))
        idx, user = s["session_index"].to_numpy()[order], a["user"][order]
        new = np.r_[True, (user[1:] != user[:-1]) | (idx[1:] != idx[:-1])]
        gid = np.empty(len(order), np.int64)
        gid[order] = np.cumsum(new) - 1
        return pd.DataFrame({"session_id": gid}, index=df.index)

    with pytest.raises(vipyr.PointInTimeError, match="session_id"):
        vipyr.check_point_in_time(global_id, data["sessions"], by="user", ts="ts", cuts=3)


def test_parity_catches_an_inclusive_join_on_late_values(data):
    """When a right value stamped t is only published after t, the inclusive
    join reads it and the strict one does not. The point-in-time check flips
    only 0/1 outcomes, so this one is parity's to find."""
    df = data["asof"]
    with pytest.raises(vipyr.ParityError, match="value|age_secs"):
        vipyr.check_parity(catalog.asof_on_combined(False), df, by="user", ts="ts", outcomes=["value"],
                             requests=None)
    vipyr.check_point_in_time(catalog.asof_on_combined(False), df, by="user", ts="ts", outcomes=["value"])


def test_every_planted_bug_is_found():
    """catalog/mutants.py plants one slip at a time in the kernels; each must
    be found by the reference, the point-in-time check or the parity check."""
    import mutants
    for name, what, ref_rows, n, pit, par in mutants.run(n=2500, verbose=False):
        assert ref_rows or pit == "FAILS" or par == "FAILS", "%s: %s went unnoticed" % (name, what)

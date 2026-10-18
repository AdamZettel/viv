# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr with pandas and polars frames: binding by name, every conversion
rule, zero copy, results as frames, null masks, the point-in-time check and
the card-velocity demo.

    VIV=target/release/viv python3 -m pytest -q python/tests/test_frames.py
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=".*bottleneck.*")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

PYDIR = Path(__file__).resolve().parents[1]
REPO = PYDIR.parent
sys.path.insert(0, str(PYDIR))

import vipyr  # noqa: E402
from vipyr import frames  # noqa: E402

DEMO = REPO / "examples" / "datasci" / "card_velocity"


@pytest.fixture(scope="module")
def k(tmp_path_factory):
    d = tmp_path_factory.mktemp("frames")
    shutil.copy(Path(__file__).with_name("frames_fx.viv"), d / "fx.viv")
    return vipyr.load(d / "fx.viv")


def both(data):
    return [pd.DataFrame(data), pl.DataFrame(data)]


def arr(df, col):
    return frames.column_array(df, col)[0]


# ------------------------------------------------------------------ binding

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_columns_by_name_and_frame_out(k, lib):
    data = {"x": [1.0, 2.0, 3.0], "n": [1, 2, 3], "other": [9, 9, 9]}
    df = pd.DataFrame(data, index=[10, 20, 30]) if lib == "pandas" else pl.DataFrame(data)
    out = k.scale(df, a=2.0)
    assert frames.frame_library(out) == lib
    assert list(out.columns) == ["y", "k"]
    assert out["y"].to_list() == [3.0, 6.0, 9.0] and out["k"].to_list() == [2, 4, 6]
    if lib == "pandas":
        assert list(out.index) == [10, 20, 30]          # the input's index is kept


def test_record_param_from_frame_and_renames(k):
    for df in both({"xx": [1.0, 2.0], "n": [5, 6]}):
        out = k.scale_rec(df, columns={"x": "xx"})
        assert out["y"].to_list() == [6.0, 8.0]
        out = k.scale_rec(df, columns={"p.x": "xx"})
        assert out["y"].to_list() == [6.0, 8.0]
        out = k.scale(df, a=1.0, columns={"x": "xx"}, as_frame=False)
        assert isinstance(out, dict) and out["y"].tolist() == [6.0, 8.0]


def test_arrays_still_work_and_missing_column(k):
    assert k.total(np.arange(4.0)) == 6.0
    with pytest.raises(KeyError, match="no column 'x'"):
        k.total(pd.DataFrame({"y": [1.0]}))
    with pytest.raises(TypeError, match="columns= needs a DataFrame"):
        k.total(np.arange(2.0), columns={"x": "y"})


def test_scalar_and_other_lengths(k):
    for df in both({"x": [1.0, 2.0, 4.0]}):
        assert k.total(df) == 7.0
    for df in both({"n": [7, 8, 9]}):
        out = k.head2(df)                                # 2 rows from 3: a new frame
        assert frames.frame_library(out) == frames.frame_library(df) and len(out) == 2
        assert out["k"].to_list() == [7, 8]
    out = k.head2(pd.DataFrame({"n": [7, 8, 9]}, index=[5, 6, 7]))
    assert list(out.index) == [0, 1]


def test_single_array_result_is_a_series(k):
    df = pd.DataFrame({"v": [1.0, 2.0], "v_valid": [1, 1]}, index=["a", "b"])
    out = k.masked(df)
    assert isinstance(out, pd.Series) and list(out.index) == ["a", "b"]


def test_viv_errors_keep_their_classes(k):
    with pytest.raises(IndexError) as e:
        k.poke(pd.DataFrame({"x": [1.0, 2.0]}))
    assert isinstance(e.value, k.VivIndexError)


# -------------------------------------------------------------- conversions

def test_float_int_zero_copy_pandas():
    df = pd.DataFrame({"x": np.arange(5.0), "n": np.arange(5, dtype=np.int64)})
    assert np.shares_memory(arr(df, "x"), df["x"].to_numpy())
    assert np.shares_memory(arr(df, "n"), df["n"].to_numpy())
    assert arr(df, "x").dtype == np.float64 and arr(df, "n").dtype == np.int64


def test_float_int_zero_copy_polars():
    df = pl.DataFrame({"x": np.arange(5.0), "n": np.arange(5, dtype=np.int64)})
    for c in ("x", "n"):
        a = arr(df, c)
        assert np.shares_memory(a, df.get_column(c).to_numpy())     # the Arrow buffer
        assert not a.flags.writeable


def test_zero_copy_reaches_the_kernel(k):
    """The array handed to the C call is the frame's own memory."""
    df = pd.DataFrame({"x": np.arange(5.0), "n": np.arange(5, dtype=np.int64)})
    vals, _, reader = frames.bind_frame(k, "scale", (df,), {"a": 1.0})
    passed = np.ascontiguousarray(vals[0], dtype=np.float64)      # what _check does
    assert np.shares_memory(passed, df["x"].to_numpy())


def test_narrow_numbers_widen():
    for df in both({"a": np.array([1, 2], np.int32), "b": np.array([1.5, 2.5], np.float32)}):
        assert arr(df, "a").dtype == np.int64 and arr(df, "a").tolist() == [1, 2]
        assert arr(df, "b").dtype == np.float64


def test_bool_is_0_1():
    for df in both({"b": [True, False, True]}):
        a = arr(df, "b")
        assert a.dtype == np.int64 and a.tolist() == [1, 0, 1]


def test_datetime_any_unit_and_timezone_to_utc_ns():
    t = dt.datetime(2026, 1, 2, 3, 4, 5)
    ns = int(pd.Timestamp(t).value)
    p = pd.DataFrame({"t": pd.to_datetime([t]).as_unit("s")})
    assert arr(p, "t").tolist() == [ns]
    paris = pd.DataFrame({"t": pd.to_datetime([t]).tz_localize("Europe/Paris")})
    assert arr(paris, "t").tolist() == [ns - 3600 * 10 ** 9]        # 03:04 Paris = 02:04 UTC
    for unit in ("ms", "us", "ns"):
        q = pl.DataFrame({"t": [t]}).with_columns(pl.col("t").cast(pl.Datetime(unit)))
        assert arr(q, "t").tolist() == [ns]
    q = pl.DataFrame({"t": [t]}).with_columns(pl.col("t").dt.replace_time_zone("Europe/Paris"))
    assert arr(q, "t").tolist() == [ns - 3600 * 10 ** 9]


def test_datetime_ns_zero_copy_pandas():
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=4, freq="h")})
    assert np.shares_memory(arr(df, "t"), df["t"].to_numpy())


def test_durations_to_ns():
    p = pd.DataFrame({"d": pd.to_timedelta([1, 90], unit="s")})
    assert arr(p, "d").tolist() == [10 ** 9, 90 * 10 ** 9]
    q = pl.DataFrame({"d": [dt.timedelta(seconds=1), dt.timedelta(seconds=90)]})
    assert arr(q, "d").tolist() == [10 ** 9, 90 * 10 ** 9]


def test_categorical_and_string_codes_and_rebuild(k):
    p = pd.DataFrame({"tag": pd.Categorical(["b", "a", "b"], categories=["b", "a", "z"]),
                      "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]).tz_localize("UTC")})
    assert arr(p, "tag").tolist() == [0, 1, 0]
    out = k.echo(p)
    assert out["tag"].dtype == p["tag"].dtype and out["tag"].tolist() == ["b", "a", "b"]
    assert out["when"].tolist() == p["when"].tolist()                 # tz-aware back again
    s = pd.DataFrame({"tag": ["x", "y", "x"], "when": [1, 2, 3]})
    assert arr(s, "tag").tolist() == [0, 1, 0]                        # order of first appearance
    assert k.echo(s)["tag"].tolist() == ["x", "y", "x"]
    q = pl.DataFrame({"tag": ["y", "x", "y"], "when": [1, 2, 3]})
    assert arr(q, "tag").tolist() == [1, 0, 1]                        # sorted
    assert k.echo(q)["tag"].to_list() == ["y", "x", "y"]
    e = pl.DataFrame({"tag": pl.Series(["x", "y"], dtype=pl.Enum(["y", "x"])), "when": [1, 2]})
    assert arr(e, "tag").tolist() == [1, 0]
    assert k.echo(e)["tag"].dtype == pl.Enum(["y", "x"])
    c = pl.DataFrame({"tag": pl.Series(["y", "x"], dtype=pl.Categorical), "when": [1, 2]})
    assert arr(c, "tag").tolist() == [1, 0]
    assert k.echo(c)["tag"].to_list() == ["y", "x"]


def test_other_dtypes_are_refused_by_name():
    with pytest.raises(TypeError, match=r"column 'o'.*object"):
        arr(pd.DataFrame({"o": [object(), object()]}), "o")
    with pytest.raises(TypeError, match=r"column 'l'.*List"):
        arr(pl.DataFrame({"l": [[1], [2]]}), "l")
    with pytest.raises(TypeError, match=r"column 'c'.*complex"):
        arr(pd.DataFrame({"c": [1j, 2j]}), "c")


# ------------------------------------------------------------------- nulls

def test_nulls_refused_without_a_mask(k):
    frames_ = [pd.DataFrame({"x": [1.0, np.nan]}), pd.DataFrame({"x": pd.array([1.0, None], dtype="Float64")}),
               pl.DataFrame({"x": [1.0, None]})]
    for df in frames_:
        with pytest.raises(ValueError, match=r"column 'x' holds nulls.*x_valid"):
            k.total(df)


def test_nulls_with_a_mask(k):
    for df in [pd.DataFrame({"v": pd.array([1.5, None, 3.0], dtype="Float64")}),
               pd.DataFrame({"v": [1.5, np.nan, 3.0]}), pl.DataFrame({"v": [1.5, None, 3.0]})]:
        out = k.masked(df)
        assert list(out) == [1.5, -1.0, 3.0]
    df = pd.DataFrame({"v": [1.0, 2.0]})                    # no nulls: the mask is all ones
    assert list(k.masked(df)) == [1.0, 2.0]
    ints = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")})
    with pytest.raises(ValueError, match="column 'n' holds nulls"):
        k.head2(ints)


# ----------------------------------------------------- point-in-time check

def _counts(df, include_current):
    """Rows of the same card in the hour before (or, leaky, up to and
    including) each row, by brute force."""
    card = df["card"].to_numpy()
    t = df["ts"].to_numpy().astype(np.int64)
    out = np.zeros(len(t), np.int64)
    for i in range(len(t)):
        same = card == card[i]
        hi = t <= t[i] if include_current else t < t[i]
        out[i] = np.sum(same & hi & (t >= t[i] - 3600))
    return pd.DataFrame({"n": out}, index=df.index)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"card": rng.integers(0, 5, n), "ts": rng.integers(0, 20000, n),
                         "declined": rng.integers(0, 2, n)})


def test_pit_passes_an_honest_feature():
    rep = vipyr.check_point_in_time(lambda d: _counts(d, False), _toy(), by="card", ts="ts", outcomes=["declined"])
    assert rep["rows_checked"] > 0 and rep["features"] == ["n"]


def test_pit_catches_a_future_read():
    def next_hour(d):
        card, t = d["card"].to_numpy(), d["ts"].to_numpy()
        n = [int(np.sum((card == card[i]) & (t > t[i]) & (t <= t[i] + 3600))) for i in range(len(t))]
        return pd.DataFrame({"n": n}, index=d.index)
    with pytest.raises(vipyr.PointInTimeError, match="rows after t were changed"):
        vipyr.check_point_in_time(next_hour, _toy(), by="card", ts="ts")


def test_pit_catches_the_current_outcome():
    def declines_incl_current(d):
        card, t, y = d["card"].to_numpy(), d["ts"].to_numpy(), d["declined"].to_numpy()
        n = [int(np.sum(y[(card == card[i]) & (t <= t[i]) & (t >= t[i] - 3600)])) for i in range(len(t))]
        return pd.DataFrame({"n": n}, index=d.index)
    with pytest.raises(vipyr.PointInTimeError, match="outcomes .declined. at t were flipped"):
        vipyr.check_point_in_time(declines_incl_current, _toy(), by="card", ts="ts", outcomes=["declined"])


def test_pit_catches_row_order_dependence():
    def by_position(d):
        return pd.DataFrame({"n": np.arange(len(d))}, index=d.index)
    with pytest.raises(vipyr.PointInTimeError):
        vipyr.check_point_in_time(by_position, _toy(), by="card", ts="ts")


# ---------------------------------------------------------------- the demo

@pytest.fixture(scope="module")
def demo():
    sys.path.insert(0, str(DEMO))
    import velocity
    return velocity


def test_demo_viv_equals_pandas_reference(demo):
    df = demo.make_transactions(10 ** 6, seed=1)
    ref = demo.pandas_reference(df)
    out = demo.viv_features(df)
    assert out.index.equals(df.index)
    for c in ("n_prev_1h", "status"):
        np.testing.assert_array_equal(out[c].to_numpy(), ref[c].to_numpy())
    assert set(np.unique(ref["status"])) == {0, 1, 2}
    pol = demo.viv_features(pl.from_pandas(df.iloc[:50_000]))
    np.testing.assert_array_equal(pol["status"].to_numpy(), ref["status"].to_numpy()[:50_000])
    shuffled = df.iloc[:50_000].sample(frac=1.0, random_state=3)
    np.testing.assert_array_equal(demo.viv_features(shuffled)["n_prev_1h"].to_numpy(),
                                  ref["n_prev_1h"].reindex(shuffled.index).to_numpy())


def test_demo_point_in_time(demo):
    df = demo.make_transactions(50_000, seed=4)
    vipyr.check_point_in_time(demo.viv_features, df, by="card", ts="ts", outcomes=["declined", "chargeback"])
    with pytest.raises(vipyr.PointInTimeError, match="'status'"):
        vipyr.check_point_in_time(demo.polars_llm, pl.from_pandas(df), by="card", ts="ts",
                                    outcomes=["declined", "chargeback"])
    llm = demo.polars_llm(pl.from_pandas(df))
    ref = demo.pandas_reference(df)
    assert (llm["n_prev_1h"].to_numpy() != ref["n_prev_1h"].to_numpy()).any()


def test_demo_unsorted_input_is_refused_by_the_kernel(demo):
    df = demo.make_transactions(1000, seed=5)
    k = demo.module()
    with pytest.raises(k.VivStop, match="sorted by"):
        k.card_features(df.iloc[::-1])

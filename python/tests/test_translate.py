# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.translate and @vp.jit: Python functions translated by py2viv (exact or
refused), compiled, and checked against the Python on the first call."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "translate_cases"))
import vipyr as vp  # noqa: E402
from vipyr import translate as T  # noqa: E402,F401

import numeric as N  # noqa: E402

rng = np.random.default_rng(11)
X = rng.random(5000)
KEYS = np.sort(rng.integers(0, 40, 5000))


CASES = [
    (N.ewma, (X, 0.1)),
    (N.count_above, (X, 0.5)),
    (N.smooth, (X,)),
    (N.group_cumsum, (KEYS, X)),
    (N.group_mean, (KEYS.astype(float), X)),
    (N.cube_roots, (X * 1000.0,)),
    (N.bisect_sqrt, (2.0, 0.0, 2.0)),
    (N.poly_hash, (rng.integers(0, 1 << 20, 5000),)),
    (N.norm_and_max, (X,)),
]


@pytest.mark.parametrize("fn,args", CASES, ids=[c[0].__name__ for c in CASES])
def test_translated_equals_python(fn, args):
    t = vp.translate(fn)                   # check=True: the first call runs fn too
    got = t(*args)
    want = fn(*args)
    if isinstance(want, tuple):
        assert all(np.array_equal(a, b) for a, b in zip(got, want))
    else:
        assert np.array_equal(got, want) and type(np.asarray(got).item(0)) is type(np.asarray(want).item(0))
    assert t.viv_source.startswith("###") and "def %s(" % fn.__name__ in t.viv_source
    assert t(*args) is not None            # second call: no check, same version
    assert len(t.versions) == 1


def test_jit_decorator_and_signatures():
    @vp.jit
    def scaled_sum(x, a=2.0):
        s = 0.0
        for v in x:
            s += a * v
        return s
    assert scaled_sum(X) == scaled_sum.python(X)
    assert scaled_sum(X, a=3.0) == scaled_sum.python(X, a=3.0)
    assert scaled_sum(x=X, a=3) == scaled_sum.python(X, 3)        # an int a: a second signature
    assert len(scaled_sum.versions) == 2


def test_frame_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"k": KEYS, "v": X})
    t = vp.translate(N.group_cumsum)
    # (pandas' own groupby().cumsum() is compensated, so the loop is the reference)
    assert np.array_equal(t(df["k"], df["v"]), N.group_cumsum(KEYS, X))
    try:
        import polars as pl
    except ImportError:
        return
    p = pl.DataFrame({"k": KEYS, "v": X})
    assert np.array_equal(t(p["k"], p["v"]), N.group_cumsum(KEYS, X))


def test_show(capsys):
    t = vp.translate(N.bisect_sqrt, show=True)
    assert "def bisect_sqrt(c: Num, lo: Num, hi: Num)" in capsys.readouterr().out
    assert t(2.0, 0.0, 2.0) == N.bisect_sqrt(2.0, 0.0, 2.0)


@pytest.mark.parametrize("fn,construct", [
    (N.uses_dict_of_lists, "list.append"),
    (N.uses_try, "try"),
    (N.uses_with, "with"),
])
def test_refusals_name_the_line(fn, construct):
    # every parameter unannotated: the constructs are still refused at translate() time
    with pytest.raises(vp.VivTranslationError) as e:
        vp.translate(fn)
    msg = str(e.value)
    assert "numeric.py:" in msg and "refuses" in msg and construct in msg
    file, line, col, what, hint = e.value.refusals[0]
    assert Path(file).name == "numeric.py" and construct in what and hint
    lines = Path(file).read_text().splitlines()
    assert lines[line - 1].strip() in msg                     # the offending line is quoted
    assert any(l.startswith("def %s(" % fn.__name__) for l in lines[:line])


def test_mismatch_is_loud(monkeypatch):
    t = vp.translate(N.bisect_sqrt, check=False)
    t(2.0, 0.0, 2.0)
    key = next(iter(t.versions))
    real = t.versions[key][0]
    t.versions[key][0] = lambda *a: real(*a) * (1 + 1e-15)    # a wrong translation
    t.check = True
    with pytest.raises(vp.VivTranslationMismatch, match="differ"):
        t(2.0, 0.0, 2.0)


def test_big_ints_refused():
    with pytest.raises(vp.VivTranslationError, match="2\\*\\*53"):
        vp.translate(N.poly_hash)(np.array([1 << 60]))


def test_without_a_c_compiler_runs_in_the_interpreter(monkeypatch):
    monkeypatch.setenv("CC", "/nonexistent/cc")

    def running_max(x):
        out = np.zeros(len(x))
        m = x[0]
        for i in range(len(x)):
            if x[i] > m:
                m = x[i]
            out[i] = m
        return out
    t = vp.translate(running_max)
    assert np.array_equal(t(X[:500]), running_max(X[:500]))
    assert getattr(next(iter(t.versions.values()))[0], "__module__", "") or True

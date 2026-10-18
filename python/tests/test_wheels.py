# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Precompiled libraries (vipyr/_prebuilt.py) and the interpreter fallback
(vipyr/_interp.py): what a platform wheel promises without a C compiler."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vipyr as vp  # noqa: E402
import importlib  # noqa: E402
from vipyr import _core, _interp, _prebuilt  # noqa: E402

build = importlib.import_module("vipyr.build")   # vipyr.build the attribute is _core.build

SRC = """\
def ewma(x: Arr, a: Num) -> Arr {
    y = zeros(len(x))
    s = x[0]
    for i to len(x) {
        s = a * x[i] + (1.0 - a) * s
        y[i] = s
    }
    return y
}
def stats(x: Arr, k: [Int]) {
    n: Int = 0
    for v in x {
        if v > 0.5 {
            n = n + 1
        }
    }
    k[0] = k[0] + n
    return n, sum(x), k, reshape(x, 2, -1)
}
"""


def _same(a, b):
    a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
    return all(np.array_equal(p, q) and np.asarray(p).dtype == np.asarray(q).dtype for p, q in zip(a, b))


@pytest.fixture
def no_cc(monkeypatch):
    monkeypatch.setenv("CC", "/nonexistent/cc")


def test_interpreter_fallback_matches_compiled(tmp_path, monkeypatch):
    f = tmp_path / "kfall.viv"
    f.write_text(SRC)
    compiled = vp.load(f)
    monkeypatch.setenv("CC", "/nonexistent/cc")
    interp = vp.load(f, name="kfall_i")
    assert getattr(interp, "__vipyr_interp__", False)
    x = np.random.default_rng(3).random(1000)
    k = np.arange(4)
    assert _same(compiled.ewma(x, 0.25), interp.ewma(x, 0.25))
    assert _same(compiled.stats(x, k), interp.stats(x, k))
    assert k[0] == 0                         # the caller's array is not written
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": x})
    assert _same(np.asarray(compiled.ewma(df, a=0.25)), np.asarray(interp.ewma(df, a=0.25)))
    for mod in (compiled, interp):          # the same argument checks
        with pytest.raises(ValueError):
            mod.ewma("not an array", 0.25)


def test_no_cc_without_fallback_raises(tmp_path, no_cc, monkeypatch):
    f = tmp_path / "knofb.viv"
    f.write_text(SRC)
    monkeypatch.setenv("VIPYR_INTERP", "0")
    with pytest.raises(_core.NoCompiler):
        vp.load(f)


def test_precompiled_library_is_used_without_a_compiler(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "kpre.viv").write_text(SRC)
    files = build.prebuilt_files("kpre", {"prebuild": ["kpre.viv"], "prebuild-root": str(root)}, "pkg")
    site = tmp_path / "site"
    for src, arc, _ in files:
        p = site / arc
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(src if isinstance(src, bytes) else Path(src).read_bytes())
    assert (site / "pkg" / "_root" / "kpre.viv").is_file()
    monkeypatch.setattr(_prebuilt, "_package_dirs", lambda: [site / "pkg"])
    monkeypatch.setattr(_prebuilt, "_memo", {})
    monkeypatch.setenv("CC", "/nonexistent/cc")
    # the same bytes at another path: found by content
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copy(root / "kpre.viv", other / "kpre.viv")
    hits, compiles = _prebuilt.stats["prebuilt_hits"], _core.stats["compiles"]
    m = vp.load(other / "kpre.viv")
    assert not getattr(m, "__vipyr_interp__", False)
    assert _prebuilt.stats["prebuilt_hits"] == hits + 1 and _core.stats["compiles"] == compiles
    assert m.stats(np.array([0.2, 0.7]), np.array([1]))[0] == 1
    # an edited source does not take the library: the interpreter runs it
    (other / "kpre.viv").write_text(SRC + "\n")
    m2 = vp.load(other / "kpre.viv", name="kpre2")
    assert getattr(m2, "__vipyr_interp__", False)

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The %%viv cell magic (vipyr.magic): cells as modules, the build cache,
DataFrame arguments, run records, the rerun check, C export and errors
located in the cell. The IPython tests drive a real InteractiveShell; the
rest call the functions under the magics directly.

    VIV=target/release/viv python3 -m pytest -q python/tests/test_magic.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", message=".*bottleneck.*")

PYDIR = Path(__file__).resolve().parents[1]
REPO = PYDIR.parent
sys.path.insert(0, str(PYDIR))

import vipyr  # noqa: E402
from vipyr import _core, magic  # noqa: E402

CELL = """\
#| a + b, elementwise, and their sum
def addw(x: Arr, w: Num) -> Arr {
    y = zeros(len(x))
    for i to len(x) {
        y[i] = x[i] + w
    }
    return y
}

def total(x: Arr) -> Num {
    s = 0.0
    for i to len(x) {
        s = s + x[i]
    }
    return s
}
"""

BAD = """\
def ok(x: Num) -> Num {
    return x
}

def broken(x: Num) -> Num {
    return x + nowhere
}
"""


@pytest.fixture
def nb(tmp_path, monkeypatch):
    """A notebook directory: cells and the run log go there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(magic.state, "base", tmp_path)
    monkeypatch.setitem(magic.state, "runs", None)
    monkeypatch.setitem(magic.state, "enabled", True)
    monkeypatch.setitem(magic.state, "modules", {})
    monkeypatch.delenv("VIPYR_RUNS", raising=False)
    magic._index.clear()
    return tmp_path


def records(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


# ------------------------------------------------------------- the functions

def test_define_compiles_once_and_records(nb):
    m = magic.define("pipe", CELL)
    assert sorted(m.__all__) == ["addw", "total"]
    assert m.addw(np.arange(3.0), 1.0).tolist() == [1.0, 2.0, 3.0]
    assert m.total(np.arange(4.0)) == 6.0
    before = dict(_core.stats)
    m2 = magic.define("pipe", CELL)                        # the same cell again: from the cache
    assert _core.stats["exports"] == before["exports"] and _core.stats["cache_hits"] == before["cache_hits"] + 1
    m2.total(np.arange(4.0))
    recs = records(nb / ".viv_runs.jsonl")
    assert [r["program"]["function"] for r in recs] == ["addw", "total", "total"]
    r = recs[1]
    assert r["schema"] == magic.SCHEMA and r["program"]["module"] == "pipe"
    assert len(r["program"]["source_sha256"]) == 64 and r["engine"]["mode"] == "exact"
    assert r["engine"]["viv"] == _core.compiler_identity()
    assert r["time"].endswith("Z") and r["wall_s"] >= 0
    assert recs[1]["id"] == recs[2]["id"]                  # same run, same content address
    assert "mismatch_of" not in recs[2]


def test_hashes_are_exact():
    d = magic.describe
    assert d(1.0)["hash"] != d(1.0 + 2 ** -52)["hash"]
    assert d(1)["hash"] != d(1.0)["hash"]
    assert d(np.arange(3.0))["hash"] == d(np.arange(3.0).copy())["hash"]
    assert d(np.arange(3.0))["hash"] != d(np.arange(3))["hash"]
    assert d((1.0, 2.0))["hash"] != d((2.0, 1.0))["hash"]


def test_dataframe_inputs_hash_per_column(nb):
    pd = pytest.importorskip("pandas")
    m = magic.define("fr", CELL)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "tag": ["a", "b", "a"]})
    out = m.addw(df, w=0.5)
    assert isinstance(out, pd.Series) and out.tolist() == [1.5, 2.5, 3.5]
    df2 = df.assign(tag=["a", "b", "c"])                    # a column the function does not read
    m.addw(df2, w=0.5)
    r1, r2 = records(nb / ".viv_runs.jsonl")
    c1, c2 = r1["inputs"]["args"][0]["columns"], r2["inputs"]["args"][0]["columns"]
    assert set(c1) == {"x", "tag", "__index__"}
    assert c1["x"] == c2["x"] and c1["tag"] != c2["tag"]
    assert r1["inputs_hash"] != r2["inputs_hash"]
    assert r1["outputs_hash"] == r2["outputs_hash"]


def test_polars_columns_hash(nb):
    pl = pytest.importorskip("polars")
    m = magic.define("frp", CELL)
    df = pl.DataFrame({"x": [1.0, 2.0], "c": ["u", "v"]})
    assert m.total(df) == 3.0
    (r,) = records(nb / ".viv_runs.jsonl")
    assert r["inputs"]["args"][0]["frame"] == "polars" and set(r["inputs"]["args"][0]["columns"]) == {"x", "c"}


def test_rerun_mismatch_warns_and_names_both(nb):
    calls = []

    def leaky(x):                                          # a function with hidden state
        calls.append(1)
        return float(np.sum(x)) + len(calls)

    f = magic.recorded(leaky, "pipe", "leaky", "0" * 64, False)
    err = io.StringIO()
    old = sys.stderr
    sys.stderr = err
    try:
        f(np.ones(3))
        f(np.ones(3))
    finally:
        sys.stderr = old
    a, b = records(nb / ".viv_runs.jsonl")
    assert a["inputs_hash"] == b["inputs_hash"] and a["outputs_hash"] != b["outputs_hash"]
    assert b["mismatch_of"] == a["id"]
    msg = err.getvalue()
    assert "rerun mismatch: pipe.leaky" in msg and a["id"] in msg and b["id"] in msg


def test_rerun_check_reads_the_existing_log(nb):
    m = magic.define("pipe", CELL)
    m.total(np.arange(4.0))
    rec = records(nb / ".viv_runs.jsonl")[0]
    forged = dict(rec, outputs_hash="f" * 64, id="rr-earlier")    # a log from an earlier session
    (nb / ".viv_runs.jsonl").write_text(json.dumps(forged) + "\n")
    magic._index.clear()
    err = io.StringIO()
    assert magic.append_record(magic.make_record("pipe", "total", rec["program"]["source_sha256"], False,
                                                 (np.arange(4.0),), {}, 6.0, 0.0), stream=err)["id"] == "rr-earlier"
    assert "rr-earlier" in err.getvalue()


def test_changed_source_is_not_a_mismatch(nb):
    magic.define("pipe", CELL).total(np.arange(4.0))
    magic.define("pipe", CELL.replace("s = s + x[i]", "s = s + 2.0 * x[i]")).total(np.arange(4.0))
    a, b = records(nb / ".viv_runs.jsonl")
    assert a["program"]["source_sha256"] != b["program"]["source_sha256"] and "mismatch_of" not in b


def test_configured_path_and_off(nb, monkeypatch):
    log = nb / "logs" / "runs.jsonl"
    monkeypatch.setitem(magic.state, "runs", str(log))
    m = magic.define("pipe", CELL)
    m.total(np.arange(2.0))
    assert len(records(log)) == 1 and not (nb / ".viv_runs.jsonl").exists()
    monkeypatch.setitem(magic.state, "enabled", False)
    m.total(np.arange(2.0))
    assert len(records(log)) == 1


def test_compile_error_points_at_the_cell(nb):
    with pytest.raises(vipyr.VivCompileError) as ei:
        magic.define("bad", BAD)
    e = ei.value
    assert e.filename == "<%%viv bad>"
    assert e.lineno == 7                                   # the %%viv line is line 1
    assert "nowhere" in e.text and "nowhere" in str(e.msg)
    assert ".viv" not in str(e.msg).replace("<%%viv", "")


def test_runtime_error_points_at_the_cell(nb):
    m = magic.define("rt", "def g(x: Arr) -> Num {\n    return x[5]\n}\n")
    with pytest.raises(m.VivIndexError) as ei:
        m.g(np.ones(2))
    msg = str(ei.value)
    assert "--> <%%viv rt>:3:12" in msg and "   3 |     return x[5]" in msg and "rt.viv" not in msg
    assert not (nb / ".viv_runs.jsonl").exists()          # a failed call leaves no record


def test_bad_name(nb):
    with pytest.raises(ValueError):
        magic.define("not a name", CELL)


def test_export_c_compiles_standalone(nb):
    magic.define("pipe", CELL)
    out = magic.export_c("pipe", nb / "pipeline.c")
    text = out.read_text()
    assert "addw" in text and "total" in text
    r = subprocess.run([_core.cc_command(), "-c", "-O2", str(out), "-o", str(nb / "pipeline.o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with pytest.raises(KeyError):
        magic.export_c("nope", nb / "x.c")


# ------------------------------------------------------------------ IPython

@pytest.fixture
def ip(nb):
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "vipyr")
    yield shell
    shell.user_ns.pop("pipe", None)


def test_ipython_cell_magic(ip, nb):
    ip.run_cell_magic("viv", "pipe", CELL)
    r = ip.run_cell("import numpy as np\nv = pipe.total(np.arange(5.0))")
    assert r.success and ip.user_ns["v"] == 10.0
    assert records(nb / ".viv_runs.jsonl")[0]["program"]["function"] == "total"
    ip.run_line_magic("viv_export", "pipe -o pipeline.c")
    assert (nb / "pipeline.c").is_file()
    ip.run_line_magic("viv_runs", "off")
    ip.run_cell("pipe.total(np.arange(5.0))")
    assert len(records(nb / ".viv_runs.jsonl")) == 1
    ip.run_line_magic("viv_runs", "on")


def test_ipython_compile_error(ip, capsys):
    r = ip.run_cell("%%viv bad\n" + BAD)
    assert isinstance(r.error_in_exec.viv_error, vipyr.VivCompileError)
    out = capsys.readouterr()
    shown = out.out + out.err
    assert "<%%viv bad>, line 7: unknown variable `nowhere`" in shown
    assert "\n        return x + nowhere\n                   ^\n" in shown
    assert "magic.py" not in shown and "Traceback" not in shown


def test_example_notebook_runs(ip, nb):
    """Each code cell of examples/datasci/notebook_demo.ipynb, in order."""
    cells = json.loads((REPO / "examples" / "datasci" / "notebook_demo.ipynb").read_text())["cells"]
    for c in cells:
        if c["cell_type"] == "code":
            assert not c["outputs"]
            r = ip.run_cell("".join(c["source"]))
            assert r.success, "".join(c["source"])
    assert ip.user_ns["out"]["mean_prev"].tolist() == [0.0, 10.0, 20.0, 0.0, 5.0, 0.0]
    assert ip.user_ns["out"]["n_prev"].tolist() == [0, 1, 2, 0, 1, 0]
    assert (nb / "prior_mean.c").is_file()
    runs = records(nb / ".viv_runs.jsonl")
    assert len(runs) == 2 and runs[0]["id"] == runs[1]["id"]

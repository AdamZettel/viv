# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""runner.py: a cell's output is exactly the part of the whole program's output it printed."""

import glob
import os
import subprocess

import pytest

import vivnb
from vivkernel import figures
from vivkernel.runner import Runner, find_viv

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "..", "examples", "*.viv")))


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    r = Runner(workdir=str(tmp_path_factory.mktemp("cells")))
    yield r
    r.close()


def code_cells(path):
    with open(path) as f:
        return [c["source"] for c in vivnb.viv_to_cells(f.read()) if c["cell_type"] == "code"]


def whole_program(path):
    out = subprocess.run([find_viv(), "run", "--quiet", path], capture_output=True, text=True).stdout
    return out.strip("\n")


@pytest.mark.parametrize("engine", ["interp", "native"])
@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_cells_add_up_to_the_whole_program(runner, path, engine):
    cells = code_cells(path)
    outs = []
    for n, cell in enumerate(cells):
        res = runner.run(cells[:n], cell, engine=engine)
        assert res.ok, res.error
        assert res.failed_tests == []
        outs.append(res.output)
    joined = "\n".join(o for o in outs if o)
    assert joined == whole_program(path)


def test_a_failing_test_is_output_not_an_error(runner):
    res = runner.run(["x = 2.0"], 'y = x * 3.0\nshow y\ntest "y is five" {\n    assert_near(y, 5.0, 1.0e-12)\n}')
    assert res.ok
    assert res.output.split("\n") == ["y: 6", "FAIL y is five -- assert_near: |6 - 5| = 1 > 1e-12"]
    assert res.failed_tests == ["y is five -- assert_near: |6 - 5| = 1 > 1e-12"]
    assert "assertion" not in res.stderr  # the engine's count line is dropped


def test_an_earlier_failing_test_does_not_leak_into_this_cell(runner):
    res = runner.run(['test "no" {\n    assert_near(1.0, 2.0, 0.1)\n}'], "show 3.0")
    assert res.ok and res.output == "3"


@pytest.mark.parametrize("engine", ["interp", "native"])
def test_an_undefined_name_is_an_error_located_in_the_cell(runner, engine):
    res = runner.run(["a = 1.0", "b = 2.0\nc = 3.0"], "d = 4.0\nshow zz", engine=engine)
    assert not res.ok
    assert "zz" in res.error
    # the compiler's location, counted from the cell's own first line
    assert "--> this cell, line 2, column 6" in res.error
    assert "   2 | show zz" in res.error


def test_a_syntax_error_is_an_error(runner):
    res = runner.run([], "x = (1.0 +\n")
    assert not res.ok
    assert res.error.startswith("error")


def test_a_runtime_error_keeps_the_interpreters_message(runner):
    res = runner.run(["v = zeros(3)"], "show v[5]")
    assert not res.ok
    assert "out of bounds" in res.error


def test_native_reuses_the_compiled_program(runner):
    cells = ["x = 1.0", "show x + 1.0"]
    a = runner.run(cells[:1], cells[1], engine="native")
    n = len(runner._binaries)
    b = runner.run(cells[:1], cells[1], engine="native")
    assert a.output == b.output == "2"
    assert len(runner._binaries) == n


def test_modules_resolve_beside_the_notebook(tmp_path):
    (tmp_path / "helper.viv").write_text("def twice(v) {\n    return 2.0 * v\n}\n")
    r = Runner(workdir=str(tmp_path))
    try:
        res = r.run([], "from helper import twice\nshow twice(21.0)")
    finally:
        r.close()
    assert res.ok, res.error
    assert res.output == "42"
    # the program file written next to the notebook is gone again
    assert sorted(p.name for p in tmp_path.iterdir()) == ["helper.viv"]


def test_figure_protocol_groups_like_the_site():
    out = "\n".join([
        "figure: T ; x ; y",
        "plot: a ; [0, 1, 2] ; [0, 1, 4]",
        "plot: b ; [0, 1, 2] ; [0, 2, 3]",
        "plot: c ; [0, 0.5] ; [1, 1]",
        "some text",
        "scatter: s1 ; red o ; [1, 2] ; [3, 4]",
        "scatter: s2 ;  ; [1, 2] ; [4, 5]",
        "bars: h ; [1, 2, 3] ; [5, 6, 7]",
        "plot: fit ; [1, 2, 3] ; [5, 6, 7]",
        "plot2d: f ; 2 3 ; [1, 2, 3, 4, 5, 6]",
        "plot2d: f ; 2 3 ; [6, 5, 4, 3, 2, 1]",
        "narrate: a sentence",
        "stage: anim ; 0 1 0 1",
        "frame: 1",
    ])
    items = figures.parse(out)
    kinds = [k for k, _ in items]
    # figure() is the only grouping: everything after
    # `figure: T` joins its chart, except records that cannot (bars on another
    # x, a heatmap), which become charts of their own
    assert kinds == ["chart", "text", "chart", "chart", "chart", "narrate", "note"]
    first = items[0][1]
    assert [s["name"] for s in first["series"]] == ["a", "b", "c", "s1", "s2", "fit"]
    assert first["fig"]["title"] == "T"
    assert first["series"][3]["open"] is True
    assert [s["name"] for s in items[2][1]["series"]] == ["h"]
    assert items[3][1]["kind"] == items[4][1]["kind"] == "heatmap"
    for _, ch in [i for i in items if i[0] == "chart"]:
        svg = figures.render(ch)
        assert svg.startswith("<svg") and svg.endswith("</svg>")

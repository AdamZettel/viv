# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivnb.py: .viv -> .ipynb -> .viv gives back the same bytes, and the cells are the ones the format defines."""

import glob
import json
import os

import pytest

import vivnb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "..", "examples", "*.viv")))


def read(p):
    with open(p, newline="") as f:
        return f.read()


def corpus():
    skip = (os.sep + "node_modules" + os.sep, os.sep + "target" + os.sep)
    out = []
    for p in glob.glob(os.path.join(REPO, "**", "*.viv"), recursive=True):
        if not any(s in p for s in skip):
            out.append(p)
    return sorted(out)


def test_every_viv_file_in_the_repository_round_trips():
    files = corpus()
    assert len(files) > 100
    bad = [p for p in files if not vivnb.roundtrip_ok(read(p))]
    assert bad == []


# Layouts the repository's files may not contain: blank lines and none
# around fences, a header with no prose, two prose blocks in a row, an
# unterminated block at the end, trailing spaces on a fence, CRLF endings,
# a file with no fences, an indented ### inside code.
TRICKY = [
    "",
    "x = 1.0\nshow x\n",
    "x = 1.0\nshow x",
    "###\npost: t\n###\nshow 1.0\n",
    "###\npost: t\ndate: d\n\n\nprose\n###\n\nx = 1.0\n",
    "###\n###\n",
    "###\n\n###\n",
    "###\none\n###\n###\ntwo\n###\n\n\n###\nthree\n###\nx = 1.0\n",
    "x = 1.0\n###\nprose at the end, never closed\n",
    "x = 1.0\n###\nprose at the end, never closed",
    "###  \nkey: v\n###\t\nx = 1.0\n\n\n\n",
    "###\r\npost: t\r\n\r\nprose\r\n###\r\n\r\nx = 1.0\r\n",
    "def f(x) {\n    ###\n    inside\n###\n    return x\n}\n",
    "\n\n\nx = 1.0\n###\nA\n###\n   \n\t\n",
    "###\nnot: a header? yes, key form\n###\n",
    "###\nplain prose first\n###\ny = 2.0\n",
]


@pytest.mark.parametrize("text", TRICKY)
def test_tricky_layouts_round_trip(text):
    assert vivnb.roundtrip_ok(text), repr(text)


def test_example_cells_have_the_expected_kinds():
    text = read(os.path.join(HERE, "..", "examples", "heat-rod.viv"))
    cells = vivnb.viv_to_cells(text)
    kinds = [c["cell_type"] for c in cells]
    # header keys (raw), header prose, then prose/code pairs
    assert kinds == ["raw", "markdown"] + ["markdown", "code"] * 4
    assert cells[0]["source"].startswith("post: Heat spreading along a rod")
    assert cells[1]["metadata"]["viv"]["role"] == "header-prose"
    assert cells[2]["source"].startswith("## The problem")
    assert cells[3]["source"].startswith("pi = 3.14159")
    # the usual layout needs no metadata
    assert all(c["metadata"] == {} for c in cells[2:])


def test_a_cell_added_in_jupyter_gets_the_usual_layout():
    text = "###\nA\n###\n\nx = 1.0\n"
    nb = vivnb.to_ipynb(text)
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": "B"})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": "show x", "outputs": [], "execution_count": None})
    assert vivnb.from_ipynb(nb) == "###\nA\n###\n\nx = 1.0\n\n###\nB\n###\n\nshow x\n"


def test_two_code_cells_in_a_row_are_joined():
    nb = {"cells": [
        {"cell_type": "code", "metadata": {}, "source": "x = 1.0"},
        {"cell_type": "code", "metadata": {}, "source": "show x"},
    ]}
    assert vivnb.from_ipynb(nb) == "x = 1.0\n\nshow x\n"


def test_cli_writes_both_directions(tmp_path):
    src = os.path.join(HERE, "..", "examples", "newton-vs-bisection.viv")
    nb_path = tmp_path / "n.ipynb"
    back = tmp_path / "n.viv"
    assert vivnb.main([src, "-o", str(nb_path)]) == 0
    nb = json.loads(nb_path.read_text())
    assert nb["metadata"]["kernelspec"]["name"] == "viv"
    assert vivnb.main([str(nb_path), "-o", str(back)]) == 0
    assert back.read_bytes() == open(src, "rb").read()
    assert vivnb.main(["--check", *EXAMPLES]) == 0

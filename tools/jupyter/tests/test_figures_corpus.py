# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The Jupyter parser against the figure-record corpus.

lang/viv/tests/figures/NAME.out is a record stream and NAME.json the charts it
must parse to, in the canonical form the spec defines. The site's parser is
held to the same files (www/scripts/figuretest.mjs), so a cell groups and
reads its records the same way in both places. Every chart of the corpus is
also drawn, to check that the renderer accepts it.
"""

import json
import math
import os
import re

import pytest

from vivkernel import figures

CORPUS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "lang", "viv", "tests", "figures"))


def _num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def _nums(a):
    return [_num(v) for v in a]


def _mark(m):
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in m.items()}


def _fig(f):
    if not f:
        return None
    out = {"title": f["title"], "xlabel": f["xlabel"], "ylabel": f["ylabel"]}
    for k in ("xlim", "ylim"):
        if k in f:
            out[k] = _nums(f[k])
    for k in ("style", "xscale", "yscale", "aspect", "yright", "styles", "origin", "colormap"):
        if k in f:
            out[k] = f[k]
    for k in ("clim", "levels"):
        if k in f:
            out[k] = _nums(f[k])
    return out


def canon(charts):
    """The canonical form of the spec: what a chart is, without bookkeeping."""
    out = []
    for c in charts:
        if c["kind"] == "grid":
            out.append({"kind": "grid", "title": c["title"], "rows": c["rows"], "cols": c["cols"],
                        "share": c["share"], "panels": [canon(p) for p in c["panels"]]})
            continue
        if c["kind"] == "anim":
            out.append({"kind": "anim", "records": len(c["lines"])})
            continue
        if c["kind"] == "diagram":
            out.append({"kind": "diagram", "fig": _fig(c["fig"]),
                        "marks": [dict(m, x=_nums(m["x"]), y=_nums(m["y"])) if m["kind"] == "arrow" else _mark(m) for m in c["marks"]]})
            continue
        if c["kind"] in ("scene", "quiver", "curve3d"):
            out.append({"kind": c["kind"]})
            continue
        marks = [_mark(m) for m in figures.marks_of(c)]
        if c["kind"] == "heatmap":
            out.append({"kind": "heatmap", "fig": _fig(c["fig"]), "label": c["label"], "nx": c["nx"], "ny": c["ny"],
                        "frames": [_nums(f) for f in c["frames"]], "marks": marks})
            continue
        series = []
        for s in c["series"]:
            if s["type"] in ("fill", "rangebars"):
                series.append({"name": s["name"], "type": s["type"], "x": _nums(s["xs"]), "lo": _nums(s["lo"]), "hi": _nums(s["hi"])})
                continue
            e = {"name": s["name"], "type": s["type"], "x": _nums(s["xs"]), "y": _nums(s["ys"])}
            if s["type"] == "points":
                e["color"] = s["color"]
                e["open"] = s["open"]
            series.append(e)
        out.append({"kind": "xy", "fig": _fig(c["fig"]), "series": series, "marks": marks})
    return out


def _names():
    return sorted(n[:-4] for n in os.listdir(CORPUS) if n.endswith(".out"))


@pytest.mark.parametrize("name", _names())
def test_corpus_parses_as_the_site_does(name):
    with open(os.path.join(CORPUS, name + ".out")) as f:
        text = f.read()
    with open(os.path.join(CORPUS, name + ".json")) as f:
        want = json.load(f)
    charts = figures.charts_of(text)
    assert canon(charts) == want
    for c in charts:
        if c["kind"] in ("xy", "heatmap", "grid", "diagram"):
            svg = figures.render(c)
            assert svg.startswith("<svg") and svg.endswith("</svg>")
            assert re.match(r'<svg [^>]*width="(\d+)', svg) and int(re.match(r'<svg [^>]*width="(\d+)', svg).group(1)) >= 200, "the chart has its width"
            # no coordinate is nan or inf: a gap is left out, not drawn at nowhere
            assert not re.search(r'="[^"]*\b(nan|inf)\b', svg), "a non-finite value reached the drawing"


def test_marks_on_a_heatmap_are_drawn_in_its_coordinates():
    # heatmap_marks.out: a 2 x 3 field drawn 120 x 80 at (34, 8); with a window
    # (0..3, 0..2) the hline at y = 1 is halfway down; without one the vline at
    # column 1 is at the middle column's center and the rect covers row 0, at the top
    with open(os.path.join(CORPUS, "heatmap_marks.out")) as f:
        charts = [c for kind, c in figures.parse(f.read()) if kind == "chart"]
    win, cells = figures.render(charts[0]), figures.render(charts[1])
    assert '<line x1="34.00" y1="48.00" x2="154.00" y2="48.00"' in win
    assert '<line x1="34.00" y1="88.00" x2="154.00" y2="8.00"' in win
    assert ">top right</text>" in win
    assert '<line x1="94.00" y1="8.00" x2="94.00" y2="88.00"' in cells
    assert '<rect x="34.00" y="8.00" width="120.00" height="40.00"' in cells

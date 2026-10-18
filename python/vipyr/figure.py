# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Figures drawn by Viv: `vipyr.figure`.

A Figure collects the same records a Viv program prints when it calls
`figure`, `plot`, `scatter`, `style`, `xscale` ...,
and `viv plot` draws them as SVG. So a figure made from Python looks like one
made in Viv, and the same records give the same bytes on every machine
(lang/viv/src/figsvg.rs): a gallery page can commit its figures and compare
them byte for byte.

    import numpy as np
    from vipyr.figure import figure

    fig = figure("Pricing error against tree depth", "N", "|error|")
    fig.plot(N, err, "CRR")
    fig.style("CRR", "s1 circle")
    fig.xscale("log"); fig.yscale("log")
    open("crr.svg", "w").write(fig.svg())

A matrix is a heatmap: `fig.heatmap(Z)`, with `origin("lower")`,
`colormap("diverging")` and `clim(lo, hi)`; `grid("title", rows, cols)` puts
figures side by side.

Labels may not contain " ; " or a newline (the record separator).
"""
from __future__ import annotations

import math
import subprocess

from . import _core

# figures made since the last `take()`; the gallery runner collects them per cell
_made: list = []


def _num(v) -> str:
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    r = repr(v)
    return r[:-2] if r.endswith(".0") else r


def _arr(a) -> str:
    return "[" + ", ".join(_num(v) for v in list(a)) + "]"


def _text(s) -> str:
    s = str(s)
    if " ; " in s or "\n" in s:
        raise ValueError("vipyr.figure: a label may not contain ' ; ' or a newline: %r" % s)
    return s


class Figure:
    """One chart: a title, axis labels, an optional window, and its series."""

    def __init__(self, title, xlabel="", ylabel="", window=None):
        head = "figure: %s ; %s ; %s" % (_text(title), _text(xlabel), _text(ylabel))
        if window is not None:
            head += " ; " + " ".join(_num(v) for v in window)
        self.title = title
        self.lines = [head]
        _made.append(self)

    def plot(self, x, y, label=""):
        self.lines.append("plot: %s ; %s ; %s" % (_text(label), _arr(x), _arr(y)))
        return self

    def scatter(self, x, y, label="", color=""):
        self.lines.append("scatter: %s ; %s ; %s ; %s" % (_text(label), _text(color), _arr(x), _arr(y)))
        return self

    def bars(self, x, y, label=""):
        self.lines.append("bars: %s ; %s ; %s" % (_text(label), _arr(x), _arr(y)))
        return self

    def fill_between(self, label, x, lo, hi):
        self.lines.append("fill_between: %s ; %s ; %s ; %s" % (_text(label), _arr(x), _arr(lo), _arr(hi)))
        return self

    def hline(self, label, y):
        self.lines.append("hline: %s ; %s" % (_text(label), _num(y)))
        return self

    def vline(self, label, x):
        self.lines.append("vline: %s ; %s" % (_text(label), _num(x)))
        return self

    def heatmap(self, Z, label=""):
        """A matrix as a heatmap (`plot2d`): row 0 at the top unless `origin("lower")`;
        with a `window` the picture spans it, otherwise the axes count cells."""
        import numpy as np
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError("vipyr.figure: heatmap wants a 2-D array, got shape %s" % (Z.shape,))
        self.lines.append("plot2d: %s ; %d %d ; %s" % (_text(label), Z.shape[0], Z.shape[1], _arr(Z.ravel())))
        return self

    def origin(self, where):
        """"lower" puts a heatmap's row 0 at the bottom; "upper" (the default) at the top."""
        self.lines.append("origin: " + where)
        return self

    def colormap(self, name):
        """"viridis" (the default) or "diverging" (white at zero, limits symmetric about it)."""
        self.lines.append("colormap: " + name)
        return self

    def clim(self, lo, hi):
        """The values the colormap's two ends stand for."""
        self.lines.append("clim: %s %s" % (_num(lo), _num(hi)))
        return self

    def segment(self, label, x0, y0, x1, y1):
        self.lines.append("segment: %s ; %s %s %s %s" % (_text(label), _num(x0), _num(y0), _num(x1), _num(y1)))
        return self

    def label(self, name, text, x, y):
        """Text on the chart at (x, y), in data units."""
        self.lines.append("label: %s ; %s ; %s %s" % (_text(name), _text(text), _num(x), _num(y)))
        return self

    def xtick(self, text, x):
        self.lines.append("xtick: %s ; %s" % (_text(text), _num(x)))
        return self

    def ytick(self, text, y):
        self.lines.append("ytick: %s ; %s" % (_text(text), _num(y)))
        return self

    def style(self, label, words):
        self.lines.append("style: %s ; %s" % (_text(label), _text(words)))
        return self

    def xscale(self, kind):
        self.lines.append("xscale: " + kind)
        return self

    def aspect(self, kind):
        """"equal": one unit of x as long on the screen as one unit of y."""
        self.lines.append("aspect: " + kind)
        return self

    def yscale(self, kind):
        self.lines.append("yscale: " + kind)
        return self

    def yright(self, label, axis_label):
        self.lines.append("yright: %s ; %s" % (_text(label), _text(axis_label)))
        return self

    def records(self) -> str:
        return "\n".join(self.lines) + "\n"

    def svg(self, width=640) -> str:
        """The figure as SVG, drawn by `viv plot`."""
        r = subprocess.run([_core.compiler(), "plot", "-", "--width", str(int(width))], input=self.records(),
                           capture_output=True, text=True, env=_core._env())
        if r.returncode != 0:
            raise RuntimeError("viv plot failed: " + r.stderr.strip())
        return r.stdout


class Grid:
    """Figures side by side, drawn as one SVG: `g = grid("title", 1, 3)`, then
    `g.figure(...)` once per panel, filled row by row. `share` is "x", "y" or
    "xy": the panels of a column (x) or a row (y) get one range."""

    def __init__(self, title, rows, cols, share=""):
        self.title = title
        self.head = ["grid: %s ; %d %d" % (_text(title), int(rows), int(cols))]
        if share:
            self.head.append("share: " + share)
        self.panels = []
        _made.append(self)

    def figure(self, title, xlabel="", ylabel="", window=None) -> Figure:
        f = Figure(title, xlabel, ylabel, window)
        _made.remove(f)
        self.panels.append(f)
        return f

    def records(self) -> str:
        return "\n".join(self.head) + "\n" + "".join(f.records() for f in self.panels)

    svg = Figure.svg


def grid(title, rows, cols, share="") -> Grid:
    """Start a grid of figures (`Grid`)."""
    return Grid(title, rows, cols, share)


def figure(title, xlabel="", ylabel="", window=None) -> Figure:
    """Start a figure: `figure("title", "x label", "y label")`; `window=(xlo, xhi, ylo, yhi)` fixes the axes."""
    return Figure(title, xlabel, ylabel, window)


def take() -> list:
    """The figures made since the last call, in order (and forget them)."""
    out = list(_made)
    _made.clear()
    return out

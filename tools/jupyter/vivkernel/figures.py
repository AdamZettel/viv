# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Turn a cell's printed output into text and inline SVG figures.

A Viv program draws by printing records, one line per plotting call. The
grammar of every record, and the rules for which records share one chart, are
the site's; this module follows them, and so does the site's
parser (parseOwnOutput in www/src/components/VivCell.jsx). Both are held to one
corpus, lang/viv/tests/figures/ (tools/jupyter/tests/test_figures_corpus.py
here, www/scripts/figuretest.mjs there), so a cell draws the same charts in
both places:

  - figure() is the only grouping. From a `figure:` record to the next one,
    every plot, scatter, bars, plot2d frame and mark is part of one chart;
    with no figure open, each record is its own chart;
  - plot() and bars() on the figure's x are series of one chart; a plot or a
    scatter on another x turns the chart into lines that each carry their own x;
  - plot2d() calls with one label and shape inside one figure are the frames
    of an animation. Jupyter shows the last frame, the run's result, and says
    how many frames there were;
  - a mark (hline, vline, xtick, label, rect, segment) inside a figure is
    drawn on that figure's chart; outside one it is part of an animation.

A `nan` or `inf` in an array is kept: it is a gap in the line (or an empty
cell), never a reason to drop the series. Records this renderer does not draw
(animations, 3-D curves and scenes, quiver fields) become one line of text
saying so.
"""

import math
import re
from html import escape

# Light-theme series colors from www/src/styles/viv.css (--s1..--s6).
# the data palette (www/src/lib/dataPalette.mjs: cyan, amber, indigo, green), then a fifth and sixth
SERIES = ["#0891b2", "#d97706", "#4a3fc4", "#1f8a5b", "#be185d", "#64748b"]
NAMED = {"red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c", "purple": "#9467bd",
         "orange": "#ff7f0e", "black": "#333333"}
INK = {"bg": "#ffffff", "border": "#e0daea", "axis": "#c5bad6", "grid": "#e0daea",
       "tick": "#8d82a1", "label": "#6b5f80", "title": "#1a1226", "mark": "#8d82a1",
       "band": "#8d82a1"}
FONT = "'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, sans-serif"

VIRIDIS = [
    (68, 1, 84), (72, 26, 108), (71, 47, 125), (65, 68, 135), (57, 86, 140), (49, 104, 142),
    (42, 120, 142), (35, 136, 142), (31, 152, 139), (34, 168, 132), (53, 183, 121), (84, 197, 104),
    (122, 209, 81), (165, 219, 54), (210, 226, 27), (253, 231, 37),
]

# lang/viv/src/lower.rs ANIM_VERBS, and www/src/lib/animParse.js ANIM_VERBS
ANIM_VERBS = ("stage", "axes", "curve", "dots", "label", "hline", "vline", "xtick", "ytick", "circle", "rect",
              "segment", "poly", "draw", "fade_in", "fade_out", "morph", "move_to", "say", "wait",
              "frame", "together", "apart")
_ANIM = re.compile(r"^(" + "|".join(ANIM_VERBS) + r"):(?: (.*))?$")
_MARK = re.compile(r"^(hline|vline|xtick|ytick|label|rect|segment|box|arrow|note): (.*)$")
_STYLE = re.compile(r"^style: (.*?) ; (.*?)\s*$")
# the words of a series style, one per aspect (lower.rs STYLE_WORDS)
STYLE_WORDS = {
    "line": ("solid", "dash", "dot", "noline"),
    "color": ("s1", "s2", "s3", "s4", "s5", "s6"),
    "marker": ("circle", "ring", "square", "cross", "nomarker"),
    "width": ("thin", "thick"),
}

_FIG = re.compile(r"^figure: (.*?) ; (.*?) ; ([^;]*?)(?: ; ([-\d.eE+ ]+))?(?: ; ([A-Za-z][\w-]*))?\s*$")
_PLOT = re.compile(r"^plot: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$")
_BARS = re.compile(r"^bars: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$")
_SCATTER = re.compile(r"^scatter: (.*?) ; ([^;]*) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$")
_SCALE = re.compile(r"^(xscale|yscale): (log|linear)\s*$")
_ASPECT = re.compile(r"^(aspect): (equal|auto)\s*$")
_GRID = re.compile(r"^grid: (.*?) ; (\S+) (\S+)\s*$")
_SHARE = re.compile(r"^share: (x|y|xy)\s*$")
_INTERVAL = re.compile(r"^(fill_between|rangebars): (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$")
_YRIGHT = re.compile(r"^yright: (.*?) ; (.*?)\s*$")
_HEAT_WORD = re.compile(r"^(origin|colormap): (\w+)\s*$")
_CLIM = re.compile(r"^clim: (\S+) (\S+)\s*$")
_LEVELS = re.compile(r"^levels: \[([^\]]*)\]\s*$")
_P2 = re.compile(r"^plot2d: (.*?) ; (\d+) (\d+) ; \[([^\]]*)\]\s*$")
_QUIVER = re.compile(r"^quiver: ")
_P3 = re.compile(r"^plot3d: ")
_SCENE = re.compile(r"^scene: (.*)$")
_WIRE = re.compile(r"^(wire|verts): ")
_UNDRAWN = {
    "anim": "an animation",
    "curve3d": "a 3-D curve",
    "scene": "a 3-D scene",
    "quiver": "a vector field",
}


def num(t):
    """One number of a record: C's spellings, `nan`, `-nan`, `inf`, `-inf` included."""
    t = t.strip()
    try:
        return float(t)
    except ValueError:
        return float("nan")


def nums(s):
    """The inside of a record's `[...]`. Non-finite values are KEPT (they are gaps
    when drawn); `[]` is an empty array."""
    s = s.strip()
    return [] if not s else [num(t) for t in s.split(",")]


def finite(v):
    return v is not None and math.isfinite(v)


def parse_mark(line):
    """The static mark a record describes, or None (www/src/lib/animParse.js parseMark)."""
    m = _MARK.match(line)
    if not m:
        return None
    verb, parts = m.group(1), m.group(2).split(" ; ")

    def n(i):
        return [num(t) for t in (parts[i] if i < len(parts) else "").split()]

    try:
        if verb == "hline":
            mark = {"kind": "hline", "id": parts[0], "y": n(1)[0]}
        elif verb == "vline":
            mark = {"kind": "vline", "id": parts[0], "x": n(1)[0]}
        elif verb == "xtick":
            mark = {"kind": "xtick", "text": parts[0], "x": n(1)[0]}
        elif verb == "ytick":
            mark = {"kind": "ytick", "text": parts[0], "y": n(1)[0]}
        elif verb == "label":
            x, y = n(2)[:2]
            mark = {"kind": "label", "id": parts[0], "text": parts[1], "x": x, "y": y}
        elif verb == "rect":
            x, y, w, h = n(1)[:4]
            mark = {"kind": "rect", "id": parts[0], "x": x, "y": y, "w": w, "h": h}
        elif verb == "segment":
            x1, y1, x2, y2 = n(1)[:4]
            mark = {"kind": "segment", "id": parts[0], "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif verb == "box":
            v = n(2)
            if len(v) != 4:
                return None
            mark = {"kind": "box", "id": parts[0], "style": parts[1].strip(), "x0": v[0], "y0": v[1], "x1": v[2], "y1": v[3]}
        elif verb == "arrow":
            def arr(t):
                t = t.strip()
                return nums(t[1:-1]) if t.startswith("[") and t.endswith("]") else None
            x, y = arr(parts[2]), arr(parts[3])
            # a non-finite point is a gap in the arrow, not a reason to drop it
            if x is None or y is None or len(x) != len(y) or len(x) < 2:
                return None
            return {"kind": "arrow", "id": parts[0], "style": parts[1].strip(), "x": x, "y": y}
        else:
            v = n(3)
            if len(v) != 2:
                return None
            mark = {"kind": "note", "id": parts[0], "text": parts[1], "style": parts[2].strip(), "x": v[0], "y": v[1]}
    except (IndexError, ValueError):
        return None
    vals = [v for k, v in mark.items() if k not in ("kind", "id", "text", "style")]
    return mark if all(finite(v) for v in vals) else None


def _items(text):
    """Every item of the output in order, charts of every kind included. A chart
    is {"kind": "xy" | "heatmap" | ..., "fig": dict | None, "marks": [...], ...};
    an xy chart has "series" (each {"name", "xs", "ys", "type": "line" | "bars"
    | "points"}) and "x" (the shared x, or None once its series carry their own)."""
    items = []
    charts = []
    pending = {"fig": None}
    fig_chart = None  # the chart the latest figure() opened
    fig_active = False
    cur_fig = None  # the figure most recently opened, for its settings
    # a grid of figures: the next rows * cols figures are its panels, row by
    # row; `panel` is the open panel's list of charts, or None at the top level
    grid = None
    panel = None

    def add(ch):
        if panel is not None:
            panel.append(ch)
        else:
            items.append(("chart", ch))
            charts.append(ch)

    def last_of():
        where = panel if panel is not None else charts
        return where[-1] if where else None

    def take_fig():
        f, pending["fig"] = pending["fig"], None
        return f

    def own(ch):
        nonlocal fig_chart
        add(ch)
        if fig_active and fig_chart is None:
            fig_chart = ch
        return ch

    def push(ch):
        add(ch)
        return ch

    def same_x(ch, xs):
        x = ch.get("x")
        if x is None or len(x) != len(xs):
            return False
        return all(a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(x, xs))

    def to_own_x(ch):
        # a figure whose plots do not share an x: lines that each carry their own x
        ch["x"] = None

    def push_text(line):
        if items and items[-1][0] == "text":
            items[-1] = ("text", items[-1][1] + "\n" + line)
        else:
            items.append(("text", line))

    for line in text.split("\n"):
        m = _GRID.match(line)
        if m:
            rows, cols = num(m.group(2)), num(m.group(3))
            if finite(rows) and finite(cols) and rows >= 1 and cols >= 1 and int(rows) * int(cols) <= 64:
                grid = {"kind": "grid", "title": m.group(1).strip(), "rows": int(rows), "cols": int(cols),
                        "share": "", "panels": []}
                panel = None
                add(grid)
                fig_chart = None
                fig_active = False
                pending["fig"] = None
            continue
        m = _SHARE.match(line)
        if m:
            if grid is not None:
                grid["share"] = m.group(1)
            continue
        m = _YRIGHT.match(line)
        if m:
            # that series reads against a right y axis with this label
            if cur_fig is not None:
                r = cur_fig.setdefault("yright", {"label": "", "series": []})
                r["label"] = m.group(2)
                if m.group(1) not in r["series"]:
                    r["series"].append(m.group(1))
            continue
        m = _FIG.match(line)
        if m:
            # inside a grid with a panel left, a figure opens the next panel
            if grid is not None and len(grid["panels"]) < grid["rows"] * grid["cols"]:
                panel = []
                grid["panels"].append(panel)
            else:
                grid = None
                panel = None
            fig = {"title": m.group(1).strip(), "xlabel": m.group(2).strip(), "ylabel": m.group(3).strip()}
            if m.group(4):
                w = [num(v) for v in m.group(4).split()]
                if len(w) == 4 and all(finite(v) for v in w):
                    # an equal pair means "auto-scale that axis"
                    if w[0] != w[1]:
                        fig["xlim"] = (w[0], w[1])
                    if w[2] != w[3]:
                        fig["ylim"] = (w[2], w[3])
            if m.group(5):
                fig["style"] = m.group(5)
            pending["fig"] = fig
            cur_fig = fig
            fig_chart = None
            fig_active = True
            # a diagram is a chart of marks alone
            if fig.get("style") == "diagram":
                own({"kind": "diagram", "fig": take_fig(), "marks": []})
            continue
        m = _HEAT_WORD.match(line) or _CLIM.match(line) or _LEVELS.match(line)
        if m:
            # a heatmap's settings, for the open figure
            if cur_fig is not None:
                if line.startswith("clim:"):
                    cur_fig["clim"] = [num(m.group(1)), num(m.group(2))]
                elif line.startswith("levels:"):
                    cur_fig["levels"] = nums(m.group(1))
                else:
                    cur_fig[m.group(1)] = m.group(2)
            continue
        m = _STYLE.match(line)
        if m:
            # how the open figure draws the series of that label
            if cur_fig is not None:
                st = {}
                for w in m.group(2).split():
                    for aspect, words in STYLE_WORDS.items():
                        if w in words:
                            st[aspect] = w
                cur_fig.setdefault("styles", {})[m.group(1)] = st
            continue
        m = _SCALE.match(line) or _ASPECT.match(line)
        if m:
            # a setting of the open figure, wherever in it the call is
            if cur_fig is not None:
                cur_fig[m.group(1)] = m.group(2)
            continue
        mark = parse_mark(line)
        if mark:
            last = last_of()
            if fig_chart is not None and last is fig_chart:
                fig_chart.setdefault("marks", []).append(mark)
                continue
            if pending["fig"] is not None:
                pending["fig"].setdefault("marks", []).append(mark)
                continue
        if _ANIM.match(line):
            last = last_of()
            if line.startswith("stage:") or last is None or last["kind"] != "anim":
                push({"kind": "anim", "lines": [line]})
            else:
                last["lines"].append(line)
            continue
        sn = _SCENE.match(line)
        if sn:
            push({"kind": "scene", "title": sn.group(1).strip(), "fig": take_fig()})
            continue
        if _WIRE.match(line):
            last = last_of()
            if last is None or last["kind"] != "scene":
                push({"kind": "scene", "title": "", "fig": take_fig()})
            continue

        m = _BARS.match(line)
        if m:
            xs, ys = nums(m.group(2)), nums(m.group(3))
            if xs and len(xs) == len(ys):
                s = {"name": m.group(1).strip(), "xs": xs, "ys": ys, "type": "bars"}
                if fig_chart is not None and same_x(fig_chart, xs):
                    fig_chart["series"].append(s)
                else:
                    own({"kind": "xy", "x": xs, "series": [s], "fig": take_fig()})
            continue

        m = _PLOT.match(line)
        if m:
            xs, ys = nums(m.group(2)), nums(m.group(3))
            if xs and len(xs) == len(ys):
                s = {"name": m.group(1).strip(), "xs": xs, "ys": ys, "type": "line"}
                if fig_chart is not None and same_x(fig_chart, xs):
                    fig_chart["series"].append(s)
                elif fig_chart is not None and fig_chart["kind"] == "xy":
                    to_own_x(fig_chart)
                    fig_chart["series"].append(s)
                else:
                    own({"kind": "xy", "x": xs, "series": [s], "fig": take_fig()})
            continue

        m = _INTERVAL.match(line)
        if m:
            # fill_between / rangebars: an interval from lo to hi at each x, grouped like a plot
            xs, lo, hi = nums(m.group(3)), nums(m.group(4)), nums(m.group(5))
            if xs and len(xs) == len(lo) == len(hi):
                s = {"name": m.group(2).strip(), "xs": xs, "ys": [math.nan] * len(xs), "lo": lo, "hi": hi,
                     "type": "fill" if m.group(1) == "fill_between" else "rangebars"}
                if fig_chart is not None and same_x(fig_chart, xs):
                    fig_chart["series"].append(s)
                elif fig_chart is not None and fig_chart["kind"] == "xy":
                    to_own_x(fig_chart)
                    fig_chart["series"].append(s)
                else:
                    own({"kind": "xy", "x": xs, "series": [s], "fig": take_fig()})
            continue

        m = _SCATTER.match(line)
        if m:
            tok = m.group(2).strip().split()
            color = tok[0] if tok else ""
            xs, ys = nums(m.group(3)), nums(m.group(4))
            if xs and len(xs) == len(ys):
                s = {"name": m.group(1).strip(), "xs": xs, "ys": ys, "type": "points",
                     "color": color, "open": len(tok) > 1 and tok[1] == "o"}
                if fig_chart is not None and fig_chart["kind"] == "xy":
                    to_own_x(fig_chart)
                    fig_chart["series"].append(s)
                else:
                    own({"kind": "xy", "x": None, "series": [s], "fig": take_fig()})
            continue

        m = _P2.match(line)
        if m:
            nx, ny = int(m.group(2)), int(m.group(3))
            z = nums(m.group(4))
            label = m.group(1).strip()
            if nx > 0 and ny > 0 and len(z) == nx * ny:
                # inside one figure(), plot2d() calls with one label and shape are frames
                if (fig_chart is not None and fig_chart["kind"] == "heatmap" and fig_chart["label"] == label
                        and (fig_chart["nx"], fig_chart["ny"]) == (nx, ny)):
                    fig_chart["frames"].append(z)
                else:
                    own({"kind": "heatmap", "label": label, "nx": nx, "ny": ny, "frames": [z], "fig": take_fig()})
            continue
        if _QUIVER.match(line):
            push({"kind": "quiver", "fig": take_fig()})
            continue
        if _P3.match(line):
            push({"kind": "curve3d", "fig": take_fig()})
            continue

        nar = re.match(r"^narrate: (.*)$", line)
        if nar:
            items.append(("narrate", nar.group(1)))
            continue
        push_text(line)

    return items


def parse(text):
    """Split output into an ordered list of items:
    ("text", str) | ("narrate", str) | ("chart", dict) | ("note", str).
    A chart this renderer does not draw becomes a note."""
    out = []
    for kind, v in _items(text):
        if kind == "text":
            v = v.strip("\n")
            if not v:
                continue
        if kind == "chart" and v["kind"] in _UNDRAWN:
            note = f"{_UNDRAWN[v['kind']]} (drawn on the site, not in Jupyter)"
            if out and out[-1] == ("note", note):
                continue
            out.append(("note", note))
            continue
        out.append((kind, v))
    return out


def charts_of(text):
    """Every chart the output makes, in order, drawn here or not: what the
    corpus test compares with the site's parser."""
    return [v for k, v in _items(text) if k == "chart"]


# -- drawing ----------------------------------------------------------------

def nice_ticks(a, b, n):
    span = (b - a) or 1.0
    step0 = span / max(1, n)
    mag = 10 ** math.floor(math.log10(step0))
    norm = step0 / mag
    step = (10 if norm >= 5 else 5 if norm >= 2 else 2 if norm >= 1 else 1) * mag
    t = math.ceil(a / step - 1e-9) * step
    ticks = []
    while t <= b + step * 1e-9:
        ticks.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return ticks, step


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def scale_of(kind):
    """Data value to t-space: the value on a linear axis, log10 of it on a log
    axis, where a value <= 0 is nan (a gap). www/src/lib/axisScale.js scaleOf."""
    if kind == "log":
        return lambda v: math.log10(v) if v > 0 else float("nan")
    return lambda v: v


def log_ticks(lo, hi, n):
    """Ticks of a log axis over t-space [lo, hi]: [(t, label)], [minor t].
    www/src/lib/axisScale.js logTicks, line for line."""
    k0 = math.ceil(lo - 1e-9)
    k1 = math.floor(hi + 1e-9)
    if k1 - k0 + 1 >= 2:
        m = max(1, math.ceil((k1 - k0 + 1) / max(1, n)))
        ticks = [(k, "10" + str(k).translate(SUP)) for k in range(math.ceil(k0 / m) * m, k1 + 1, m)]
        minor = []
        if m == 1:
            for k in range(math.floor(lo), k1 + 1):
                for d in range(2, 10):
                    t = k + math.log10(d)
                    if lo <= t <= hi:
                        minor.append(t)
        return ticks, minor
    def within(digits):
        out = []
        for k in range(math.floor(lo), math.ceil(hi) + 1):
            for d in digits:
                t = k + math.log10(d)
                if lo - 1e-9 <= t <= hi + 1e-9:
                    out.append((t, _num_label(d, k)))
        return out

    ticks = within([1, 2, 5])
    if len(ticks) < 3:
        ticks = within(range(1, 10))
    return ticks, []


def _num_label(d, k):
    if k >= 5 or k < -4:
        return f"{d}e{k}"
    return str(d * 10 ** k) if k >= 0 else f"{d / 10 ** -k:.{-k}f}"


def axis_ticks(kind, lo, hi, n):
    if kind == "log":
        return log_ticks(lo, hi, n)
    ticks, step = nice_ticks(lo, hi, n)
    fmt = tick_format(step)
    return [(t, fmt(t)) for t in ticks], []


def mark_to_t(k, tx, ty):
    """A mark with its coordinates in t-space (axisScale.js markToT)."""
    k = dict(k)
    if k["kind"] == "hline":
        k["y"] = ty(k["y"])
    elif k["kind"] in ("vline", "xtick"):
        k["x"] = tx(k["x"])
    elif k["kind"] == "ytick":
        k["y"] = ty(k["y"])
    elif k["kind"] == "label":
        k["x"], k["y"] = tx(k["x"]), ty(k["y"])
    elif k["kind"] == "segment":
        k["x1"], k["y1"], k["x2"], k["y2"] = tx(k["x1"]), ty(k["y1"]), tx(k["x2"]), ty(k["y2"])
    elif k["kind"] == "rect":
        x0, x1 = tx(k["x"] - k["w"] / 2), tx(k["x"] + k["w"] / 2)
        y0, y1 = ty(k["y"] - k["h"] / 2), ty(k["y"] + k["h"] / 2)
        k.update(x=(x0 + x1) / 2, y=(y0 + y1) / 2, w=abs(x1 - x0), h=abs(y1 - y0))
    return k


def tick_format(step):
    decimals = max(0, min(8, -math.floor(math.log10(step) + 1e-9)))

    def fmt(v):
        if v == 0:
            return "0"
        if abs(v) >= 1e5 or abs(v) < 1e-4:
            return f"{v:.1e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-").replace(".0e", "e")
        return f"{v:.{decimals}f}".replace("-", "−")
    return fmt


def palette_slots(series, styles=None):
    """Each series' palette slot, as www/src/lib/seriesStyle.jsx assigns them:
    in order of first appearance, a label naming one thing (series that share
    a label share a slot), an unlabeled series a slot of its own; slots a
    style claims are skipped, and a styled series gets -1 (its style sets it)."""
    styles = styles or {}
    claimed = {int(st["color"][1]) - 1 for st in styles.values() if "color" in st}
    slots, by_name = [], {}
    nxt = 0
    for s in series:
        name = s.get("name")
        if name and name in by_name:
            slots.append(by_name[name])
            continue
        if "color" in styles.get(name, {}):
            c = -1
        else:
            tries = 0
            while nxt % len(SERIES) in claimed and tries < len(SERIES):
                nxt += 1
                tries += 1
            c = nxt
            nxt += 1
        if name:
            by_name[name] = c
        slots.append(c)
    return slots


def _series_color(slot, s):
    if s.get("color") in NAMED:
        return NAMED[s["color"]]
    return SERIES[slot % len(SERIES)]


def marker_svg(kind, x, y, color, r=3.0):
    """One marker: dot (small filled), circle, ring (open), square, cross."""
    if kind == "dot":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="{color}" fill-opacity="0.85"/>'
    if kind == "ring":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="1.3"/>'
    if kind == "square":
        a = r * 0.9
        return f'<rect x="{x - a:.2f}" y="{y - a:.2f}" width="{2 * a:.2f}" height="{2 * a:.2f}" fill="{color}"/>'
    if kind == "cross":
        a = r * 0.95
        return (f'<path d="M{x - a:.2f} {y - a:.2f}L{x + a:.2f} {y + a:.2f}M{x - a:.2f} {y + a:.2f}L{x + a:.2f} {y - a:.2f}" '
                f'stroke="{color}" stroke-width="1.5" stroke-linecap="round" fill="none"/>')
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.1f}" fill="{color}"/>'


def _frame(width, height, body, title, legend):
    """The card around a chart: a title and a legend above the plot, as on the site."""
    top = 0
    head = []
    if title:
        head.append(f'<text x="14" y="22" font-size="14" font-weight="600" fill="{INK["title"]}">{escape(title)}</text>')
        top = 30
    if legend:
        x = 14
        y = top + 18
        for name, color, mark_kind in legend:
            w = 16 + len(name) * 6.6  # swatch plus an estimate of the text's width
            if x + w > width - 10 and x > 14:
                x = 14
                y += 18
            if mark_kind == "fill":
                mark = f'<rect x="{x}" y="{y - 9}" width="11" height="10" fill="{color}" fill-opacity="0.2"/>'
            elif mark_kind == "rangebars":
                mark = f'<path d="M{x + 2} {y - 10}H{x + 9}M{x + 5.5} {y - 10}V{y}M{x + 2} {y}H{x + 9}" stroke="{color}" stroke-width="1.3" fill="none"/>'
            elif mark_kind == "bars":
                mark = f'<rect x="{x}" y="{y - 9}" width="11" height="10" rx="1.5" fill="{color}" fill-opacity="0.88"/>'
            else:
                # (has a line, its dash, its width, its marker): the series as drawn
                has_line, dash, lw, marker = mark_kind
                mark = ""
                if has_line:
                    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                    mark += f'<line x1="{x}" x2="{x + 14}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="{max(2, lw)}"{dash_attr}/>'
                if marker:
                    mark += marker_svg("circle" if marker == "dot" else marker, x + 7, y - 4, color)
            head.append(mark + f'<text x="{x + 17}" y="{y}" font-size="12" fill="{INK["label"]}">{escape(name)}</text>')
            x += w + 14
        top = y + 8
    total = top + height
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total}" viewBox="0 0 {width} {total}" '
        f'font-family="{escape(FONT)}" role="img" aria-label="{escape(title or "figure")}">'
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{total - 1}" rx="8" fill="{INK["bg"]}" stroke="{INK["border"]}"/>'
        + "".join(head)
        + f'<g transform="translate(0 {top})">{body}</g></svg>'
    )


def equal_aspect(xlo, xhi, ylo, yhi, pw, min_h, max_h):
    """aspect("equal") (www/src/lib/axisScale.js equalAspect): the plot keeps its
    width and takes the height that makes a unit of y as long as a unit of x; held
    to [min_h, max_h], with the other axis's range widened about its center."""
    xs = (xhi - xlo) or 1
    ys = (yhi - ylo) or 1
    ph = pw * ys / xs
    if ph > max_h:
        ph = max_h
        c, half = (xlo + xhi) / 2, ys * pw / max_h / 2
        xlo, xhi = c - half, c + half
    elif ph < min_h:
        ph = min_h
        c, half = (ylo + yhi) / 2, xs * min_h / pw / 2
        ylo, yhi = c - half, c + half
    return xlo, xhi, ylo, yhi, ph


def marks_of(chart):
    """The program's marks on a chart, in the order printed: those before its first
    plot (kept on the figure) and those after."""
    return list((chart.get("fig") or {}).get("marks", [])) + list(chart.get("marks", []))


def _mark_extent(marks):
    mx, my = [], []
    for k in marks:
        if k["kind"] == "hline":
            my.append(k["y"])
        elif k["kind"] in ("vline", "xtick"):
            mx.append(k["x"])
        elif k["kind"] == "ytick":
            my.append(k["y"])
        elif k["kind"] == "label":
            mx.append(k["x"])
            my.append(k["y"])
        elif k["kind"] == "rect":
            mx += [k["x"] - k["w"] / 2, k["x"] + k["w"] / 2]
            my += [k["y"] - k["h"] / 2, k["y"] + k["h"] / 2]
        elif k["kind"] == "segment":
            mx += [k["x1"], k["x2"]]
            my += [k["y1"], k["y2"]]
    return mx, my


def render_xy(chart, width=640):
    """Line plots, bar charts and scatter plots: one frame of axes, many series,
    and the program's marks."""
    fig = chart.get("fig") or {}
    # everything below is in t-space: log10 of the value on a log axis
    xscale, yscale = fig.get("xscale", "linear"), fig.get("yscale", "linear")
    # yright("label", "axis"): those series read against a linear y axis on the right
    right_names = set((fig.get("yright") or {}).get("series", []))
    tx, ty = scale_of(xscale), scale_of(yscale)
    hidden = []
    if xscale == "log":
        n = sum(1 for s in chart["series"] for v in s["xs"] if finite(v) and v <= 0)
        if n:
            hidden.append(f"{n} x value{'s' if n > 1 else ''} ≤ 0")
    if yscale == "log":
        n = sum(1 for s in chart["series"] if s.get("name") not in right_names
                for v in (s["lo"] + s["hi"] if "lo" in s else s["ys"]) if finite(v) and v <= 0)
        if n:
            hidden.append(f"{n} y value{'s' if n > 1 else ''} ≤ 0")
    def tys(s, key):
        return list(s[key]) if s.get("name") in right_names else [ty(v) for v in s[key]]

    series = [dict(s, right=s.get("name") in right_names, xs=[tx(v) for v in s["xs"]], ys=tys(s, "ys"),
                   **({"lo": tys(s, "lo"), "hi": tys(s, "hi")} if "lo" in s else {}))
              for s in chart["series"]]
    has_right = any(s["right"] for s in series)
    inner = fig.get("inner") or {}
    marks = [mark_to_t(k, tx, ty) for k in marks_of(chart)]
    marks = [k for k in marks if all(finite(v) for v in k.values() if isinstance(v, (int, float)))]
    fig = dict(fig)
    for key, f in (("xlim", tx), ("ylim", ty)):
        if key in fig:
            a, b = f(fig[key][0]), f(fig[key][1])
            if finite(a) and finite(b):
                fig[key] = (a, b)
            else:
                del fig[key]
    mx, my = _mark_extent(marks)
    has_bars = any(s["type"] == "bars" for s in series)
    has_points = any(s["type"] == "points" for s in series)
    # the extent of what is drawn: a nan or inf is a gap, not a coordinate
    allx = [x for s in series for x in s["xs"] if finite(x)] + mx
    def yvals(s):
        return s["lo"] + s["hi"] if "lo" in s else s["ys"]

    ally = [y for s in series if not s["right"] for y in yvals(s) if finite(y)] + my
    ally2 = [y for s in series if s["right"] for y in yvals(s) if finite(y)]
    y2lo, y2hi = (min(ally2), max(ally2)) if ally2 else (0.0, 1.0)
    pad2 = ((y2hi - y2lo) or abs(y2hi) or 1) * 0.07
    y2lo, y2hi = y2lo - pad2, y2hi + pad2
    if not allx:
        allx = [0.0, 1.0]
    if "xlim" in fig:
        xlo, xhi = fig["xlim"]
    else:
        xlo, xhi = min(allx), max(allx)
        if has_points:
            pad = ((xhi - xlo) or abs(xhi) or 1) * 0.06
            xlo, xhi = xlo - pad, xhi + pad
    if has_bars and "xlim" not in fig:
        xs0 = sorted(set(x for x in series[0]["xs"] if finite(x)))
        gaps = [b - a for a, b in zip(xs0, xs0[1:])]
        if gaps:
            xlo, xhi = xlo - min(gaps) / 2, xhi + min(gaps) / 2
    if "ylim" in fig:
        ylo, yhi = fig["ylim"]
    else:
        ylo, yhi = (min(ally), max(ally)) if ally else (0.0, 1.0)
        on_zero = has_bars and yscale == "linear"
        if on_zero and ylo > 0:
            ylo = 0.0  # bars stand on zero
        pad = ((yhi - ylo) or abs(yhi) or 1) * 0.07
        if not (on_zero and ylo == 0):
            ylo -= pad
        yhi += pad

    equal = fig.get("aspect") == "equal"
    if equal:
        # the ranges are widened here; the height is set once the margins are known
        pw0 = width - 80
        xlo, xhi, ylo, yhi, _ = equal_aspect(xlo, xhi, ylo, yhi, pw0, 160, min(pw0, 520))
    ytick_marks = [k for k in marks if k["kind"] == "ytick"]
    if ytick_marks:
        yt, yminor = [(k["y"], escape(k["text"])) for k in ytick_marks], []
    else:
        yt, yminor = axis_ticks(yscale, ylo, yhi, 5)
    xtick_marks = [k for k in marks if k["kind"] == "xtick"]
    if xtick_marks:
        # the program's own x ticks replace the automatic ones
        xt, xminor = [(k["x"], escape(k["text"])) for k in xtick_marks], []
    else:
        xt, xminor = axis_ticks(xscale, xlo, xhi, 7)
    widest = max([len(label) for _, label in yt] + [1])
    y2t = axis_ticks("linear", y2lo, y2hi, 5)[0] if has_right else []
    widest2 = max([len(label) for _, label in y2t] + [1])
    # a panel of a grid keeps room for 8-character tick labels, so a column's plots line up
    room = 16 + 8 * 6.4 + 18
    ml = 12 if inner.get("y") else max(16 + widest * 6.4 + (18 if fig.get("ylabel") else 0), room if fig.get("inGrid") else 0)
    mr = max(16 + widest2 * 6.4 + 18 if has_right else 14, room if fig.get("inGrid") and fig.get("roomRight") else 0)
    mt, mb = 10, (10 if inner.get("x") else 44 if fig.get("xlabel") else 26)
    pw = width - ml - mr
    square = not equal and "xlim" in fig and "ylim" in fig and abs((xhi - xlo) - (yhi - ylo)) < 1e-9 * abs(xhi - xlo or 1)
    ph = pw if square else round(max(200, min(340, pw * 0.52)))
    if square and ph > 420:
        ph = pw = 420
    if equal:
        ph = round(pw * (yhi - ylo) / ((xhi - xlo) or 1))
    h = ph + mt + mb

    def X(v):
        return ml + (v - xlo) / ((xhi - xlo) or 1) * pw

    def Y(v):
        return mt + (1 - (v - ylo) / ((yhi - ylo) or 1)) * ph

    def Y2(v):
        return mt + (1 - (v - y2lo) / ((y2hi - y2lo) or 1)) * ph

    g = []
    for t, label in yt:
        g.append(f'<line x1="{ml:.1f}" x2="{ml + pw:.1f}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{INK["grid"]}"/>')
        if not inner.get("y"):
            g.append(f'<text x="{ml - 8:.1f}" y="{Y(t) + 3.8:.1f}" text-anchor="end" font-size="11" fill="{INK["tick"]}">{label}</text>')
    for t in yminor:
        g.append(f'<line x1="{ml - 3:.1f}" x2="{ml:.1f}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{INK["axis"]}"/>')
    if yscale == "linear" and ylo < 0 < yhi:
        g.append(f'<line x1="{ml:.1f}" x2="{ml + pw:.1f}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{INK["tick"]}" opacity="0.7"/>')
    g.append(f'<line x1="{ml:.1f}" x2="{ml + pw:.1f}" y1="{mt + ph}" y2="{mt + ph}" stroke="{INK["axis"]}"/>')
    for t, label in xt:
        g.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}" stroke="{INK["axis"]}"/>')
        if not inner.get("x"):
            g.append(f'<text x="{X(t):.1f}" y="{mt + ph + 17}" text-anchor="middle" font-size="11" fill="{INK["tick"]}">{label}</text>')
    for t in xminor:
        g.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 2.5}" stroke="{INK["axis"]}"/>')
    if has_right:
        xr = ml + pw
        g.append(f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{mt}" y2="{mt + ph}" stroke="{INK["axis"]}"/>')
        for t, label in y2t:
            g.append(f'<line x1="{xr:.1f}" x2="{xr + 4:.1f}" y1="{Y2(t):.1f}" y2="{Y2(t):.1f}" stroke="{INK["axis"]}"/>')
            g.append(f'<text x="{xr + 8:.1f}" y="{Y2(t) + 3.8:.1f}" font-size="11" fill="{INK["tick"]}">{label}</text>')
        label2 = (fig.get("yright") or {}).get("label")
        if label2:
            cy = mt + ph / 2
            g.append(f'<text x="{width - 10}" y="{cy:.1f}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{INK["label"]}" transform="rotate(90 {width - 10} {cy:.1f})">{escape(label2)}</text>')
    if fig.get("xlabel") and not inner.get("x"):
        g.append(f'<text x="{ml + pw / 2:.1f}" y="{h - 6}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{INK["label"]}">{escape(fig["xlabel"])}</text>')
    if fig.get("ylabel") and not inner.get("y"):
        cy = mt + ph / 2
        g.append(f'<text x="12" y="{cy:.1f}" text-anchor="middle" font-size="11.5" font-weight="500" fill="{INK["label"]}" transform="rotate(-90 12 {cy:.1f})">{escape(fig["ylabel"])}</text>')

    clip = f"c{abs(hash((len(allx), xlo, xhi, ylo, yhi))) % 10**8}"
    g.append(f'<defs><clipPath id="{clip}"><rect x="{ml:.1f}" y="{mt - 1}" width="{pw:.1f}" height="{ph + 2}"/></clipPath></defs>')
    body = [f'<g clip-path="url(#{clip})">']
    for k in marks:
        if k["kind"] == "rect":
            x0, x1 = X(k["x"] - k["w"] / 2), X(k["x"] + k["w"] / 2)
            y0, y1 = Y(k["y"] + k["h"] / 2), Y(k["y"] - k["h"] / 2)
            body.append(f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" fill="{INK["band"]}" fill-opacity="0.14"/>')
        elif k["kind"] == "xtick":
            body.append(f'<line x1="{X(k["x"]):.2f}" x2="{X(k["x"]):.2f}" y1="{mt}" y2="{mt + ph}" stroke="{INK["grid"]}"/>')
    legend = []
    seen = set()
    styles = fig.get("styles", {})
    slots = palette_slots(series, styles)
    for k, s in enumerate(series):
        st = styles.get(s.get("name"), {})
        color = SERIES[int(st["color"][1]) - 1] if "color" in st else _series_color(slots[k], s)
        line = st.get("line", "solid" if s["type"] == "line" else "noline")
        dash = {"dash": "6 4", "dot": "1.5 3.5"}.get(line)
        lw = {"thin": 1.1, "thick": 2.8}.get(st.get("width"), 1.8)
        if "marker" in st:
            marker = None if st["marker"] == "nomarker" else st["marker"]
        elif s["type"] == "points":
            marker = "ring" if s.get("open") else "dot"
        else:
            marker = "circle" if len(s["xs"]) <= 24 else None
        # series that share a label share one color and one legend entry, as on the
        # site, drawn like the label's first series that is not an interval
        interval = s["type"] in ("fill", "rangebars")
        if s.get("name") and s["name"] not in seen and (not interval or all(
                t.get("name") != s["name"] or t["type"] in ("fill", "rangebars") for t in series)):
            seen.add(s["name"])
            name = s["name"] + (" (right axis)" if s["right"] else "")
            legend.append((name, color, s["type"] if s["type"] in ("bars", "fill", "rangebars") else (line != "noline", dash, lw, marker)))
        if interval:
            Ys = Y2 if s["right"] else Y

            def ok(i):
                return finite(s["xs"][i]) and finite(s["lo"][i]) and finite(s["hi"][i])

            if s["type"] == "fill":
                # the band from lo to hi; a gap splits it
                runs, run = [], []
                for i in range(len(s["xs"])):
                    if ok(i):
                        run.append(i)
                    elif run:
                        runs.append(run)
                        run = []
                if run:
                    runs.append(run)
                d = "".join("M" + "L".join(f'{X(s["xs"][i]):.2f} {max(-1e4, min(1e4, Ys(s["hi"][i]))):.2f}' for i in r)
                            + "L" + "L".join(f'{X(s["xs"][i]):.2f} {max(-1e4, min(1e4, Ys(s["lo"][i]))):.2f}' for i in reversed(r)) + "Z"
                            for r in runs)
                body.insert(1, f'<path d="{d}" fill="{color}" fill-opacity="0.2" stroke="none"/>')
            else:
                # a capped bar from lo to hi at each x
                d = []
                for i in range(len(s["xs"])):
                    if ok(i):
                        x = X(s["xs"][i])
                        a, b = (max(-1e4, min(1e4, Ys(v))) for v in (s["lo"][i], s["hi"][i]))
                        d.append(f"M{x:.2f} {a:.2f}V{b:.2f}M{x - 3.5:.2f} {a:.2f}H{x + 3.5:.2f}M{x - 3.5:.2f} {b:.2f}H{x + 3.5:.2f}")
                body.append(f'<path d="{"".join(d)}" stroke="{color}" stroke-width="1.3" fill="none"/>')
        elif s["type"] == "bars":
            xs = sorted(set(x for x in s["xs"] if finite(x)))
            gap = min([b - a for a, b in zip(xs, xs[1:])] or [0])
            bw = max(2, gap / ((xhi - xlo) or 1) * pw * 0.86) if gap else 12
            # a bar stands on zero; on a log axis, on the bottom of the chart
            y0 = mt + ph if yscale == "log" else Y(max(ylo, min(yhi, 0)))
            for x, y in zip(s["xs"], s["ys"]):
                if not (finite(x) and finite(y)):
                    continue
                top = min(y0, Y(y))
                body.append(f'<rect x="{X(x) - bw / 2:.2f}" y="{top:.2f}" width="{bw:.2f}" height="{abs(y0 - Y(y)):.2f}" rx="1.5" fill="{color}" fill-opacity="0.88"/>')
        else:
            Ys = Y2 if s["right"] else Y
            if line != "noline":
                # a nan or inf lifts the pen: the line has a gap there
                d = []
                pen = False
                for x, y in zip(s["xs"], s["ys"]):
                    if not (finite(x) and finite(y)):
                        pen = False
                        continue
                    yy = max(-1e4, min(1e4, Ys(y)))
                    d.append(f'{"L" if pen else "M"}{X(x):.2f} {yy:.2f}')
                    pen = True
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                body.append(f'<path d="{"".join(d)}" fill="none" stroke="{color}" stroke-width="{lw}"{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>')
            if marker:
                for x, y in zip(s["xs"], s["ys"]):
                    if finite(x) and finite(y):
                        body.append(marker_svg(marker, X(x), Ys(y), color))
    for k in marks:
        if k["kind"] == "hline":
            body.append(f'<line x1="{ml:.1f}" x2="{ml + pw:.1f}" y1="{Y(k["y"]):.2f}" y2="{Y(k["y"]):.2f}" stroke="{INK["mark"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
        elif k["kind"] == "vline":
            body.append(f'<line x1="{X(k["x"]):.2f}" x2="{X(k["x"]):.2f}" y1="{mt}" y2="{mt + ph}" stroke="{INK["mark"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
        elif k["kind"] == "segment":
            body.append(f'<line x1="{X(k["x1"]):.2f}" y1="{Y(k["y1"]):.2f}" x2="{X(k["x2"]):.2f}" y2="{Y(k["y2"]):.2f}" stroke="{INK["mark"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    body.append("</g>")
    for k in marks:
        if k["kind"] == "label":
            body.append(f'<text x="{X(k["x"]):.2f}" y="{Y(k["y"]):.2f}" text-anchor="middle" dominant-baseline="middle" font-size="11.5" fill="{INK["label"]}">{escape(k["text"])}</text>')
    if hidden:
        # what a log axis cannot show is said under the chart, as on the site
        body.append(f'<text x="{ml:.1f}" y="{h + 8}" font-size="11" fill="{INK["tick"]}">not drawn on the log axis: {", ".join(hidden)}</text>')
        h += 16
    return _frame(width, h, "".join(g) + "".join(body), fig.get("title"), legend)


def viridis(t):
    x = min(1.0, max(0.0, t)) * (len(VIRIDIS) - 1)
    i = min(len(VIRIDIS) - 2, int(x))
    f = x - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return "#%02x%02x%02x" % tuple(round(a[c] + f * (b[c] - a[c])) for c in range(3))


# colormap("diverging"): ColorBrewer RdBu, reversed (www/src/components/VivCell.jsx DIVERGING)
DIVERGING = [
    (5, 48, 97), (33, 102, 172), (67, 147, 195), (146, 197, 222), (209, 229, 240), (247, 247, 247),
    (253, 219, 199), (244, 165, 130), (214, 96, 77), (178, 24, 43), (103, 0, 31),
]


def cmap(stops, t):
    x = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    f = x - i
    a, b = stops[i], stops[i + 1]
    return "#%02x%02x%02x" % tuple(round(a[c] + f * (b[c] - a[c])) for c in range(3))


def contour_segments(get, nx, ny, level):
    """Marching squares over grid values; segments in (row, col) cell-center
    coordinates (VivCell.jsx contourSegs)."""
    segs = []
    table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
             8: [(2, 3)], 9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(0, 3)]}
    for r in range(nx - 1):
        for c in range(ny - 1):
            v = [get(r, c), get(r, c + 1), get(r + 1, c + 1), get(r + 1, c)]
            if not all(finite(x) for x in v):
                continue
            idx = sum(1 << k for k in range(4) if v[k] > level)
            if idx in (0, 15):
                continue

            def f(a, b):
                return (level - a) / ((b - a) or 1e-30)
            pts = {0: (r, c + f(v[0], v[1])), 1: (r + f(v[1], v[2]), c + 1),
                   2: (r + 1, c + f(v[3], v[2])), 3: (r + f(v[0], v[3]), c)}
            for e1, e2 in table[idx]:
                segs.append((pts[e1], pts[e2]))
    return segs


def heat_extent(nx, ny, fig):
    """Where a heatmap's marks go (www/src/lib/axisScale.js heatExtent): a window
    with both pairs set is the picture's extent, ylo at the bottom; with none, a
    mark is in cells (x the column, y the row, centers at whole numbers), row 0
    at the top unless origin("lower"). Returns (xlo, xhi, ylo, yhi, place),
    place(x, y) being the point's fraction of the picture from the left and top."""
    win = fig.get("xlim") and fig.get("ylim")
    xlo, xhi = fig["xlim"] if win else (-0.5, ny - 0.5)
    ylo, yhi = fig["ylim"] if win else (-0.5, nx - 0.5)
    down = not win and fig.get("origin") != "lower"

    def place(x, y):
        fx = (x - xlo) / ((xhi - xlo) or 1)
        fy = (y - ylo) / ((yhi - ylo) or 1) if down else (yhi - y) / ((yhi - ylo) or 1)
        return fx, fy
    return xlo, xhi, ylo, yhi, place


def render_heatmap(chart, width=640):
    """A matrix as colored cells with a colorbar. The color scale spans every
    frame, as on the site, so the shown frame reads against the whole run."""
    fig = chart.get("fig") or {}
    nx, ny = chart["nx"], chart["ny"]
    frames = chart["frames"]
    z = frames[-1]
    vals = [v for f in frames for v in f if finite(v)] or [0.0, 1.0]
    zmin, zmax = min(vals), max(vals)
    # the color limits: clim; else symmetric about zero for a diverging map; else the data's range
    diverging = fig.get("colormap") == "diverging"
    stops = DIVERGING if diverging else VIRIDIS
    clim = fig.get("clim")
    if clim and len(clim) == 2 and all(finite(v) for v in clim) and clim[0] != clim[1]:
        zmin, zmax = clim
    elif diverging:
        m = max(abs(zmin), abs(zmax)) or 1.0
        zmin, zmax = -m, m
    span = (zmax - zmin) or 1.0
    lower = fig.get("origin") == "lower"  # row 0 at the bottom
    # A big field is shown at most 160 cells a side (averaging blocks), which
    # keeps the SVG small; a small matrix keeps every cell.
    step = max(1, math.ceil(max(nx, ny) / 160))
    rows, cols = math.ceil(nx / step), math.ceil(ny / step)
    avail = width - 110
    cell = min(avail / cols, 420 / rows, 40)
    pw, ph = cols * cell, rows * cell
    ml, mt = 20 if not fig.get("ylabel") else 34, 8
    g = []
    for r in range(rows):
        for c in range(cols):
            vals = [z[i * ny + j] for i in range(r * step, min(nx, (r + 1) * step))
                    for j in range(c * step, min(ny, (c + 1) * step))]
            vals = [v for v in vals if finite(v)]
            if not vals:
                continue  # a nan or inf cell is left empty
            v = sum(vals) / len(vals)
            ry = rows - 1 - r if lower else r
            g.append(f'<rect x="{ml + c * cell:.2f}" y="{mt + ry * cell:.2f}" width="{cell + 0.4:.2f}" height="{cell + 0.4:.2f}" fill="{cmap(stops, (v - zmin) / span)}"/>')
    # contour lines at the program's levels (or 7 even ones with the "contour" style)
    levels = [v for v in fig.get("levels", []) if finite(v)]
    if not levels and fig.get("style") == "contour":
        levels = [zmin + span * k / 8 for k in range(1, 8)]
    if levels:
        cw, chh = pw / ny, ph / nx
        stroke = "#282828" if diverging else "#ffffff"

        def py(r):
            return mt + ((nx - 1 - r if lower else r) + 0.5) * chh
        d = []
        for lv in levels:
            for (r1, c1), (r2, c2) in contour_segments(lambda r, c: z[r * ny + c], nx, ny, lv):
                d.append(f"M{ml + (c1 + 0.5) * cw:.2f} {py(r1):.2f}L{ml + (c2 + 0.5) * cw:.2f} {py(r2):.2f}")
        g.append(f'<path d="{"".join(d)}" fill="none" stroke="{stroke}" stroke-opacity="0.55" stroke-width="1"/>')
    # the program's marks, in the window's units or in cells (heat_extent), clipped to the picture
    xlo, xhi, ylo, yhi, place = heat_extent(nx, ny, fig)

    def at(x, y):
        fx, fy = place(x, y)
        return ml + fx * pw, mt + fy * ph
    ink = "#1e1e1e" if diverging else "#ffffff"
    m = []
    for k in marks_of(chart):
        kind = k["kind"]
        if kind == "rect":
            (x0, y0), (x1, y1) = at(k["x"] - k["w"] / 2, k["y"] - k["h"] / 2), at(k["x"] + k["w"] / 2, k["y"] + k["h"] / 2)
            m.append(f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" fill="{ink}" fill-opacity="0.22"/>')
        elif kind == "label":
            x, y = at(k["x"], k["y"])
            m.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="600" fill="{ink}">{escape(k["text"])}</text>')
        elif kind in ("hline", "vline", "segment"):
            if kind == "hline":
                (x0, y0), (x1, y1) = at(xlo, k["y"]), at(xhi, k["y"])
            elif kind == "vline":
                (x0, y0), (x1, y1) = at(k["x"], ylo), at(k["x"], yhi)
            else:
                (x0, y0), (x1, y1) = at(k["x1"], k["y1"]), at(k["x2"], k["y2"])
            m.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{ink}" stroke-opacity="0.9" stroke-width="1.5" stroke-dasharray="5 4"/>')
    if m:
        g.append(f'<svg x="{ml}" y="{mt}" width="{pw:.2f}" height="{ph:.2f}" viewBox="{ml} {mt} {pw:.2f} {ph:.2f}" overflow="hidden">{"".join(m)}</svg>')
    # colorbar
    bx = ml + pw + 16
    for k in range(40):
        t = 1 - k / 39
        g.append(f'<rect x="{bx:.1f}" y="{mt + k * ph / 40:.2f}" width="12" height="{ph / 40 + 0.5:.2f}" fill="{cmap(stops, t)}"/>')
    for lv in levels:
        if zmin <= lv <= zmax:
            y = mt + (1 - (lv - zmin) / span) * ph
            g.append(f'<line x1="{bx:.1f}" x2="{bx + 12:.1f}" y1="{y:.2f}" y2="{y:.2f}" stroke="{INK["title"]}" stroke-width="1.5"/>')
    ticks, st = nice_ticks(zmin, zmax, 4)
    fmt = tick_format(st)
    for t in ticks:
        y = mt + (1 - (t - zmin) / span) * ph
        g.append(f'<text x="{bx + 17:.1f}" y="{y + 3.8:.1f}" font-size="11" fill="{INK["tick"]}">{fmt(t)}</text>')
    h = mt + ph + (30 if fig.get("xlabel") or len(frames) > 1 else 10)
    if fig.get("xlabel"):
        g.append(f'<text x="{ml + pw / 2:.1f}" y="{mt + ph + 20:.1f}" text-anchor="middle" font-size="11.5" fill="{INK["label"]}">{escape(fig["xlabel"])}</text>')
    if fig.get("ylabel"):
        cy = mt + ph / 2
        g.append(f'<text x="14" y="{cy:.1f}" text-anchor="middle" font-size="11.5" fill="{INK["label"]}" transform="rotate(-90 14 {cy:.1f})">{escape(fig["ylabel"])}</text>')
    if len(frames) > 1:
        g.append(f'<text x="{ml}" y="{h - 8}" font-size="11" fill="{INK["tick"]}">last of {len(frames)} frames</text>')
    width = int(bx + 17 + 7 * max(len(fmt(t)) for t in ticks) + 14)
    title = fig.get("title") or chart["label"]
    return _frame(max(width, 260), h, "".join(g), title, None)


def _is_xy(c):
    return c.get("kind") == "xy"


def shared_range(charts, axis):
    """The union of the charts' ranges on one axis, in data units, or None
    (www/src/lib/figureGrid.js sharedRange)."""
    lo, hi, fixed, log = math.inf, -math.inf, True, False
    for c in charts:
        fig = c.get("fig") or {}
        kind = fig.get("xscale" if axis == "x" else "yscale", "linear")
        log = log or (axis == "y" and kind == "log")
        t = scale_of(kind)
        lim = fig.get("xlim" if axis == "x" else "ylim")
        right = set((fig.get("yright") or {}).get("series", []))
        if lim:
            vals = list(lim)
        else:
            fixed = False
            if axis == "x":
                vals = [v for s in c["series"] for v in s["xs"]]
            else:
                vals = [v for s in c["series"] if s.get("name") not in right for v in s["ys"]]
                if any(s["type"] == "bars" for s in c["series"]):
                    vals.append(0.0)
        for v in vals:
            if finite(v) and finite(t(v)):
                lo, hi = min(lo, v), max(hi, v)
    if not finite(lo) or hi == lo:
        return None
    if axis == "y" and not fixed:
        if log:
            a, b = math.log10(lo), math.log10(hi)
            pad = (b - a) * 0.07
            lo, hi = 10 ** (a - pad), 10 ** (b + pad)
        else:
            pad = (hi - lo) * 0.07
            lo, hi = lo - pad, hi + pad
    return (lo, hi)


def layout_panels(grid):
    """The panels of a grid, each chart's figure told the shared ranges and which
    tick labels a neighbor shows (www/src/lib/figureGrid.js layoutPanels)."""
    cols, share = grid["cols"], grid["share"]
    n = len(grid["panels"])
    out = [[dict(c, fig=dict(c.get("fig") or {}, inGrid=True)) if _is_xy(c) else c for c in p] for p in grid["panels"]]
    for axis, group_of in (("x", lambda k: k % cols), ("y", lambda k: k // cols)):
        if axis not in share:
            continue
        groups = {}
        for k, p in enumerate(out):
            groups.setdefault(group_of(k), []).extend(c for c in p if _is_xy(c))
        for charts in groups.values():
            r = shared_range(charts, axis)
            if r:
                for c in charts:
                    c["fig"]["xlim" if axis == "x" else "ylim"] = r
    for k, p in enumerate(out):
        col = k % cols
        right = any(_is_xy(c) and c["fig"].get("yright") for j in range(col, n, cols) for c in out[j])
        for c in p:
            if _is_xy(c):
                c["fig"]["inner"] = {"x": "x" in share and k + cols < n, "y": "y" in share and col > 0}
                c["fig"]["roomRight"] = right
    return out


def render_grid(chart, width=640):
    """A grid of figures: each panel drawn as a figure is, placed in rows and columns
    under the grid's title. A panel that needs more than its share of the width
    (a heatmap keeps a least width) pushes the next one right."""
    cols, gap = chart["cols"], 8
    pw = int((width - gap * (cols - 1)) / cols)
    top = 30 if chart.get("title") else 0
    parts = []
    y = top
    total_w = width
    panels = layout_panels(chart)
    for r in range(0, len(panels), cols):
        row_h = 0
        x = 0.0
        for p in panels[r:r + cols]:
            yy, col_w = y, pw
            for c in p:
                if c.get("kind") not in ("xy", "heatmap"):
                    continue
                svg = render(c, pw)
                w = float(re.match(r'<svg [^>]*width="([\d.]+)"', svg).group(1))
                h = float(re.match(r'<svg [^>]*height="([\d.]+)"', svg).group(1))
                parts.append(svg.replace("<svg ", f'<svg x="{x:.0f}" y="{yy:.0f}" ', 1))
                yy += h + 4
                col_w = max(col_w, w)
            row_h = max(row_h, yy - y)
            x += col_w + gap
        total_w = max(total_w, round(x - gap))
        y += row_h + gap
    head = ""
    if chart.get("title"):
        head = f'<text x="{total_w / 2:.1f}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="{INK["title"]}">{escape(chart["title"])}</text>'
    total = round(y)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total}" viewBox="0 0 {total_w} {total}" '
            f'font-family="{escape(FONT)}" role="img" aria-label="{escape(chart.get("title") or "grid of figures")}">'
            + head + "".join(parts) + "</svg>")


# a diagram's text sizes and marks, as lang/viv/src/figsvg.rs draws them
_NOTE_SIZE = {"small": 11, "normal": 13, "large": 15}
_MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def _word_color(words, default):
    for w in words:
        if re.fullmatch(r"s[1-6]", w):
            return SERIES[int(w[1]) - 1]
        if w == "ink":
            return INK["title"]
        if w == "muted":
            return INK["tick"]
    return default


def _path_d(xs, ys, curved):
    n = len(xs)
    d = f"M{xs[0]:.2f} {ys[0]:.2f}"
    for i in range(1, n):
        if curved and n > 2:
            x0, y0 = (xs[i - 2], ys[i - 2]) if i >= 2 else (xs[i - 1], ys[i - 1])
            x3, y3 = (xs[i + 1], ys[i + 1]) if i + 1 < n else (xs[i], ys[i])
            x1, y1, x2, y2 = xs[i - 1], ys[i - 1], xs[i], ys[i]
            d += f"C{x1 + (x2 - x0) / 6:.2f} {y1 + (y2 - y0) / 6:.2f} {x2 - (x3 - x1) / 6:.2f} {y2 - (y3 - y1) / 6:.2f} {x2:.2f} {y2:.2f}"
        else:
            d += f"L{xs[i]:.2f} {ys[i]:.2f}"
    return d


def render_diagram(chart):
    """A diagram: boxes, arrows and notes where the records put them, one unit
    to a pixel, no axes (figsvg.rs render_diagram)."""
    fig = chart.get("fig") or {}
    pts = []
    for m in chart["marks"]:
        if m["kind"] == "box":
            pts += [(m["x0"], m["y0"]), (m["x1"], m["y1"])]
        elif m["kind"] == "arrow":
            pts += [(x, y) for x, y in zip(m["x"], m["y"]) if finite(x) and finite(y)]
        elif m["kind"] == "note":
            pts.append((m["x"], m["y"]))
    xlo, xhi = fig.get("xlim") or (min((p[0] for p in pts), default=0), max((p[0] for p in pts), default=1))
    ylo, yhi = fig.get("ylim") or (min((p[1] for p in pts), default=0), max((p[1] for p in pts), default=1))
    margin = 12

    def px(x):
        return margin + (x - xlo)

    def py(y):
        return margin + (yhi - y)
    body = []
    for m in chart["marks"]:
        words = m.get("style", "").split()
        if m["kind"] == "box":
            color = _word_color(words, INK["tick"])
            fill, op = (color, 0.07) if "tint" in words else ("none", 1) if "clear" in words else (INK["bg"], 1)
            dash = ' stroke-dasharray="5 3"' if "dash" in words else ' stroke-dasharray="1.5 2.5"' if "dot" in words else ""
            body.append(f'<rect x="{px(min(m["x0"], m["x1"])):.2f}" y="{py(max(m["y0"], m["y1"])):.2f}" width="{abs(m["x1"] - m["x0"]):.2f}" '
                        f'height="{abs(m["y1"] - m["y0"]):.2f}" rx="{0 if "square" in words else 5}" fill="{fill}" fill-opacity="{op}" '
                        f'stroke="{"none" if "noline" in words else color}" stroke-width="{0.8 if "thin" in words else 2 if "thick" in words else 1.2}"{dash}/>')
        elif m["kind"] == "arrow":
            p = [(px(x), py(y)) for x, y in zip(m["x"], m["y"]) if finite(x) and finite(y)]
            if len(p) < 2:
                continue
            color = _word_color(words, INK["title"])
            xs, ys = [a for a, _ in p], [b for _, b in p]
            ux, uy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            ln = math.hypot(ux, uy)
            tip = (xs[-1], ys[-1])
            head = "nohead" not in words and ln > 0
            if head:
                ux, uy = ux / ln, uy / ln
                xs[-1] -= ux * min(7, ln)
                ys[-1] -= uy * min(7, ln)
            dash = ' stroke-dasharray="5 3"' if "dash" in words else ' stroke-dasharray="1.5 2.5"' if "dot" in words else ""
            body.append(f'<path d="{_path_d(xs, ys, "curved" in words)}" fill="none" stroke="{color}" '
                        f'stroke-width="{0.9 if "thin" in words else 2.2 if "thick" in words else 1.3}"{dash}/>')
            if head:
                bx, by = tip[0] - ux * 7, tip[1] - uy * 7
                body.append(f'<path d="M{tip[0]:.2f} {tip[1]:.2f}L{bx - uy * 3.2:.2f} {by + ux * 3.2:.2f}L{bx + uy * 3.2:.2f} {by - ux * 3.2:.2f}Z" fill="{color}"/>')
        elif m["kind"] == "note":
            size = next((_NOTE_SIZE[w] for w in words if w in _NOTE_SIZE), _NOTE_SIZE["small"])
            anchor = "start" if "left" in words else "end" if "right" in words else "middle"
            base = 0.78 * size if "top" in words else -0.22 * size if "bottom" in words else 0.35 * size
            font = f' font-family="{escape(_MONO)}"' if "mono" in words else ""
            weight = ' font-weight="600"' if "bold" in words else ""
            body.append(f'<text x="{px(m["x"]):.2f}" y="{py(m["y"]) + base:.2f}" font-size="{size}" text-anchor="{anchor}" '
                        f'fill="{_word_color(words, INK["title"])}"{font}{weight}>{escape(m["text"])}</text>')
    return _frame(round(xhi - xlo + 2 * margin), round(yhi - ylo + 2 * margin), "".join(body), fig.get("title"), None)


def render(chart, width=640):
    if chart["kind"] == "diagram":
        return render_diagram(chart)
    if chart["kind"] == "grid":
        return render_grid(chart, width)
    if chart["kind"] == "heatmap":
        return render_heatmap(chart, width)
    return render_xy(chart, width)


def describe(chart):
    """A one-line text form of a chart, for consoles and text-only viewers."""
    if chart["kind"] == "diagram":
        n = sum(1 for m in chart["marks"] if m["kind"] == "box")
        return f"[diagram {(chart.get('fig') or {}).get('title', '')}: {n} boxes]"
    if chart["kind"] == "grid":
        inner = "; ".join(describe(c) for p in chart["panels"] for c in p if c.get("kind") in ("xy", "heatmap"))
        return f"[grid {chart['title']}: {chart['rows']} x {chart['cols']}: {inner}]"
    fig = chart.get("fig") or {}
    title = fig.get("title") or chart.get("label") or ""
    if chart["kind"] == "heatmap":
        return f"[heatmap {title}: {chart['nx']} x {chart['ny']}, {len(chart['frames'])} frame(s)]"
    names = ", ".join(dict.fromkeys(s["name"] for s in chart["series"] if s.get("name")))
    kind = "/".join(dict.fromkeys(s["type"] for s in chart["series"]))
    return f"[{kind} chart {title}: {names}]".replace("  ", " ")

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The gallery: worked examples run when the docs are built.

    gallery/index.txt         one line per entry: `<topic> <section> <script>`,
                              in page order
    gallery/<topic>/<name>.py the entry: a Python script in cells

`python -m vipyr.docs` runs every script and writes what it printed and drew
into www/src/data/gallery.json, which /vipyr/gallery renders; `--check`
fails when a fresh run differs from the stored file. Adding an entry is one
script and one line of index.txt.

A script is laid out like a scikit-learn (sphinx-gallery) example:

    \"\"\"
    Title of the page
    =================

    The opening paragraph: what the example shows (markdown).
    \"\"\"

    # %%
    # Text before the next block of code (markdown, each line after "# ").

    import numpy as np
    ...

Cells run in order in one namespace, from the repository root. A cell's
printed output is kept under its code, and every `vipyr.figure` figure it
makes is drawn by `viv plot` and kept as SVG. Everything printed must be the
same on every run: print computed numbers, not timings (a measured table is
read from a committed file and printed, with where it was measured).
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import sys
from pathlib import Path

from . import _spec

# topic -> its sections, in page order; the titles are TOPICS in
# www/src/lib/site.ts, which must list the same ids
TOPICS = {
    "data": ("features",),
    "stats": ("dist", "regression", "tests", "bayes", "timeseries", "survival"),
    "ml": ("nn", "optim", "models"),
    "finance": ("options", "sde", "rates", "risk"),
    "science": ("qchem", "solids", "nuclear", "plasma", "statmech"),
    "math": ("ode", "linalg", "fitting", "accuracy"),
    "control": ("linear", "estimation", "robotics"),
    "hardware": ("logic", "processors", "memory", "analog"),
    "power": ("flow", "stability"),
}
FORMAT = 2


def gallery_dir() -> Path:
    return _spec.repo_root() / "gallery"


def site_file() -> Path:
    return _spec.repo_root() / "www" / "src" / "data" / "gallery.json"


def read_index(root: Path):
    out = []
    for n, line in enumerate((root / "index.txt").read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in TOPICS:
            raise ValueError("gallery/index.txt:%d: want `<topic> <section> <script>`, topic one of %s: %r"
                             % (n, ", ".join(TOPICS), line))
        if parts[1] not in TOPICS[parts[0]]:
            raise ValueError("gallery/index.txt:%d: section %r is not one of %s's: %s"
                             % (n, parts[1], parts[0], ", ".join(TOPICS[parts[0]])))
        out.append((parts[0], parts[1], parts[2]))
    return out


def split_script(text: str):
    """(title, intro markdown, [(markdown, code)]) of a gallery script."""
    tree = ast.parse(text)
    doc = ast.get_docstring(tree, clean=True) or ""
    lines = doc.splitlines()
    if len(lines) < 2 or not re.fullmatch(r"=+", lines[1].strip()):
        raise ValueError("a gallery script opens with a docstring whose first line is the title, underlined with =")
    title, intro = lines[0].strip(), "\n".join(lines[2:]).strip()
    body = text
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        body = "\n".join(text.splitlines()[tree.body[0].end_lineno:])
    cells = []
    for chunk in re.split(r"^# %%[^\n]*\n", body, flags=re.M):
        md, code = [], []
        rows = chunk.splitlines()
        i = 0
        while i < len(rows) and (rows[i].startswith("# ") or rows[i] == "#"):
            md.append(rows[i][2:])
            i += 1
        code = "\n".join(rows[i:]).strip("\n")
        md = "\n".join(md).strip()
        if md or code.strip():
            cells.append((md, code))
    return title, intro, cells


def run_script(path: Path, root: Path):
    from . import figure

    title, intro, cells = split_script(path.read_text(encoding="utf-8"))
    ns = {"__name__": "__gallery__", "__file__": str(path)}
    out_cells = []
    old = os.getcwd()
    os.chdir(root)
    figure.take()
    try:
        with _one_blas_thread():
            for md, code in cells:
                buf = io.StringIO()
                if code.strip():
                    with contextlib.redirect_stdout(buf):
                        exec(compile(code, str(path), "exec"), ns)
                figs = [f.svg() for f in figure.take()]
                out_cells.append({"md": md, "code": code, "out": buf.getvalue().rstrip("\n"), "figs": figs})
    finally:
        os.chdir(old)
    return title, intro, out_cells


@contextlib.contextmanager
def _one_blas_thread():
    """NumPy's BLAS on one thread while a script runs: a threaded reduction
    can round differently from run to run, a page must print the same digits
    every time, and a multithreaded LAPACK call on a loaded machine can run
    100 times slower. Without threadpoolctl the BLAS keeps its threads."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(1):
        yield


def build() -> dict:
    root = _spec.repo_root()
    g = gallery_dir()
    entries = []
    for topic, section, script in read_index(g):
        p = g / script
        print("gallery: running", script, file=sys.stderr)
        title, intro, cells = run_script(p, root)
        summary = intro.split("\n\n")[0].replace("\n", " ").replace("`", "")
        entries.append({"slug": p.stem.replace("_", "-"), "topic": topic, "section": section,
                        "title": title, "summary": summary,
                        "intro": intro, "file": str(p.relative_to(root)), "cells": cells,
                        "figures": sum(len(c["figs"]) for c in cells)})
    slugs = [e["slug"] for e in entries]
    dup = {s for s in slugs if slugs.count(s) > 1}
    if dup:
        raise ValueError("gallery: two scripts make the same page: %s" % ", ".join(sorted(dup)))
    return {"format": FORMAT, "entries": entries}

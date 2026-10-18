#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Convert a Viv notebook (.viv) to a Jupyter notebook (.ipynb) and back.

    python tools/jupyter/vivnb.py post.viv              writes post.ipynb
    python tools/jupyter/vivnb.py post.ipynb            writes post.viv
    python tools/jupyter/vivnb.py post.viv -o out.ipynb
    python tools/jupyter/vivnb.py --check a.viv b.viv   converts each both ways, compares bytes

The .viv file is the source of truth; the .ipynb is a view of it. A Viv
notebook is an ordinary program. A line that is exactly `###` opens a prose
block that runs to the next `###` line, and the compiler skips the block.
The cells are:

  - the post header: the first prose block, when it starts with `key: value`
    lines. It becomes two cells: a raw cell with the key lines (Markdown would
    run them together into one paragraph) and a Markdown cell with the prose
    after them;
  - every later prose block: a Markdown cell;
  - the code between two prose blocks: a code cell.

Converting .viv -> .ipynb -> .viv gives back the same bytes. The blank lines
around a code cell and any fence line that is not exactly `###` are kept in
the cell's metadata (under "viv"), and only when they differ from the usual
layout, so a converted notebook's metadata stays nearly empty. A cell added
in Jupyter gets the usual layout. Two code cells with no Markdown cell
between them cannot stay apart in a .viv file; they are joined, with a blank
line between.
"""

import argparse
import json
import os
import re
import sys

FENCE = "###\n"
_HEADER_KEY = re.compile(r"^[A-Za-z][\w-]*:(\s|$)")
_CODE = re.compile(r"^((?:[ \t\r]*\n)*)(.*?)(\s*)$", re.S)


def _is_open(line):
    # the lexer opens a block at a line whose only text is ###, indented or not
    return line.strip() == "###"


def _is_close(line):
    # and closes it at the next line that is ### with nothing before it
    return line.rstrip() == "###"


def split(text):
    """The file as alternating segments: code, prose, code, ..., code.
    A prose segment is (open line, body, close line); close is "" when the
    block runs to the end of the file."""
    lines = text.splitlines(keepends=True)
    segments = []
    code = []
    i = 0
    while i < len(lines):
        if _is_open(lines[i]):
            segments.append(("code", "".join(code)))
            code = []
            j = i + 1
            while j < len(lines) and not _is_close(lines[j]):
                j += 1
            body = "".join(lines[i + 1:j])
            close = lines[j] if j < len(lines) else ""
            segments.append(("prose", lines[i], body, close))
            i = j + 1
        else:
            code.append(lines[i])
            i += 1
    segments.append(("code", "".join(code)))
    return segments


def _is_header(body):
    first = body.split("\n", 1)[0]
    return bool(_HEADER_KEY.match(first))


def _body_meta(body, meta):
    """A prose body as cell source: its final newline goes, and is noted when absent."""
    if body.endswith("\n"):
        return body[:-1]
    meta["nl"] = False
    return body


def _cell(kind, source, meta, n):
    cell = {"cell_type": kind, "id": f"c{n}", "metadata": {"viv": meta} if meta else {}, "source": source}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def viv_to_cells(text):
    segs = split(text)
    cells = []
    pending_before = None  # a whitespace-only code segment before the next prose cell

    def add(kind, source, meta):
        cells.append(_cell(kind, source, meta, len(cells)))

    for k, seg in enumerate(segs):
        at_start = k == 0
        at_end = k == len(segs) - 1
        if seg[0] == "code":
            code = seg[1]
            m = _CODE.match(code)
            lead, src, trail = m.group(1), m.group(2), m.group(3)
            if not src:
                if at_end:
                    if code and cells:
                        # on the cell that writes the last block (the header
                        # cell writes its prose cell too)
                        last = cells[-1]
                        if last["metadata"].get("viv", {}).get("role") == "header-prose":
                            last = cells[-2]
                        last["metadata"].setdefault("viv", {})["after"] = code
                else:
                    pending_before = code
                continue
            meta = {}
            if lead != ("" if at_start else "\n"):
                meta["lead"] = lead
            if trail != ("\n" if at_end else "\n\n"):
                meta["trail"] = trail
            add("code", src, meta)
            continue

        _, open_, body, close = seg
        meta = {}
        before = pending_before if pending_before is not None else ""
        pending_before = None
        prev_is_code = bool(cells) and cells[-1]["cell_type"] == "code"
        if not prev_is_code and before != ("" if not cells else "\n"):
            meta["before"] = before
        if open_ != FENCE:
            meta["open"] = open_
        if close != FENCE:
            meta["close"] = close
        if not cells and not before.strip() and _is_header(body):
            # the post header: key lines, blank lines, then prose
            m = re.match(r"^(.*?\n)((?:[ \t]*\n)+)(.*)$", body, re.S)
            if m and m.group(3):
                keys, sep, prose = m.groups()
                meta["role"] = "header"
                add("raw", keys[:-1], meta)
                pmeta = {"role": "header-prose"}
                if sep != "\n":
                    pmeta["sep"] = sep
                add("markdown", _body_meta(prose, pmeta), pmeta)
            else:
                meta["role"] = "header"
                add("raw", _body_meta(body, meta), meta)
            continue
        add("markdown", _body_meta(body, meta), meta)
    return cells


def to_ipynb(text, name=None):
    nb = {
        "cells": viv_to_cells(text),
        "metadata": {
            "kernelspec": {"name": "viv", "display_name": "Viv", "language": "viv"},
            "language_info": {"name": "viv", "file_extension": ".viv", "mimetype": "text/x-viv",
                              "codemirror_mode": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if name:
        nb["metadata"]["viv"] = {"source": name}
    return nb


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def cells_to_viv(cells):
    out = []
    prev = None  # None, "code" or "prose"
    for k, cell in enumerate(cells):
        meta = cell.get("metadata", {}).get("viv", {})
        src = _src(cell)
        kind = cell.get("cell_type")
        if kind == "code":
            if not src.strip():
                continue
            nxt = next((c for c in cells[k + 1:] if c.get("cell_type") != "code" or _src(c).strip()), None)
            if prev == "code":
                # two code cells in a row: one code segment in the file
                out.append("\n\n" + src)
            else:
                out.append(meta.get("lead", "" if prev is None else "\n") + src)
            if nxt is None or nxt.get("cell_type") != "code":
                out.append(meta.get("trail", "\n" if nxt is None else "\n\n"))
            prev = "code"
            continue
        if meta.get("role") == "header-prose":
            continue  # written with the header cell before it
        body = src + ("\n" if meta.get("nl", True) else "")
        if meta.get("role") == "header":
            nxt = cells[k + 1] if k + 1 < len(cells) else None
            nmeta = (nxt or {}).get("metadata", {}).get("viv", {})
            if nxt is not None and nmeta.get("role") == "header-prose":
                body = body + nmeta.get("sep", "\n") + _src(nxt) + ("\n" if nmeta.get("nl", True) else "")
        if prev != "code":
            out.append(meta.get("before", "" if prev is None else "\n"))
        out.append(meta.get("open", FENCE) + body + meta.get("close", FENCE))
        out.append(meta.get("after", ""))
        prev = "prose"
    return "".join(out)


def from_ipynb(nb):
    return cells_to_viv(nb["cells"])


def write_ipynb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def roundtrip_ok(text):
    """Does text -> ipynb -> text give back the same bytes? Also through JSON,
    as a saved file would."""
    nb = json.loads(json.dumps(to_ipynb(text)))
    return from_ipynb(nb) == text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", help="output path (one input only)")
    ap.add_argument("--check", action="store_true", help="round-trip each .viv file and report")
    args = ap.parse_args(argv)

    if args.check:
        bad = 0
        for p in args.files:
            with open(p, newline="") as f:
                text = f.read()
            ok = roundtrip_ok(text)
            bad += not ok
            if not ok:
                print(f"DIFFERS  {p}")
        print(f"{len(args.files) - bad} of {len(args.files)} files round-trip byte for byte")
        return 1 if bad else 0

    if args.output and len(args.files) > 1:
        ap.error("-o takes one input file")
    for p in args.files:
        base, ext = os.path.splitext(p)
        if ext == ".viv":
            with open(p, newline="") as f:
                nb = to_ipynb(f.read(), os.path.basename(p))
            out = args.output or base + ".ipynb"
            write_ipynb(nb, out)
        elif ext == ".ipynb":
            with open(p) as f:
                text = from_ipynb(json.load(f))
            out = args.output or base + ".viv"
            with open(out, "w", newline="") as f:
                f.write(text)
        else:
            ap.error(f"{p}: expected a .viv or .ipynb file")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

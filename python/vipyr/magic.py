# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The `%%viv` cell magic for IPython and Jupyter, with run records.

    %load_ext vipyr

    %%viv pipe
    def score(x: Arr, w: Num) -> Arr { ... }

    pipe.score(df, w=2.0)          # `pipe` is now a module in the notebook

The cell is written to `__pycache__/vipyr/cells/<name>-<sha>/<name>.viv`
beside the notebook and loaded with vipyr.load, so an unchanged cell is
not compiled twice and DataFrame arguments work as they do in vipyr.
`%%viv pipe --fast` builds with the fast flags.

Every call of a function of a magic-defined module appends one JSON line to
the run log (`.viv_runs.jsonl` beside the notebook; `%viv_runs PATH` moves
it, `%viv_runs off` stops recording, `%viv_runs` shows where it goes). The
record names the function, the SHA-256 of the cell's Viv source, the
compiler's identity, a hash per input (a hash per column for a DataFrame),
the hash of the output and the time. Its `id` is the content address of
runrecord.py on branch claims: the SHA-256 of what ran, the inputs hash and
the outputs hash, so two runs that agree have one id.

The rerun check: when a function is called again with the same source,
compiler, flags and input hashes as an earlier record in the log, and the
output hash differs, the magic prints a warning that names both records,
and the new record carries `mismatch_of`. Viv code that reads only its
arguments gives the same bits every time, so a mismatch means a hidden
input: state kept between calls, the clock, uninitialized memory.

`%viv_export pipe -o pipeline.c` writes the cell module as the one C file
`viv export` made for it (the same file the shared library is built from).

A compile error in a cell is located in the cell: the filename is
`<%%viv pipe>` and the line number counts the `%%viv` line as line 1, as
Jupyter numbers the cell's lines. The magic shows it as one message (an
IPython UsageError whose `viv_error` is the VivCompileError); `define`
raises the VivCompileError itself.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import re
import shlex
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np

from . import _core
from .frames import column_array, frame_library

SCHEMA = "viv.cell-run/1"
RUNS_FILE = ".viv_runs.jsonl"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# what the magic knows: the modules it defined and where the records go
state = {"runs": None, "enabled": True, "base": None, "modules": {}}
_index: dict = {}          # run-log path -> {rerun key: the latest record}


# ------------------------------------------------------------------ hashing

def _h(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def _array_hash(a) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "O":
        return _h("object", a.shape, repr(a.tolist()))
    return _h(a.dtype.str, a.shape, a.tobytes())


def column_hash(df, col) -> str:
    """The hash of one column as Viv reads it: its values, null mask and
    conversion (categories, time zone); a column Viv cannot take is hashed
    by its library's own values."""
    try:
        vals, mask, meta = column_array(df, col)
        parts = [_array_hash(vals), "-" if mask is None else _array_hash(mask)]
        if meta is not None:
            cats = None if meta.categories is None else repr(list(meta.categories))
            parts += [meta.kind, cats, str(meta.tz)]
        return _h(*parts)
    except (TypeError, ValueError):
        s = df[col] if frame_library(df) == "pandas" else df.get_column(col)
        if frame_library(df) == "pandas":
            import pandas as pd
            return _h(str(s.dtype), _array_hash(pd.util.hash_pandas_object(s, index=False).to_numpy()))
        return _h(str(s.dtype), repr(s.to_list()))


def frame_hashes(df) -> dict:
    """{column: hash} for a pandas or polars frame (a pandas frame's index
    under `__index__`)."""
    out = {str(c): column_hash(df, c) for c in df.columns}
    if frame_library(df) == "pandas":
        import pandas as pd
        out["__index__"] = _h(_array_hash(pd.util.hash_pandas_object(df.index).to_numpy()))
    return out


def describe(v):
    """A JSON description of a value whose hashes are stable across runs: a
    frame as a hash per column, an array by dtype, shape and bytes, a scalar
    by its exact bits, containers element by element."""
    lib = frame_library(v)
    if lib is not None:
        cols = frame_hashes(v)
        return {"frame": lib, "rows": len(v), "columns": cols, "hash": _h(json.dumps(cols, sort_keys=True))}
    mod = type(v).__module__ or ""
    if mod.startswith(("pandas", "polars")) and type(v).__name__ == "Series":
        return {"series": str(v.name), "hash": _array_hash(v.to_numpy())}
    if v is None:
        return {"hash": _h("none")}
    if isinstance(v, bool):
        return {"hash": _h("bool", int(v))}
    if isinstance(v, (int, np.integer)):
        return {"hash": _h("int", struct.pack("<q", int(v)))}
    if isinstance(v, (float, np.floating)):
        return {"hash": _h("float", struct.pack("<d", float(v)))}
    if isinstance(v, (complex, np.complexfloating)):
        return {"hash": _h("complex", struct.pack("<dd", v.real, v.imag))}
    if isinstance(v, str):
        return {"hash": _h("str", v)}
    if isinstance(v, np.ndarray):
        return {"shape": list(v.shape), "dtype": v.dtype.str, "hash": _array_hash(v)}
    if isinstance(v, dict):
        items = {str(k): describe(x) for k, x in v.items()}
        return {"items": items, "hash": _h("dict", json.dumps({k: d["hash"] for k, d in items.items()}, sort_keys=True))}
    if isinstance(v, (tuple, list)):
        items = [describe(x) for x in v]
        return {"items": items, "hash": _h("seq", *[d["hash"] for d in items])}
    if hasattr(v, "_asdict"):
        return describe(dict(v._asdict()))
    try:
        return describe(np.asarray(v)) if not hasattr(v, "__dict__") else describe(dict(vars(v)))
    except Exception:
        return {"hash": _h("repr", repr(v))}


# -------------------------------------------------------------- run records

def notebook_dir(ns=None) -> Path:
    """The directory of the running notebook: VS Code's `__vsc_ipynb_file__`,
    Jupyter's $JPY_SESSION_NAME when it is a path, else the working
    directory (where Jupyter starts a notebook's kernel)."""
    for p in ((ns or {}).get("__vsc_ipynb_file__"), os.environ.get("JPY_SESSION_NAME")):
        if p and str(p).endswith(".ipynb") and os.path.isdir(os.path.dirname(os.path.abspath(p))):
            return Path(os.path.abspath(p)).parent
    return Path.cwd()


def runs_path() -> Path:
    if state["runs"] is not None:
        return Path(state["runs"])
    if os.environ.get("VIPYR_RUNS"):
        return Path(os.environ["VIPYR_RUNS"])
    return (state["base"] or Path.cwd()) / RUNS_FILE


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_id(rec) -> str:
    """runrecord.py's content address (branch claims): program, engine,
    inputs hash, outputs hash; not the time."""
    core = {k: rec[k] for k in ("program", "engine", "inputs_hash", "outputs_hash")}
    return "rr-" + hashlib.sha256(_canonical(core).encode()).hexdigest()[:16]


def _rerun_key(rec):
    return _canonical([rec["program"], rec["engine"], rec["inputs_hash"]])


def _load_index(path: Path) -> dict:
    key = str(path)
    if key not in _index:
        idx = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        idx[_rerun_key(r)] = r
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        _index[key] = idx
    return _index[key]


def make_record(module, function, source_sha, fast, args, kw, out, wall_s):
    inputs = {"args": [describe(a) for a in args], "kwargs": {k: describe(v) for k, v in sorted(kw.items())}}
    inputs_hash = _h(_canonical({"args": [a["hash"] for a in inputs["args"]],
                                 "kwargs": {k: v["hash"] for k, v in inputs["kwargs"].items()}}))
    output = describe(out)
    rec = {
        "schema": SCHEMA,
        "program": {"module": module, "function": function, "source_sha256": source_sha},
        "engine": {"kind": "vipyr", "viv": _core.compiler_identity(), "mode": "fast" if fast else "exact"},
        "inputs": inputs,
        "inputs_hash": inputs_hash,
        "output": output,
        "outputs_hash": output["hash"],
        "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "wall_s": round(wall_s, 6),
    }
    rec["id"] = record_id(rec)
    return rec


def append_record(rec, path=None, stream=None):
    """Append `rec` to the run log and run the rerun check against the log;
    returns the earlier record it disagrees with, or None."""
    path = Path(path) if path is not None else runs_path()
    idx = _load_index(path)
    key = _rerun_key(rec)
    prev = idx.get(key)
    if prev is not None and prev["outputs_hash"] != rec["outputs_hash"]:
        rec["mismatch_of"] = prev["id"]
        p = rec["program"]
        print("vipyr: rerun mismatch: %s.%s gave a different output on the same inputs\n"
              "    earlier: %s  (%s)\n"
              "    now:     %s  (%s)\n"
              "    same Viv source, compiler and input hashes; the function reads something besides its "
              "arguments (state kept between calls, the clock, uninitialized memory). Records in %s"
              % (p["module"], p["function"], prev["id"], prev["time"], rec["id"], rec["time"], path),
              file=stream or sys.stderr)
    else:
        prev = None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_canonical(rec) + "\n")
    idx[key] = rec
    return prev


def _cell_location(text: str, module: str) -> str:
    """A run-time error's `--> NAME.viv:L:C` and `L |` gutter moved to the
    cell's numbering (the `%%viv` line is line 1)."""
    m = re.search(r"-->\s*%s\.viv:(\d+):(\d+)" % re.escape(module), text)
    if not m:
        return text
    n = int(m.group(1))
    text = text.replace(m.group(0), "--> <%%%%viv %s>:%d:%s" % (module, n + 1, m.group(2)))
    return re.sub(r"(?m)^(\s*)%d( \|)" % n, lambda g: "%s%d%s" % (g.group(1), n + 1, g.group(2)), text, count=1)


def recorded(fn, module, function, source_sha, fast, error_class=None):
    """`fn` with a run record appended after each successful call; a Viv
    run-time error (`error_class`) is relocated to the cell's lines."""
    @functools.wraps(fn)
    def call(*args, **kw):
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kw)
        except Exception as e:
            if error_class is not None and isinstance(e, error_class) and e.args and isinstance(e.args[0], str):
                e.args = (_cell_location(e.args[0], module),) + e.args[1:]
            raise
        wall = time.perf_counter() - t0
        if state["enabled"]:
            append_record(make_record(module, function, source_sha, fast, args, kw, out, wall))
        return out
    call.__vipyr_unrecorded__ = fn
    return call


# -------------------------------------------------------------------- cells

def cell_file(name: str, source: str, base=None) -> Path:
    """Where the cell's source is kept: one directory per content, so a
    cell's build is cached like any vipyr.load of an unchanged file."""
    sha = hashlib.sha256(source.encode()).hexdigest()
    d = Path(base or Path.cwd()) / "__pycache__" / "vipyr" / "cells" / ("%s-%s" % (name, sha[:16]))
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name + ".viv")
    if not p.is_file() or p.read_text(encoding="utf-8") != source:
        tmp = d / (".%s.%d" % (name, os.getpid()))
        tmp.write_text(source, encoding="utf-8")
        os.replace(tmp, p)
    return p


def _cell_error(e: _core.VivCompileError, name: str, path: Path, source: str):
    """The compiler's error moved from the cell's file to the cell: the
    `%%viv` line is line 1, so the file's line n is the cell's line n + 1."""
    try:
        same = e.filename and Path(e.filename).resolve() == path.resolve()
    except OSError:
        same = False
    if not same:
        return e                                   # an imported module's error stays in that file
    where = "<%%%%viv %s>" % name
    lineno = e.lineno + 1 if e.lineno else None
    lines = source.splitlines()
    text = lines[e.lineno - 1] + "\n" if e.lineno and e.lineno <= len(lines) else e.text
    msg = re.sub(r"\(%s, line (\d+)\)" % re.escape(path.name),
                 lambda m: "(cell %s, line %d)" % (name, int(m.group(1)) + 1), e.msg)
    err = _core.VivCompileError(msg, (where, lineno, e.offset, text))
    err.viv_output = getattr(e, "viv_output", "")
    return err


def format_error(e: _core.VivCompileError) -> str:
    """A compile error as the compiler would print it, at the cell's line."""
    out = "%s, line %s: %s" % (e.filename, e.lineno, e.msg)
    if e.text:
        out += "\n    " + e.text.rstrip("\n")
        if e.offset:
            out += "\n    " + " " * (e.offset - 1) + "^"
    return out


def define(name: str, source: str, fast=False, base=None) -> types.ModuleType:
    """Compile a cell's Viv source and return it as the module `name`, its
    functions recording each call."""
    if not _NAME.match(name or ""):
        raise ValueError("%%%%viv needs a module name, a Python identifier: %%%%viv %s" % (name or "NAME"))
    base = Path(base) if base is not None else (state["base"] or Path.cwd())
    path = cell_file(name, source, base)
    try:
        mod = _core.load(path, fast=fast, name=name)
    except _core.VivCompileError as e:
        raise _cell_error(e, name, path, source) from None
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    for fn in list(getattr(mod, "__all__", [])):
        raw = getattr(mod, fn, None)
        if callable(raw):
            setattr(mod, fn, recorded(raw, name, fn, source_sha, fast, getattr(mod, "VivError", None)))
    mod.__vivcell__ = {"source": source, "sha256": source_sha, "path": str(path), "fast": fast}
    state["modules"][name] = mod
    return mod


def export_c(name: str, out) -> Path:
    """Write the C file of the cell module `name` (what `viv export` made
    for it, the shared library's source) to `out`."""
    mod = state["modules"].get(name)
    if mod is None:
        raise KeyError("no %%%%viv module named %r in this session (defined: %s)"
                       % (name, ", ".join(sorted(state["modules"])) or "none"))
    e = mod.__vipyr__
    c = e.path / (e.name + ".c")
    out = Path(out)
    out.write_bytes(c.read_bytes())
    return out


# ----------------------------------------------------------------- IPython

def _magics_class():
    from IPython.core.error import UsageError
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

    @magics_class
    class VivMagics(Magics):
        @cell_magic
        def viv(self, line, cell):
            """%%viv NAME [--fast]: compile the cell's Viv defs into the module NAME."""
            words = shlex.split(line)
            fast = "--fast" in words
            names = [w for w in words if w != "--fast"]
            if len(names) != 1:
                raise ValueError("usage: %%viv NAME [--fast]")
            state["base"] = state["base"] or notebook_dir(self.shell.user_ns)
            try:
                mod = define(names[0], cell, fast=fast)
            except _core.VivCompileError as e:
                # one located message, not a traceback through the magic
                err = UsageError(format_error(e))
                err.viv_error = e
                raise err from None
            self.shell.user_ns[names[0]] = mod
            for n in mod.__vipyr__.notes:
                print("note: %s" % n)

        @line_magic
        def viv_export(self, line):
            """%viv_export NAME -o FILE.c: write the cell module NAME as one C file."""
            words = shlex.split(line)
            if len(words) != 3 or words[1] not in ("-o", "--output"):
                raise ValueError("usage: %viv_export NAME -o FILE.c")
            out = export_c(words[0], words[2])
            print("wrote %s" % out)

        @line_magic
        def viv_runs(self, line):
            """%viv_runs [PATH | off | on]: where run records go, or stop/start recording."""
            arg = line.strip()
            if arg == "off":
                state["enabled"] = False
            elif arg == "on":
                state["enabled"] = True
            elif arg:
                state["runs"] = os.path.abspath(os.path.expanduser(arg))
            state["base"] = state["base"] or notebook_dir(self.shell.user_ns)
            print("run records: %s%s" % (runs_path(), "" if state["enabled"] else " (off)"))

    return VivMagics


def load_ipython_extension(ip):
    ip.register_magics(_magics_class())

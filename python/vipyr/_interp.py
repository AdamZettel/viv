# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Running a .viv file in Viv's interpreter when there is no C compiler.

`vipyr.load(path)` normally builds a shared library. When no C compiler is
installed and no precompiled library matches (a user's own file on a machine
without `cc`, a locked-down laptop, Windows without MinGW), it runs the
module's functions in the interpreter of the bundled `viv` binary instead,
and says so once per process:

    vipyr: no C compiler found; running kernels.viv in Viv's interpreter
    (typically 10-100x slower than compiled). Install a C compiler for speed.

The module is the one `viv export` writes (no C compiler is needed for the
export), so the call interface is the same: the same functions, argument
checks, DataFrame arguments (vipyr.frames), results and error classes. Each
call writes the arguments to a scratch directory, runs a generated driver
that imports the def and prints every result with `repr` (shortest
round-trip text, so every double comes back bit for bit), and parses the
output. The interpreter computes what the exact build computes: the exact
build's flags exist to make the C agree with the interpreter.

Not supported in the interpreter, raised as VivBuildError naming the reason:
a program manifest (`<name>.export.toml` with `[params]`; programs need the
compiled library), Complex arguments and results (the export carries Cplx;
the driver does not), and record (struct) parameters and results. Errors
raised by the def come back as the module's VivError with Viv's message,
not mapped to the subclasses (VivIndexError, ...). Set VIPYR_INTERP=0 to
get the VivBuildError instead of the fallback.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from . import _core

_noticed = set()
stats = {"interp_calls": 0}


def _notice(src: Path):
    if "any" in _noticed:
        return
    _noticed.add("any")
    sys.stderr.write("vipyr: no C compiler found; running %s in Viv's interpreter (typically 10-100x slower "
                     "than compiled; results are the exact build's). Install a C compiler (Xcode command line "
                     "tools, gcc or clang; MinGW-w64 gcc on Windows) for speed.\n" % src.name)


def _strip_library(text: str) -> str:
    """The wrapper without the part that loads and initializes the shared
    library: from `_lib = _C.CDLL(` up to the error classes."""
    a = text.index("_lib = _C.CDLL(_lib_path)")
    b = text.index("class VivError(")
    return text[:a] + "\n_lib = None\n\n" + text[b:]


def _num_text(v):
    return repr(float(v))


def _driver(src: Path, kernel: str, args, res, files):
    lines = ["from %s import %s" % (src.stem, kernel)]
    names = []
    for i, ((name, kind, *_), v) in enumerate(zip(args, files)):
        n = "vipyr_a%d" % i
        if kind == "Num":
            lines.append("%s = float(%r)" % (n, v))
        elif kind == "Int":
            lines.append("%s: Int = int(%r)" % (n, v))
        elif kind == "Bool":
            lines.append("%s = %s" % (n, v))
        elif kind in ("Arr", "IntArr"):
            path, shape = v
            ln = "vipyr_l%d" % i
            lines.append("%s = read_lines(%r)" % (ln, path))
            if kind == "Arr":
                lines.append("%s = zeros(len(%s))" % (n, ln))
                conv = "float"
            else:
                lines.append("%s = int_zeros(len(%s))" % (n, ln))
                conv = "int"
            lines.append("for vipyr_i to len(%s) {\n    %s[vipyr_i] = %s(%s[vipyr_i])\n}" % (ln, n, conv, ln))
            if len(shape) != 1:
                lines.append("%s = reshape(%s, %s)" % (n, n, ", ".join(str(d) for d in shape)))
        else:
            raise _core.VivBuildError("vipyr: %s.%s takes a %s (%s), which the interpreter fallback does not "
                                      "pass; install a C compiler" % (src.stem, kernel, kind, name))
        names.append(n)
    outs = ["vipyr_r%d" % i for i in range(len(res))]
    # an unpacked name is taken for a real array unless declared first
    # (`a, b = f()` with an Int array result)
    lines += ["%s = int_zeros(0)" % o for o, (_, kind) in zip(outs, res) if kind == "IntArr" and len(res) > 1]
    lines.append("%s = %s(%s)" % (", ".join(outs), kernel, ", ".join(names)))
    for o, (rname, kind) in zip(outs, res):
        if kind in ("Num", "Int"):
            lines.append('print("@S", repr(%s))' % o)
        elif kind == "Bool":
            lines.append('print("@S", %s)' % o)
        elif kind == "IntArr":            # [Int] is rank 1
            lines.append('print("@A", len(%s))' % o)
            lines.append('for vipyr_i to len(%s) {\n    print(%s[vipyr_i])\n}' % (o, o))
        elif kind == "Arr":
            lines.append('print("@A", shape(%s))' % o)
            lines.append('%s_f = ravel(%s)\nfor vipyr_i to len(%s_f) {\n    print(repr(%s_f[vipyr_i]))\n}' % (o, o, o, o))
        else:
            raise _core.VivBuildError("vipyr: %s.%s returns a %s (%s), which the interpreter fallback does not "
                                      "read; install a C compiler" % (src.stem, kernel, kind, rname))
    return "\n".join(lines) + "\n"


def _parse(out: str, res):
    it = iter(out.splitlines())
    vals = []
    for rname, kind in res:
        for line in it:
            if line.startswith("@S ") or line.startswith("@A "):
                break
        else:
            raise _core.VivBuildError("vipyr: the interpreter printed no value for %s" % rname)
        body = line[3:].strip()
        if kind == "Num":
            vals.append(float(body))
        elif kind == "Int":
            vals.append(int(float(body)) if "." in body or "e" in body else int(body))
        elif kind == "Bool":
            vals.append(body == "true")
        else:
            shape = tuple(int(x) for x in re.findall(r"-?\d+", body))
            n = int(np.prod(shape)) if shape else 1
            dt = np.int64 if kind == "IntArr" else np.float64
            data = [next(it) for _ in range(n)]
            arr = np.array([int(x) for x in data] if kind == "IntArr" else [float(x) for x in data], dtype=dt)
            vals.append(arr.reshape(shape))
    return vals


def load(src: Path, name: str) -> types.ModuleType:
    """The module of `src` with every function run by the interpreter."""
    out = Path(tempfile.mkdtemp(prefix="vipyr-interp-"))
    try:
        stem, notes, _ = _core.export(src, out)
        text = (out / (stem + ".py")).read_text(encoding="utf-8")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    _notice(src)
    mod = types.ModuleType(name)
    mod.__dict__["__file__"] = str(src)
    var = stem.upper() + "_LIB"
    old = os.environ.get(var)
    os.environ[var] = os.devnull            # the wrapper must not try to build
    try:
        exec(compile(_strip_library(text), str(src) + " (interpreter)", "exec"), mod.__dict__)
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old
    g = mod.__dict__

    def _call(kernel, vals):
        args, res = g["KERNELS"][kernel]
        if g["RECORD_ARGS"].get(kernel) or kernel in g["RECORD_RESULTS"]:
            raise _core.VivBuildError("vipyr: %s.%s takes or returns a record, which the interpreter fallback "
                                      "does not pass; install a C compiler" % (stem, kernel))
        vals = g["_expand"](kernel, vals)
        if len(vals) != len(args):
            raise TypeError("%s takes %d arguments, not %d" % (kernel, len(args), len(vals)))
        with tempfile.TemporaryDirectory(prefix="vipyr-call-") as tmp:
            specs = []
            for i, ((aname, kind, lo, hi, rank, nonfinite, lo_excl, hi_excl, square), v) in enumerate(zip(args, vals)):
                v = g["_check"](aname, kind, lo, hi, rank, None, nonfinite, v, None, lo_excl, hi_excl)
                if kind in ("Arr", "IntArr"):
                    p = os.path.join(tmp, "a%d.txt" % i)
                    flat = np.ascontiguousarray(v).ravel()
                    with open(p, "w") as f:
                        f.write("\n".join(map(repr, flat.tolist())) + "\n")
                    specs.append((p, v.shape))
                elif kind == "Num":
                    specs.append(_num_text(v))
                elif kind == "Int":
                    specs.append(str(int(v)))
                elif kind == "Bool":
                    specs.append("true" if v else "false")
                else:
                    specs.append(v)
            drv = _driver(src, kernel, args, res, specs)
            # the driver sits beside the source, so `from <stem> import` finds it
            # and the source's own imports resolve as they do for the export
            fd, dpath = tempfile.mkstemp(prefix=".vipyr-call-", suffix=".viv", dir=str(src.parent))
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(drv)
                stats["interp_calls"] += 1
                r = subprocess.run([_core.compiler(), "run", "--quiet", "--release", dpath],
                                   capture_output=True, text=True, env=_core._env())
            finally:
                os.unlink(dpath)
        if r.returncode != 0:
            msg = (r.stderr.strip() or r.stdout.strip()).replace(dpath, str(src))
            raise g["VivError"](msg)
        vals = _parse(r.stdout, res)
        return vals[0] if len(vals) == 1 else tuple(vals)

    g["_call"] = _call
    g["version"] = lambda: "interpreter"
    g["tests"] = lambda: []
    g["_FAST"] = False
    mod.__vipyr__ = None
    mod.__vipyr_interp__ = True
    mod.__file__ = str(src)
    from .frames import wrap_module
    return wrap_module(mod)

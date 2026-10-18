#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""diff.py: run a numeric-Python program in CPython+NumPy and its Viv translation in three
engines, and compare every printed value and every test.

    python3 tools/py2viv/diff.py prog.py                 report on stdout, files in prog.py2viv/
    python3 tools/py2viv/diff.py prog.py --out DIR --rtol 1e-12 --atol 1e-14
    python3 tools/py2viv/diff.py prog.py --json          the report as JSON on stdout

The engines are the ones tools/three-engines.sh runs: the Viv interpreter (`viv run`),
Viv -> C -> native (`cc -O2`), and Viv -> wasm by the direct backend, run in node
(tools/run-wasm.mjs). VIV=<path> picks the viv binary (default target/release/viv in this
checkout), VIV_WASM=<path> the viv.wasm runtime (default www/public/viv/viv.wasm), CC the C
compiler.

What is compared:
  1. The three engines: the same output and exit status, byte for byte; a run that stops
     stops with the same error line in all three (a wasm module reports only that line).
  2. Python's standard output against Viv's. Each Viv print line is CPython's text for
     numbers, lists and the supported format specs (fmt); bools, complex numbers and NumPy
     arrays are read back from the line (the manifest gives each print's parts) and printed
     by Python's own str/repr/format and NumPy. A line is `identical` when the text matches.
  3. Every printed value exactly: Python's print arguments are recorded at full precision
     (the program is instrumented) and compared with the values read back from Viv, bit for
     bit (-0.0 is not 0.0), or within a DECLARED tolerance.
  4. How the run ends: both finish, or both stop with the same exception line (Python's
     traceback's last line, Viv's `error:` line; a computed message compares its type).
  5. Tests: `def test_*` functions, called by the runner after the program as pytest would,
     against the Viv `test` blocks' PASS/FAIL lines.

Tolerances: none by default. A program whose translation uses an operation Viv does not
compute bit for bit as CPython/NumPy (listed in the manifest's `inexact`: libm, BLAS,
LAPACK, FFT) states one in a comment, `# py2viv: rtol=1e-12 atol=0`, or on the command line.
"""

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import py2viv  # noqa: E402

try:
    import numpy as np
except ImportError:  # the report can still compare numbers; arrays print as lists
    np = None

RUNNER = r'''
import builtins, json, numbers, runpy, sys, traceback
path, out = sys.argv[1], sys.argv[2]
sys.argv = [path]
res = {"tests": {}, "error": None, "exc_line": None, "test_order": [], "prints": [], "print_sites": []}
try:
    import numpy as np
except ImportError:
    np = None


def ser(a):
    """A printed argument at full precision; a string (and a list of strings) with its text, which
    reading Viv's line back needs (where the string's text ends)."""
    if isinstance(a, str):
        return {"s": 1, "t": a}
    if isinstance(a, list) and a and all(isinstance(x, str) for x in a):
        return {"s": 1, "strs": 1, "t": repr(a)}
    if isinstance(a, bool) or (np is not None and isinstance(a, np.bool_)):
        return {"v": [1.0 if a else 0.0], "bool": 1}
    if isinstance(a, numbers.Integral):
        return {"v": [float(a)], "int": str(int(a))}
    if isinstance(a, numbers.Real):
        return {"v": [float(a)]}
    if isinstance(a, numbers.Complex):
        return {"re": [a.real], "im": [a.imag]}
    if np is None and isinstance(a, (list, tuple)):
        flat, shape = [], [len(a)]
        rows = a
        if a and isinstance(a[0], (list, tuple)):
            shape.append(len(a[0]))
            rows = [x for r in a for x in r]
        for x in rows:
            if isinstance(x, complex):
                return {"other": repr(a)[:200]}
            flat.append(float(x))
        return {"v": flat, "shape": shape}
    try:
        x = np.asarray(a)
        box = {"list": 1, "dt": x.dtype.kind} if isinstance(a, (list, tuple)) else {"nd": 1, "dt": x.dtype.kind}
        if x.dtype.kind == "c":
            return dict(box, re=x.real.ravel().tolist(), im=x.imag.ravel().tolist(), shape=list(x.shape))
        return dict(box, v=x.astype(float).ravel().tolist(), shape=list(x.shape))
    except Exception:
        return {"other": repr(a)[:200]}


_print = builtins.print
_pending = []


def _is_obj(v):
    """An object py2viv prints field by field (a dict, a named tuple, a dataclass, an object of
    a class of the program): its values are compared as the line's text, which is their repr."""
    import dataclasses
    return isinstance(v, dict) or isinstance(v, tuple) and hasattr(v, "_fields") or \
        dataclasses.is_dataclass(v) and not isinstance(v, type) or getattr(type(v), "__module__", None) == "__main__"


def _py2viv_rec(v):
    """Wrapped around every value a print call prints (diff.py rewrites the program so), in
    the order Python evaluates them, which is the order the translation numbers them."""
    _pending.append(None if _is_obj(v) else ser(v))
    return v


def _recording_print(*args, **kw):
    site = kw.pop("_py2viv_site", None)
    n = kw.pop("_py2viv_n", 0)
    # this call's values are the last n recorded: a print inside the evaluation of another
    # print's arguments (a def that prints, called in the arguments) ran and took its own
    # values before this call's later arguments were recorded
    mine = _pending[len(_pending) - n:] if n else []
    del _pending[len(_pending) - n:]
    if kw.get("file") is None:
        res["prints"].append([x for x in mine if x is not None])
        res["print_sites"].append(site)
    _print(*args, **kw)


builtins._py2viv_rec = _py2viv_rec
builtins.print = _recording_print
g = {}
try:
    g = runpy.run_path(path, run_name="__main__")
except SystemExit:
    pass
except BaseException as e:
    res["error"] = traceback.format_exc(limit=4)
    res["exc_line"] = traceback.format_exception_only(type(e), e)[-1].rstrip("\n")
for name, fn in list(g.items()):
    if name.startswith("test_") and callable(fn) and getattr(fn, "__code__", None) is not None \
            and fn.__code__.co_argcount == 0:
        res["test_order"].append(name)
        try:
            fn()
            res["tests"][name] = "PASS"
        except AssertionError:
            res["tests"][name] = "FAIL"
        except Exception as e:
            res["tests"][name] = "ERROR " + repr(e)
sys.stdout.flush()
json.dump(res, open(out, "w"))
'''


class _Instrument(ast.NodeTransformer):
    """Wrap each value a print call prints in _py2viv_rec(...), choosing the same node the
    translator treats as the value (the x of repr(x) and of x.tolist(), each {} of an
    f-string)."""

    def wrap(self, node):
        self.n_wrapped += 1
        return ast.copy_location(ast.Call(func=ast.Name(id="_py2viv_rec", ctx=ast.Load()), args=[node], keywords=[]), node)

    def value(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "repr" and len(node.args) == 1:
            if not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                node.args[0] = self.wrap(node.args[0])  # (the repr of a string constant is text)
            return node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tolist" and not node.args:
            node.func.value = self.wrap(node.func.value)
            return node
        return self.wrap(node)

    def text(self, a):
        """A print argument that builds text (py2viv's funcs.py, _print_text): `+` of strings,
        f-strings, str(x) and repr(x); wrap the values inside it. None when it is not one."""
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a
        if isinstance(a, ast.JoinedStr):
            for v in a.values:
                if isinstance(v, ast.FormattedValue) and not (isinstance(v.value, ast.Constant) and isinstance(v.value.value, str)):
                    v.value = self.value(v.value)
            return a
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("str", "repr") and len(a.args) == 1 \
                and not a.keywords:
            if not (isinstance(a.args[0], ast.Constant) and isinstance(a.args[0].value, str)):
                a.args[0] = self.wrap(a.args[0])
            return a
        if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add) and self.is_text(a):
            a.left = self.text(a.left)
            a.right = self.text(a.right)
            return a
        return self.value(a)  # a side of `+` that is not text by its form (a string py2viv prints as one value)

    def is_text(self, a):
        if isinstance(a, (ast.Constant, ast.JoinedStr)):
            return isinstance(a, ast.JoinedStr) or isinstance(a.value, str)
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("str", "repr"):
            return True
        return isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add) and \
            (self.is_text(a.left) or self.is_text(a.right)) and \
            all(self.is_text(x) or isinstance(x, ast.BinOp) for x in (a.left, a.right))

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.n_wrapped = 0
            args = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) or isinstance(a, ast.Starred):
                    args.append(a)  # (a starred list of strings is read back as the line's text)
                elif isinstance(a, ast.JoinedStr):
                    for v in a.values:
                        if isinstance(v, ast.FormattedValue) and not (isinstance(v.value, ast.Constant) and isinstance(v.value.value, str)):
                            v.value = self.value(v.value)
                    args.append(a)
                elif (isinstance(a, ast.BinOp) or isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                      and a.func.id == "str") and self.is_text(a):
                    args.append(self.text(a))
                else:
                    args.append(self.value(a))
            node.args = args
            node.keywords.append(ast.keyword(arg="_py2viv_site", value=ast.Constant(value=(node.lineno, node.col_offset))))
            node.keywords.append(ast.keyword(arg="_py2viv_n", value=ast.Constant(value=self.n_wrapped)))
        return node


def instrument(src):
    tree = ast.parse(src)
    tree = _Instrument().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def default_viv():
    for p in (os.environ.get("VIV"), os.path.join(ROOT, "target", "release", "viv")):
        if p and os.path.exists(p):
            return p
    return "viv"


def default_wasm():
    for p in (os.environ.get("VIV_WASM"), os.path.join(ROOT, "www", "public", "viv", "viv.wasm")):
        if p and os.path.exists(p):
            return p
    return None


def run(cmd, cwd=None, env=None, timeout=900):
    """Run cmd: (exit status, stdout, stderr, seconds). The output is decoded as UTF-8 with no
    newline translation, so a "\r" a program prints is compared as it is."""
    t0 = time.time()
    dec = lambda b: (b or b"").decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, timeout=timeout)
        return p.returncode, dec(p.stdout), dec(p.stderr), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return -9, dec(e.stdout), "timeout after %ds" % timeout, time.time() - t0


# ---------------------------------------------------------------- running the engines

NOISE = re.compile(r"^(__graph__|__trace__|__steps__|__story__)|assertions? FAILED$|holds? FAILED$")


def _clean(text):
    """The output without the engines' own lines (__graph__, ...). Lines end at "\n" only (a
    printed string may hold U+2028 or \x0c, which splitlines would cut at), and the last line
    keeps its end, so an empty line printed last is still a line."""
    lines = text.split("\n")
    ended = bool(lines) and lines[-1] == ""
    if ended:
        lines.pop()
    kept = [l for l in lines if not NOISE.search(l)]
    return "".join(l + "\n" for l in kept) if ended else "\n".join(kept)


def data_files(src, cwd, workdir):
    """The files a program may read, for the wasm engine (a browser's program reads only the
    files attached to it): every string constant of the Python that names
    an existing file, relative to the directory the program runs in. Each is copied now, before
    any engine runs, so a run that writes it does not change what the next one reads."""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    seen = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and 0 < len(n.value) < 400 and "\0" not in n.value \
                and n.value not in seen:
            seen.add(n.value)
            path = os.path.join(cwd, n.value)
            if os.path.isfile(path):
                snap = os.path.join(workdir, "file%d" % len(out))
                shutil.copyfile(path, snap)
                out.append((n.value, snap))
    return out


def run_engines(viv_path, viv, wasm, workdir, engines, timeout, cwd=None, files=()):
    """Run the program in each engine: {engine: {"out", "err", "code", "ok", "note", "secs"}}.
    `ok` means it ran to its end or to a stop (not a timeout, not a refusal to build). The
    interpreter and the native program run in `cwd` (where Python ran the program, so a relative
    path names the same file); the wasm module reads `files`, attached by name."""
    res = {}
    base = os.path.join(workdir, "prog")
    viv_path = os.path.abspath(viv_path)

    def done(code, out, err, secs):
        return {"out": _clean(out), "err": _clean(err), "code": code, "ok": code != -9,
                "note": "" if code != -9 else err, "secs": secs}
    if "interp" in engines:
        # not --quiet: its filter (str::lines in lang/viv/src/main.rs) drops a "\r" that ends a
        # printed line; _clean takes the engine's own lines out instead
        res["interp"] = done(*run([viv, "run", viv_path], cwd=cwd, timeout=timeout))
    if "c" in engines:
        code, out, err, _ = run([viv, viv_path, "-o", base + ".c"], timeout=timeout)
        if code != 0:
            res["c"] = {"out": "", "err": "", "code": None, "ok": False, "note": "does not compile to C: " + (err or out).strip()[:400], "secs": 0}
        else:
            cc = os.environ.get("CC", "cc")
            code, out, err, _ = run([cc, "-O2", "-o", base, base + ".c", "-lm"], timeout=timeout)
            if code != 0:
                res["c"] = {"out": "", "err": "", "code": None, "ok": False, "note": "cc failed: " + err.strip()[:400], "secs": 0}
            else:
                res["c"] = done(*run([base], cwd=cwd, timeout=timeout))
    if "wasm" in engines:
        if not wasm:
            res["wasm"] = {"out": "", "err": "", "code": None, "ok": False, "note": "no viv.wasm runtime (set VIV_WASM)", "secs": 0, "skipped": True}
        else:
            code, out, err, _ = run([viv, viv_path, "-o", base + ".wasm"], timeout=timeout)
            if code != 0:
                res["wasm"] = {"out": "", "err": "", "code": None, "ok": False, "note": "does not compile to wasm: " + (err or out).strip()[:400], "secs": 0}
            else:
                env = dict(os.environ, VIV_WASM=wasm)
                res["wasm"] = done(*run(["node", os.path.join(ROOT, "tools", "run-wasm.mjs")]
                                        + ["--file=%s=%s" % (name, snap) for name, snap in files]
                                        + [base + ".wasm"], env=env, timeout=timeout))
    return res


def engines_agree(eng, engines):
    """The three engines agree: the same output and exit status; for a run that stops, the
    same output before the stop and the same error line (a wasm module reports the error line
    only, as it does for every run-time error)."""
    ran = [e for e in engines if eng.get(e, {}).get("ok")]
    expected = [e for e in engines if not eng.get(e, {}).get("skipped")]
    if not ran or len(ran) != len(expected):
        return False, {e: eng[e]["note"] for e in expected if e not in ran}
    ref = eng[ran[0]]
    diffs = {}
    for e in ran[1:]:
        r = eng[e]
        if r["code"] != ref["code"]:
            diffs[e] = {"exit": [ref["code"], r["code"]]}
            continue
        first = lambda t: (t.splitlines() or [""])[0]
        if ref["code"] == 0 or e != "wasm":
            if r["out"] != ref["out"]:
                diffs[e] = first_diff(ref["out"], r["out"])
            elif first(r["err"]) != first(ref["err"]):
                diffs[e] = {"stderr": [first(ref["err"]), first(r["err"])]}
        elif first(r["err"]) != first(ref["err"]):
            diffs[e] = {"stderr": [first(ref["err"]), first(r["err"])]}
    return not diffs, diffs


# ---------------------------------------------------------------- reading Viv's output

TEST_LINE = re.compile(r"^(PASS|FAIL) (test_\w+)(?: -- .*| \(no assertions\))?$")
VIV_ERR = re.compile(r"^error: (.*)$")
VIV_AT = re.compile(r"^\s*--> .*?:(\d+):\d+")


def parse_num(s):
    s = s.strip()
    if s in ("nan", "-nan"):
        return float("nan")
    if s == "inf":
        return float("inf")
    if s == "-inf":
        return float("-inf")
    return float(s)


def split_viv(out, err):
    """Viv's stdout -> (print lines, tests {name: PASS/FAIL/EMPTY}); stderr -> (the error
    message, its Viv line or None)."""
    lines, tests = [], {}
    for line in (out.split("\n")[:-1] if out.endswith("\n") else out.split("\n") if out else []):
        m = TEST_LINE.match(line)
        if m:
            tests[m.group(2)] = "EMPTY" if "(no assertions)" in line else m.group(1)
            continue
        lines.append(line)
    msg, at = None, None
    el = err.splitlines()
    for i, line in enumerate(el):
        m = VIV_ERR.match(line)
        if m:
            msg = m.group(1)
            for nxt in el[i + 1:i + 3]:
                ma = VIV_AT.match(nxt)
                if ma:
                    at = int(ma.group(1))
            break
    return lines, tests, msg, at


_NUMTOK = re.compile(r"\s*(?:-?(?:inf|nan)|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")


def _scan_value(text, pos, part):
    """The end of one printed value that starts at `pos`, and its parsed form."""
    kind = part["type"].get("kind")
    if kind in ("str", "strs"):
        want = part.get("want")
        if want is None or not text.startswith(want, pos):
            raise ValueError("not the text Python printed at %r" % text[pos:pos + 20])
        return pos + len(want), {"raw": want}
    if part.get("spec") is not None and part.get("want") is not None:
        if not text.startswith(part["want"], pos):
            raise ValueError("not the text Python formatted at %r" % text[pos:pos + 20])
        return pos + len(part["want"]), {"raw": part["want"]}
    if part.get("spec") is not None:
        m = _NUMTOK.match(text, pos)
        if not m:
            raise ValueError("no number at %r" % text[pos:pos + 20])
        return m.end(), {"raw": text[pos:m.end()]}
    if kind == "map":
        # a dict (maps.py): its text, compared as it is, as Python's repr of it
        if not text.startswith("{", pos):
            raise ValueError("no dict at %r" % text[pos:pos + 20])
        end = text.index("}", pos) + 1
        return end, {"raw": text[pos:end]}
    if kind == "bool":
        for word, v in (("True", 1.0), ("False", 0.0), ("true", 1.0), ("false", 0.0)):
            if text.startswith(word, pos):
                return pos + len(word), {"v": [v]}
        raise ValueError("no bool at %r" % text[pos:pos + 20])
    if kind == "num":
        m = _NUMTOK.match(text, pos)
        if not m:
            raise ValueError("no number at %r" % text[pos:pos + 20])
        if part["type"].get("py") == "int" and not re.fullmatch(r"-?\d+", text[pos:m.end()]):
            # an int prints digits only: a line of one def's clone for floats (py2viv's infer.py)
            # does not read back as the clone for ints
            raise ValueError("not an int at %r" % text[pos:pos + 20])
        if part["type"].get("py") == "int":
            # its digits too: an int past 2^53 (a Viv Int, py2viv's ints.py) is not a double
            return m.end(), {"v": [parse_num(text[pos:m.end()])], "int": text[pos:m.end()].strip()}
        return m.end(), {"v": [parse_num(text[pos:m.end()])]}
    if kind in ("cnum", "carr"):
        # "(" re "," im ")"
        if not text.startswith("(", pos):
            raise ValueError("no complex value at %r" % text[pos:pos + 20])
        depth, i = 0, pos
        while i < len(text):
            if text[i] in "([":
                depth += 1
            elif text[i] in ")]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = text[pos + 1:i]
        cut, depth = None, 0
        for j, ch in enumerate(inner):
            depth += ch == "["
            depth -= ch == "]"
            if ch == "," and depth == 0:
                cut = j
                break
        re_, im_ = _nested(inner[:cut]), _nested(inner[cut + 1:])
        flat_re, shape = _flatten(re_)
        flat_im, _ = _flatten(im_)
        return i + 1, {"re": flat_re, "im": flat_im, "shape": shape}
    # a real array: nested brackets
    if not text.startswith("[", pos):
        raise ValueError("no array at %r" % text[pos:pos + 20])
    depth, i = 0, pos
    while i < len(text):
        depth += text[i] == "["
        depth -= text[i] == "]"
        i += 1
        if depth == 0:
            break
    flat, shape = _flatten(_nested(text[pos:i]))
    return i, {"v": flat, "shape": shape}


def _nested(text):
    text = text.strip()
    js = re.sub(r"(?<![\w.])-?inf\b", lambda m: "-Infinity" if m.group(0).startswith("-") else "Infinity", text)
    js = re.sub(r"(?<![\w.])-?nan\b", "NaN", js)
    return json.loads(js)


def _flatten(v):
    if not isinstance(v, list):
        return [float(v)], []
    if not v:
        return [], [0]
    if isinstance(v[0], list):
        flat = []
        inner = None
        for row in v:
            f, sh = _flatten(row)
            flat.extend(f)
            inner = sh
        return flat, [len(v)] + inner
    return [float(x) for x in v], [len(v)]


def _slots(site):
    """The parts of a print site that stand for one value Python recorded, in order: its values
    (not those inside a printed object), and a string constant written where Python read a
    variable (a row of a loop over literal rows, unrolled)."""
    return [p for p in site["parts"] if p["kind"] == "val" and not p.get("sub") or p.get("subst")]


def _wants(site, want):
    """{index of a part in site["parts"]: the text Python printed for it}, for the string values."""
    out = {}
    if want is None:
        return out
    slots = _slots(site)
    if len(slots) != len(want):
        return out
    idx = {id(p): i for i, p in enumerate(site["parts"])}
    for p, w in zip(slots, want):
        kind = (p.get("type") or {}).get("kind")
        if p["kind"] == "val" and kind in ("str", "strs") and "t" in w:
            out[idx[id(p)]] = _rendered_text(p, w["t"], w.get("strs"))
        elif p["kind"] == "val" and kind == "strs" and w.get("list") and w.get("v") == []:
            out[idx[id(p)]] = "[]"  # an empty list: Python recorded it as numbers
        elif p["kind"] == "val" and kind == "num" and p.get("spec") and len(w.get("v") or []) == 1:
            try:
                # a number under a spec (a fill, an alignment): Python's format of the value it printed
                out[idx[id(p)]] = format(int(w["int"]) if "int" in w else w["v"][0], p["spec"])
            except (ValueError, TypeError):
                pass
    return out


def _rendered_text(part, text, is_list):
    """What Python printed for a string value it recorded as `text` (repr and a spec applied)."""
    if is_list:
        return text
    if part.get("render") == "repr":
        text = repr(text)
    spec = part.get("spec")
    if spec:
        text = format(text, spec)
    return text


def _types_agree(site, want):
    """Whether a print site's value types are those of the values Python printed (a list or a
    NumPy array, an int, a float or a Bool)."""
    if want is None:
        return True
    parts = _slots(site)
    if len(parts) != len(want):
        return True  # objects and text values: not told apart here
    for p, w in zip(parts, want):
        if p["kind"] == "str":
            if "s" not in w:
                return False
            continue
        t = p.get("type") or {}
        k = t.get("kind")
        if k in ("str", "strs"):
            if "s" not in w or (k == "strs") != ("strs" in w):
                return False
            continue
        if k == "num" and (("int" in w) != (t.get("py") == "int") or "bool" in w):
            return False
        if k == "bool" and "bool" not in w:
            return False
        if k in ("arr", "carr"):
            if ("list" in w) != (t.get("container") in ("list", "shape")):
                return False
            dt = w.get("dt")
            if dt is not None and {"i": "int", "u": "int", "f": "float", "b": "bool"}.get(dt, t.get("dtype")) != t.get("dtype") \
                    and t.get("kind") == "arr":
                return False
    return True


def read_calls(print_lines, pprints, psites, sites):
    """Viv's print lines as the print calls Python made, in order: each call takes as many
    lines as its site's text has newlines, and each value is read back from its place in the
    line. Returns (events, trouble): an event is {"site", "vals": {k: parsed}}."""
    by_pos = {}
    for s in sites:
        by_pos.setdefault((s["line"], s.get("col")), []).append(s)
    events, pos, trouble = [], 0, None

    def read(site, text, wants):
        """The values of one call's line; a string value's text is the one Python printed (wants,
        by part), and one inside a printed object is found by trying each end in turn."""
        parts = site["parts"]

        def go(i, at, vals):
            while i < len(parts):
                part = parts[i]
                if part["kind"] == "str":
                    if not text.startswith(part["text"], at):
                        return None
                    at += len(part["text"])
                    i += 1
                    continue
                kind = part["type"].get("kind")
                if kind in ("str", "strs") and part.get("sub"):
                    # a string field of an object (compared as the line's text): every end
                    for end in range(at, len(text) + 1):
                        v2 = dict(vals)
                        v2[part["k"]] = {"raw": text[at:end]}
                        r = go(i + 1, end, v2)
                        if r is not None:
                            return r
                    return None
                try:
                    at, vals[part["k"]] = _scan_value(text, at, dict(part, want=wants.get(i)))
                except (ValueError, json.JSONDecodeError):
                    return None
                i += 1
            return vals if at == len(text) else None
        vals = go(0, 0, {})
        return {"site": site["site"], "vals": vals or {}, "text": text, "ok": vals is not None}
    for i, ps in enumerate(psites):
        cands = by_pos.get(tuple(ps), []) if ps else []
        if not cands:
            trouble = "Python printed from a place the translation has no print for (line %s)" % (ps,)
            break
        # one Python print can be several Viv prints (a loop unrolled per row or per key, each
        # with its own types): the first whose parts read the line back
        ev = None
        want = pprints[i] if i < len(pprints) else None
        for site in cands:
            wants = _wants(site, want)
            nl = sum(p["text"].count("\n") for p in site["parts"] if p["kind"] == "str") + \
                sum(w.count("\n") for w in wants.values())
            if pos + nl > len(print_lines):
                continue
            e = read(site, "\n".join(print_lines[pos:pos + nl]) + "\n", wants)
            e["nl"] = nl
            if ev is None or e["ok"] and not ev["ok"]:
                ev = e
            if e["ok"] and (len(cands) == 1 or _types_agree(site, want)):
                # one def translated once per argument types (py2viv's infer.py) prints from one
                # Python print as several Viv prints whose text can be alike (a list and an
                # array both print as a list): the one of the values' Python types
                ev = e
                break
        if ev is None:
            trouble = "Viv printed %d lines, fewer than Python's calls need" % len(print_lines)
            break
        pos += ev.pop("nl")
        events.append(ev)
    if trouble is None and pos != len(print_lines):
        trouble = "Viv printed %d lines, Python's calls account for %d" % (len(print_lines), pos)
    return events, trouble


# ---------------------------------------------------------------- printing Viv values as Python would


def _pyscalar(v, ty):
    kind = ty.get("kind")
    if kind == "bool":
        return v != 0
    if kind == "num" and ty.get("py") == "int" and math.isfinite(v) and v == int(v):
        return int(v)
    return float(v)


def _array(vals, ty, shape):
    dtype = ty.get("dtype")
    if ty.get("kind") == "carr":
        re_, im_ = vals
        data = [complex(a, b) for a, b in zip(re_, im_)]
    else:
        data = vals
    cont = ty.get("container")
    if cont == "shape":
        return tuple(int(x) for x in data)
    if cont == "list":
        def conv(x):
            if dtype == "bool":
                return x != 0
            if dtype == "int" and math.isfinite(x) and x == int(x):
                return int(x)
            return x if isinstance(x, complex) else float(x)
        flat = [conv(x) for x in data]
        if shape and len(shape) == 2:
            r, c = int(shape[0]), int(shape[1])
            return [flat[i * c:(i + 1) * c] for i in range(r)]
        return flat
    if np is None:
        return data
    npdt = {"int": np.int64, "bool": np.bool_}.get(dtype, np.float64)
    if ty.get("kind") == "carr":
        npdt = np.complex128
    a = np.array(data, dtype=npdt if dtype != "int" or all(math.isfinite(x) and x == int(x) for x in data) else np.float64)
    if shape:
        a = a.reshape([int(x) for x in shape])
    return a


def render_value(part, event):
    ty = part["type"]
    k = part["k"]
    val = event["vals"].get(k, {})
    if "raw" in val:
        return val["raw"]  # a format spec of fmt's table: Python's text already
    kind = ty.get("kind")
    if kind == "num" and ty.get("py") == "int" and "int" in val:
        py = int(val["int"])
    elif kind in ("num", "bool"):
        v = (val.get("v") or [float("nan")])[0]
        py = _pyscalar(v, ty)
    elif kind == "cnum":
        py = complex((val.get("re") or [0.0])[0], (val.get("im") or [0.0])[0])
    elif kind == "carr":
        py = _array([val.get("re", []), val.get("im", [])], ty, val.get("shape"))
    else:
        py = _array(val.get("v", []), ty, val.get("shape"))
    if part.get("render") == "tolist" and np is not None and isinstance(py, np.ndarray):
        py = py.tolist()
    spec = part.get("spec")
    if spec is not None:
        return format(py, spec)
    if part.get("render") == "repr":
        return repr(py)
    return str(py)


def render_events(events, sites):
    by_site = {s["site"]: s for s in sites}
    out = []
    for ev in events:
        site = by_site[ev["site"]]
        if not ev["ok"]:
            out.append(ev["text"])  # a line that does not read back: compared as it is
            continue
        text = []
        for part in site["parts"]:
            if part["kind"] == "str":
                text.append(part["text"])
            else:
                try:
                    text.append(render_value(part, ev))
                except Exception as e:  # noqa: BLE001
                    text.append("<unrenderable: %s>" % e)
        out.append("".join(text))
    return "".join(out)


# ---------------------------------------------------------------- comparing

NUM = re.compile(r"(?<![\w.])(?:[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|[-+]?inf|nan|True|False)(?![\w])")


def numbers(line):
    out = []
    for t in NUM.findall(line):
        if t == "True":
            out.append((t, 1.0))
        elif t == "False":
            out.append((t, 0.0))
        else:
            out.append((t, parse_num(t)))
    return out


def within(v, ref, rtol, atol):
    if math.isnan(v) or math.isnan(ref):
        return math.isnan(v) and math.isnan(ref)
    if math.isinf(v) or math.isinf(ref):
        return v == ref
    return abs(v - ref) <= atol + rtol * abs(ref)


def compare_text(py_text, viv_text, rtol, atol):
    pl = py_text.splitlines()
    vl = viv_text.splitlines()
    rows = []
    worst = "identical"
    order = {"identical": 0, "within": 1, "MISMATCH": 2}
    for i in range(max(len(pl), len(vl))):
        a = pl[i] if i < len(pl) else None
        b = vl[i] if i < len(vl) else None
        row = {"n": i + 1, "python": a, "viv": b, "max_dev": 0.0, "max_rel": 0.0, "count": 0}
        if a == b:
            row["verdict"] = "identical"
            row["count"] = len(numbers(a or ""))
        elif a is None or b is None:
            row["verdict"] = "MISMATCH"
            row["why"] = "missing line in " + ("python" if a is None else "viv")
        else:
            na, nb = numbers(a), numbers(b)
            skel_a = NUM.sub("#", a)
            skel_b = NUM.sub("#", b)
            if len(na) != len(nb):
                row["verdict"] = "MISMATCH"
                row["why"] = "different count of numbers (%d vs %d)" % (len(na), len(nb))
            else:
                ok = True
                same = True
                for (ta, va), (tb, vb) in zip(na, nb):
                    d = abs(va - vb) if math.isfinite(va) and math.isfinite(vb) else (0.0 if (va == vb or (math.isnan(va) and math.isnan(vb))) else float("inf"))
                    row["max_dev"] = max(row["max_dev"], d)
                    if math.isfinite(va) and va != 0:
                        row["max_rel"] = max(row["max_rel"], d / abs(va))
                    if not within(vb, va, rtol, atol):
                        ok = False
                    if not (va == vb and math.copysign(1.0, va) == math.copysign(1.0, vb) or va != va and vb != vb):
                        same = False
                row["count"] = len(na)
                row["verdict"] = "within" if ok else "MISMATCH"
                if ok and (same or skel_a.replace(" ", "") != skel_b.replace(" ", "")):
                    # the numbers are the same (or within the declared tolerance) and the text is not:
                    # what the program printed is not Python's (RULES.md, Fixed holes: diff-text)
                    row["verdict"] = "MISMATCH"
                    row["why"] = "the text differs from Python's" + (" (the numbers are the same)" if same else "")
        rows.append(row)
        if order[row["verdict"]] > order[worst]:
            worst = row["verdict"]
    return rows, worst


# ---------------------------------------------------------------- the report


def tolerances(src, rtol, atol):
    """The tolerance the program states (`# py2viv: rtol=1e-12 atol=0`), or the command
    line's; with neither, every number is compared bit for bit."""
    declared = rtol is not None or atol is not None
    m = re.search(r"#\s*py2viv:\s*(.*)", src)
    if m:
        for kv in m.group(1).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k == "rtol" and rtol is None:
                    rtol = float(v)
                    declared = True
                if k == "atol" and atol is None:
                    atol = float(v)
                    declared = True
    return (0.0 if rtol is None else rtol), (0.0 if atol is None else atol), declared


def differential(py_path, out_dir=None, rtol=None, atol=None, engines=("interp", "c", "wasm"),
                 python=None, timeout=900, viv=None, wasm=None, run_python=True):
    src = open(py_path).read()
    rtol, atol, declared = tolerances(src, rtol, atol)
    name = os.path.splitext(os.path.basename(py_path))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(py_path)), name + ".py2viv")
    os.makedirs(out_dir, exist_ok=True)
    viv = viv or default_viv()
    wasm = wasm if wasm is not None else default_wasm()
    rep = {"program": os.path.basename(py_path), "translator": py2viv.VERSION, "rtol": rtol, "atol": atol,
           "tolerance_declared": declared, "engines": {}, "python": {}, "lines": [], "tests": {}, "verdict": None}
    res = py2viv.translate(src, py_path)
    man = py2viv.manifest(res, py_path, src, "exact")
    rep["inexact"] = man["inexact"]
    with open(os.path.join(out_dir, name + ".py2viv.json"), "w") as f:
        json.dump(man, f, indent=1)
    if res["refusals"]:
        rep["verdict"] = "REFUSED"
        rep["refusals"] = man["refusals"]
        rep["messages"] = res["messages"]
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        return rep
    viv_path = os.path.join(out_dir, name + ".viv")
    with open(viv_path, "w") as f:
        f.write(res["viv"])
    pres = {"tests": {}, "prints": [], "print_sites": []}
    pout = ""
    prog_dir = os.path.dirname(os.path.abspath(py_path))
    files_dir = tempfile.mkdtemp(prefix="py2viv_files_")
    files = data_files(src, prog_dir, files_dir)
    if run_python:
        results = os.path.join(out_dir, "python_tests.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(HERE, "pyshim") + os.pathsep + env.get("PYTHONPATH", "")
        py = python or sys.executable
        inst = os.path.join(out_dir, name + ".instrumented.py")
        with open(inst, "w") as f:
            f.write(instrument(src))
        code, pout, perr, psecs = run([py, "-c", RUNNER, os.path.abspath(inst), results],
                                      cwd=os.path.dirname(os.path.abspath(py_path)), env=env, timeout=timeout)
        pres = json.load(open(results)) if os.path.exists(results) else {"tests": {}, "error": perr[-2000:], "prints": [], "print_sites": []}
        rep["python"] = {"secs": round(psecs, 3), "error": pres.get("error"), "exc_line": pres.get("exc_line"),
                         "tests": pres.get("tests", {}),
                         "version": run([py, "-c", "import sys\ntry:\n import numpy; v = numpy.__version__\nexcept ImportError:\n v = 'no numpy'\nprint(sys.version.split()[0], v)"])[1].strip()}
        with open(os.path.join(out_dir, "python.out"), "w") as f:
            f.write(pout)
    with tempfile.TemporaryDirectory() as work:
        eng = run_engines(viv_path, viv, wasm, work, engines, timeout, cwd=prog_dir, files=files)
    shutil.rmtree(files_dir, ignore_errors=True)
    for e in engines:
        r = eng[e]
        with open(os.path.join(out_dir, e + ".out"), "w") as f:
            f.write(r["out"] + ("\n--- stderr ---\n" + r["err"] if r.get("err") else ""))
        rep["engines"][e] = {"ran": r["ok"], "exit": r.get("code"), "note": r["note"], "secs": round(r["secs"], 3), "skipped": r.get("skipped", False)}
    agree, diffs = engines_agree(eng, engines)
    rep["engines_agree"] = agree
    if diffs:
        rep["engine_diffs"] = diffs
    ref = next((e for e in engines if eng.get(e, {}).get("ok") and e != "wasm"), None) or \
        next((e for e in engines if eng.get(e, {}).get("ok")), None)
    if ref is None:
        rep["verdict"] = "VIV FAILED"
        return finish(rep, out_dir)
    lines, vtests, vmsg, vat = split_viv(eng[ref]["out"], eng[ref]["err"])
    rep["raw_identical"] = ("\n".join(lines) + "\n" if lines else "") == pout
    events, trouble = read_calls(lines, pres.get("prints", []), pres.get("print_sites", []), res["sites"])
    rep["read_back"] = trouble
    viv_text = render_events(events, res["sites"])
    with open(os.path.join(out_dir, "viv_as_python.out"), "w") as f:
        f.write(viv_text)
    rows, worst = compare_text(pout, viv_text, rtol, atol)
    rep["lines"] = rows
    rep["text_verdict"] = worst
    vals, vworst = compare_values(pres.get("prints", []), events, res["sites"], rtol, atol)
    rep["values"] = vals
    rep["values_verdict"] = vworst
    if trouble:
        vworst = "MISMATCH"
    if worst == "MISMATCH" and vworst == "within" and declared and vals and all(v["exact"] for v in vals):
        # values that differ within the declared tolerance change how NumPy lays an array out
        # (its precision, its line breaks), which is text Python chose; values that are the same
        # bit for bit must print the same text (RULES.md, Fixed holes: diff-text)
        worst = "within"
    # the end: both finished, or both stopped with the same exception
    exc = pres.get("exc_line")
    stop_ok = True
    if exc or vmsg:
        rep["stop"] = {"python": exc, "viv": vmsg, "viv_line": vat}
        if not (exc and vmsg):
            stop_ok = False
        elif exc != vmsg:
            # a computed message is kept as its type and literal parts: compare the type
            pyline = res["linemap"][vat - 1] if vat and vat - 1 < len(res["linemap"]) else None
            inexact_site = any(st["line"] == pyline and not st["exact"] for st in res.get("stops", []))
            # a helper's stop that states the start of NumPy's message (NumPy's rest is
            # computed: the shapes, the index): Python's line must start with it
            prefix_site = pyline == 0 and any(st.get("prefix") and st["message"] == vmsg for st in res.get("stops", []))
            if prefix_site:
                stop_ok = exc.startswith(vmsg)
                rep["stop"]["compared"] = "the start of Python's line (a helper states NumPy's message up to its computed part)"
            else:
                stop_ok = inexact_site and exc.split(":")[0] == vmsg.split(":")[0]
                rep["stop"]["compared"] = "type only (the message is computed)" if inexact_site else "message"
    if not stop_ok and exc and vmsg and python_line_of(vmsg) == exc:
        # a Viv builtin stopped with CPython's own message, which Python's line prefixes with the
        # exception's type (VIV_ERRORS)
        stop_ok = True
        rep["stop"]["compared"] = "Viv's message as CPython's exception line (%s)" % exc.split(":")[0]
    rep["stop_ok"] = stop_ok
    ptests = pres.get("tests", {})
    for t in sorted(set(ptests) | set(vtests)):
        rep["tests"][t] = {"python": ptests.get(t, "-"), "viv": vtests.get(t, "-")}
    tests_ok = all(v["python"] == v["viv"] for v in rep["tests"].values())
    ok = agree and worst != "MISMATCH" and vworst != "MISMATCH" and tests_ok and stop_ok
    if ok:
        exact = vworst == "bitwise" and worst in ("identical", "within")
        rep["verdict"] = "PASS (%s)" % ("bitwise" if exact else "within the declared tolerance")
    else:
        why = []
        if not agree:
            why.append("engines differ or did not all run")
        if worst == "MISMATCH" or vworst == "MISMATCH":
            why.append("values mismatch" + ("" if declared or not rep["inexact"] else " (inexact operations used and no tolerance declared)"))
        if not tests_ok:
            why.append("tests differ")
        if not stop_ok:
            why.append("the run ends differently (%s / %s)" % (exc or "Python finished", vmsg or "Viv finished"))
        rep["verdict"] = "FAIL (" + "; ".join(why) + ")"
    return finish(rep, out_dir)


# Viv's string builtins stop with CPython's messages; the traceback's last line
# adds the exception's type. Each message, and the line CPython writes for it.
VIV_ERRORS = [
    (r"could not convert string to float: .*", "ValueError: {m}"),
    (r"invalid literal for int\(\) with base \d+: .*", "ValueError: {m}"),
    (r"int\(\) base must be >= 2 and <= 36, or 0", "ValueError: {m}"),
    (r"Exceeds the limit \(\d+ digits\) for integer string conversion.*", "ValueError: {m}"),
    (r"string index out of range", "IndexError: {m}"),
    (r"list index out of range", "IndexError: {m}"),
    (r"ord\(\) expected a character, but string of length \d+ found", "TypeError: {m}"),
    (r"chr\(\) arg not in range\(0x110000\)", "ValueError: {m}"),
    (r"substring not found", "ValueError: {m}"),
    (r"empty separator", "ValueError: {m}"),
    (r"slice step cannot be zero", "ValueError: {m}"),
    (r"The fill character must be exactly one character long", "TypeError: {m}"),
    (r"No such file or directory: ('.*')", "FileNotFoundError: [Errno 2] No such file or directory: {1}"),
]


def python_line_of(vmsg):
    """The exception line CPython writes where a Viv string builtin stopped with vmsg, or None."""
    for pat, line in VIV_ERRORS:
        m = re.fullmatch(pat, vmsg)
        if m:
            return line.replace("{m}", vmsg).replace("{1}", m.group(1) if m.groups() else "")
    return None


def _vals_of(rec):
    if "re" in rec:
        return rec["re"] + rec["im"]
    return rec.get("v", [])


def compare_values(pprints, events, sites, rtol, atol):
    """Exact values: Python's print arguments (recorded at full precision) against the values
    read back from Viv's lines, call by call. One row per print site. A value printed with a
    format spec is compared as text: Python's format of its exact value against Viv's."""
    by_site = {s["site"]: s for s in sites}
    rows = {}
    order = {"bitwise": 0, "within": 1, "MISMATCH": 2}
    worst = "bitwise"
    for call, ev in zip(pprints, events):
        site = by_site[ev["site"]]
        # a value inside a printed object (a field of a dataclass, a dict's value) is compared
        # as the line's text: its repr, which reads back to the same double
        # a string (a value, or a constant written where Python read a variable) is compared as
        # the line's text: reading the line back checked it against the text Python printed
        parts = _slots(site)
        exact = len(call) == len(parts) and ev["ok"]
        for k, part in enumerate(parts):
            if exact and (part["kind"] == "str" or (part.get("type") or {}).get("kind") in ("str", "strs")):
                continue
            key = (ev["site"], k)
            row = rows.get(key)
            if row is None:
                row = rows[key] = {"line": site["line"], "expr": part.get("expr", ""), "calls": 0, "numbers": 0,
                                   "python_first": None, "viv_first": None, "max_dev": 0.0, "max_rel": 0.0,
                                   "verdict": "bitwise", "exact": exact}
            row["calls"] += 1
            if not exact:
                row["exact"] = False
                row["verdict"] = "MISMATCH"
                row["why"] = "the Viv line does not read back as this print's parts"
                worst = "MISMATCH"
                continue
            got = ev["vals"].get(part["k"], {})
            pv = _vals_of(call[k])
            if "raw" in got:
                want = None
                try:
                    want = format(_pyscalar(pv[0], part["type"]) if len(pv) == 1 else pv[0], part["spec"])
                except Exception:  # noqa: BLE001
                    pass
                row["numbers"] += 1
                if row["python_first"] is None:
                    row["python_first"], row["viv_first"] = want, got["raw"]
                if want != got["raw"]:
                    row["verdict"] = "MISMATCH"
                    row["why"] = "formatted %r in Python, %r in Viv" % (want, got["raw"])
                    worst = "MISMATCH"
                continue
            vv = (got.get("re", []) + got.get("im", [])) if "re" in got else got.get("v", [])
            if row["python_first"] is None:
                row["python_first"] = pv[:3]
                row["viv_first"] = vv[:3]
            if len(pv) != len(vv):
                row["verdict"] = "MISMATCH"
                row["why"] = "%d values in Python, %d in Viv" % (len(pv), len(vv))
                worst = "MISMATCH"
                continue
            row["numbers"] += len(pv)
            for x, y in zip(pv, vv):
                same = (x == y and math.copysign(1.0, x) == math.copysign(1.0, y)) or (x != x and y != y)
                if same:
                    continue
                d = abs(x - y) if math.isfinite(x) and math.isfinite(y) else float("inf")
                row["max_dev"] = max(row["max_dev"], d)
                if x != 0 and math.isfinite(x):
                    row["max_rel"] = max(row["max_rel"], d / abs(x))
                verdict = "within" if (within(y, x, rtol, atol) and (rtol or atol)) else "MISMATCH"
                if order[verdict] > order.get(row["verdict"], 0):
                    row["verdict"] = verdict
            if order.get(row["verdict"], 0) > order[worst]:
                worst = row["verdict"]
    if len(pprints) != len(events):
        worst = "MISMATCH"
        rows[(-1, 0)] = {"line": 0, "expr": "<print count>", "calls": 0, "numbers": 0, "python_first": len(pprints),
                         "viv_first": len(events), "max_dev": 0.0, "max_rel": 0.0, "verdict": "MISMATCH", "exact": True,
                         "why": "Python printed %d times, Viv %d" % (len(pprints), len(events))}
    return sorted(rows.values(), key=lambda r: r["line"]), worst


def first_diff(a, b):
    al, bl = a.splitlines(), b.splitlines()
    for i in range(max(len(al), len(bl))):
        x = al[i] if i < len(al) else "<end>"
        y = bl[i] if i < len(bl) else "<end>"
        if x != y:
            return {"line": i + 1, "a": x[:200], "b": y[:200]}
    return None


def finish(rep, out_dir):
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(format_report(rep))
    return rep


def fmt_g(x):
    if x == 0:
        return "0"
    return "%.2g" % x


def format_report(rep, max_rows=40):
    o = []
    o.append("py2viv %s differential report: %s" % (rep["translator"], rep["program"]))
    if rep["verdict"] == "REFUSED":
        o.append("REFUSED: %d construct(s) outside the numeric subset" % len(rep["refusals"]))
        o.extend(rep["messages"])
        return "\n".join(o) + "\n"
    py = rep.get("python", {})
    o.append("reference: CPython+NumPy %s, %.2f s%s" % (py.get("version", "?"), py.get("secs", 0),
                                                        "; RAISED:\n" + py["error"] if py.get("error") else ""))
    eng = []
    for e, r in rep["engines"].items():
        eng.append("%s %s" % (e, ("%.2f s" % r["secs"]) if r["ran"] else ("NOT RUN: " + r["note"])))
    o.append("viv engines: " + "; ".join(eng))
    o.append("engines agree byte for byte: %s" % ("yes" if rep.get("engines_agree") else "NO " + json.dumps(rep.get("engine_diffs", {}))))
    if rep.get("tolerance_declared"):
        o.append("tolerance (declared): |viv - python| <= %g + %g |python| per number" % (rep["atol"], rep["rtol"]))
    else:
        o.append("tolerance: none declared, every number compared bit for bit")
    for x in rep.get("inexact", []):
        o.append("  inexact operation (line %d): %s" % (x["line"], x["what"]))
    if rep.get("raw_identical") is not None:
        o.append("Viv's printed text is CPython's byte for byte: %s" % ("yes" if rep["raw_identical"] else "no (read back and compared by value)"))
    if rep.get("read_back"):
        o.append("reading Viv's lines back: " + rep["read_back"])
    if rep.get("stop"):
        o.append("the run ends: python %r, viv %r%s" % (rep["stop"]["python"], rep["stop"]["viv"], " (%s)" % rep["stop"]["compared"] if rep["stop"].get("compared") else ""))
    o.append("")
    o.append("values, exact (every print argument, Python's recorded at full precision; Viv's identical in every engine):")
    o.append("%-5s %-26s %6s %-24s %-24s %-9s %-9s %s" % ("line", "value", "n", "numpy", "viv", "max|dev|", "max rel", "verdict"))
    for r in rep.get("values", []):
        pf = r["python_first"]
        vf = r["viv_first"]
        def show(x):
            if isinstance(x, list):
                s = ", ".join(repr(v) for v in x[:2])
                return s if len(x) <= 1 else "[" + s + ", ...]"
            return str(x)
        o.append("%-5d %-26s %6d %-24s %-24s %-9s %-9s %s" % (r["line"], r["expr"][:26], r["numbers"], show(pf)[:24], show(vf)[:24],
                                                     fmt_g(r["max_dev"]), fmt_g(r["max_rel"]), r["verdict"] + ("  (" + r["why"] + ")" if r.get("why") else "")))
    o.append("")
    o.append("printed text (Viv's values formatted by Python's own str/repr/format and NumPy's printing):")
    o.append("%-5s %-9s %-10s %-9s %s" % ("line", "verdict", "max|dev|", "max rel", "python  |  viv (as Python prints it)"))
    rows = rep.get("lines", [])
    shown = 0
    for r in rows:
        if shown >= max_rows and r["verdict"] == "identical":
            continue
        shown += 1
        py_l = (r.get("python") or "<none>")
        vv = (r.get("viv") or "<none>")
        text = py_l if r["verdict"] == "identical" else "%s  |  %s" % (py_l, vv)
        o.append("%-5d %-9s %-10s %-9s %s%s" % (r["n"], r["verdict"], fmt_g(r["max_dev"]), fmt_g(r["max_rel"]),
                                               text[:150], ("  (" + r.get("why", r.get("note", "")) + ")") if (r.get("why") or r.get("note")) else ""))
    if len(rows) > shown:
        o.append("... %d more identical lines" % (len(rows) - shown))
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    o.append("lines: " + ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items())) + "; numbers compared: %d" % sum(r["count"] for r in rows))
    if rep.get("tests"):
        o.append("")
        o.append("tests:")
        for t, v in rep["tests"].items():
            o.append("  %-40s python %-6s viv %s" % (t, v["python"], v["viv"]))
    if rep.get("asserts"):
        o.append("top-level asserts: %d, failed %d" % (len(rep["asserts"]), sum(" FAIL" in a for a in rep["asserts"])))
    for e in rep.get("viv_errors", []):
        o.append("viv error:\n" + e)
    o.append("")
    o.append("verdict: " + rep["verdict"])
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPython+NumPy against Viv's three engines.")
    ap.add_argument("file")
    ap.add_argument("--out")
    ap.add_argument("--rtol", type=float)
    ap.add_argument("--atol", type=float)
    ap.add_argument("--engines", default="interp,c,wasm")
    ap.add_argument("--python", help="the Python that runs the reference (default: this one)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = differential(args.file, args.out, args.rtol, args.atol, tuple(args.engines.split(",")),
                       args.python, args.timeout)
    if args.json:
        json.dump(rep, sys.stdout, indent=1)
        print()
    else:
        sys.stdout.write(format_report(rep))
    return 0 if rep["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())

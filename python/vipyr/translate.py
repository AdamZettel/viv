# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Python functions as compiled Viv: `vp.translate` and `@vp.jit`.

    import vipyr as vp

    def ewma(x, a):
        y = np.zeros(len(x))
        s = x[0]
        for i in range(len(x)):
            s = a * x[i] + (1.0 - a) * s
            y[i] = s
        return y

    fast = vp.translate(ewma)          # same signature, compiled
    fast(x, 0.1)                       # first call: checked against ewma itself
    print(fast.viv_source)             # the Viv it runs (or translate(..., show=True))

    @vp.jit                            # the same, as a decorator
    def count_above(x, t): ...

The translation is tools/py2viv, the strict translator: a function is
translated so that it computes exactly what CPython and NumPy compute, or it
is refused (VivTranslationError) with the line, the construct and what to
write instead. There is no approximate mode.

Types. A Viv def is compiled for one set of argument types. Parameters
annotated in Python (`x: np.ndarray`, `n: int`, `a: float`) are taken as
written, and when every parameter is annotated the translation happens at
`translate()` time. Otherwise the types come from the first call's
arguments: a float, an int, a bool, or an array (NumPy, a pandas or polars
Series, a list): a float64 array as it is, an int or bool array as float64
(exact below 2**53; larger values are refused), since py2viv's arrays are
doubles. A later call with other argument types translates again (one
compiled version per signature, cached on disk by the source's hash).

What it reads. The function's own source (inspect.getsource), the plain
Python functions it calls from its module (translated with it), the int,
float and bool constants it reads from its module, and `numpy` and `math`
under whatever names the module imported them. Anything else it names is
refused by py2viv with a location.

The check. On the first call of each signature (`check=True`, the default)
the original Python function also runs, on copies of the same arguments, and
the results must be equal: bit for bit, unless py2viv lists an operation
whose last bits come from a different library (libm, BLAS: the manifest's
`inexact`), where the stated tolerance `rtol` (default 1e-12, relative,
elementwise) applies. A difference raises VivTranslationMismatch with both
values; it is never silent. `check=False` skips the Python run.

Results. Numbers come back as Python floats or ints, arrays as float64
ndarrays; after a checked call an array result takes the dtype the Python
function returned (an int array comes back as int64 when every value is
integral, which the check has just shown).

Without a C compiler the translated Viv runs in Viv's interpreter
(vipyr/_interp.py), with the same results.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import math
import os
import re
import sys
import textwrap
import types
from pathlib import Path

import numpy as np

from . import _core

__all__ = ["translate", "jit", "VivTranslationError", "VivTranslationMismatch", "Translated"]

_INT_LIMIT = 2 ** 53


class VivTranslationError(TypeError):
    """py2viv refused the function: `.refusals` holds (file, line, col,
    what, hint) for each construct outside the numeric subset."""

    def __init__(self, msg, refusals=()):
        super().__init__(msg)
        self.refusals = list(refusals)


class VivTranslationMismatch(AssertionError):
    """The translated function and the Python one gave different results on
    the same arguments."""


def _py2viv():
    """The translator module: the wheel's copy (vipyr/_py2viv), or the
    checkout's tools/py2viv."""
    if "py2viv" in sys.modules:
        return sys.modules["py2viv"]
    here = Path(__file__).resolve().parent
    cands = [here / "_py2viv"]
    repo = _core._repo_root()
    if repo is not None:
        cands.insert(0, repo / "tools" / "py2viv")
    for d in cands:
        if (d / "py2viv.py").is_file():
            if d.parent.name == "vipyr" and _core.viv_root() and not os.environ.get("VIV_ROOT"):
                os.environ["VIV_ROOT"] = _core.viv_root()      # the helpers' lib/ modules, for the wheel's copy
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import py2viv
            return py2viv
    raise ImportError("vipyr: the py2viv translator is not found (tools/py2viv in a checkout, vipyr/_py2viv in a wheel)")


# ------------------------------------------------------------- the source

def _func_source(fn):
    try:
        lines, start = inspect.getsourcelines(fn)
    except (OSError, TypeError) as e:
        raise VivTranslationError("vp.translate: the source of %s is not available (%s); define it in a file, "
                                  "or a notebook cell" % (getattr(fn, "__name__", fn), e))
    text = textwrap.dedent("".join(lines))
    tree = ast.parse(text)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef,)):
        raise VivTranslationError("vp.translate takes a function defined with `def`; got %s" % type(node).__name__)
    # drop @vp.jit (and @vipyr.jit, @jit) itself; other decorators stay, and py2viv refuses them
    keep = []
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name not in ("jit", "translate"):
            keep.append(d)
    first = node.lineno if not node.decorator_list else min(d.lineno for d in node.decorator_list)
    body = text.splitlines()
    def_line = node.lineno
    if not keep and node.decorator_list:
        body = body[def_line - 1:]
        offset = start + def_line - 2
    else:
        body = body[first - 1:]
        offset = start + first - 2
    return "\n".join(body) + "\n", offset, node


def _names_read(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _collect(fn):
    """The module text py2viv translates, and a line map back to the files:
    the imports and constants the function reads, the functions it calls
    (in dependency order), then the function."""
    chunks, seen, consts, imports = [], set(), {}, {}

    def visit(f):
        if f.__name__ in seen:
            return
        seen.add(f.__name__)
        src, offset, node = _func_source(f)
        g = f.__globals__
        for name in sorted(_names_read(node)):
            if name in g:
                v = g[name]
                if isinstance(v, types.ModuleType):
                    if v.__name__ in ("numpy", "math"):
                        imports[name] = v.__name__
                elif isinstance(v, types.FunctionType) and v is not f and v.__name__ == name \
                        and not v.__module__.startswith(("numpy", "vipyr")):
                    visit(v)
                elif isinstance(v, (bool, int, float, np.integer, np.floating)) and not isinstance(v, type):
                    consts[name] = v
        chunks.append((src, inspect.getsourcefile(f) or "<python>", offset))

    visit(fn)
    head = []
    for name, mod in sorted(imports.items()):
        head.append("import %s" % mod if name == mod else "import %s as %s" % (mod, name))
    if "np" not in imports and "numpy" not in imports.values():
        head.append("import numpy as np")
    for name, v in sorted(consts.items()):
        v = v.item() if isinstance(v, np.generic) else v
        if isinstance(v, float) and not math.isfinite(v):
            raise VivTranslationError("vp.translate: the constant %s = %r is not finite; pass it as an argument" % (name, v))
        head.append("%s = %r" % (name, v))
    text, lmap = "", []
    for line in head:
        text += line + "\n"
        lmap.append(None)
    for src, file, offset in chunks:
        text += "\n"
        lmap.append(None)
        for k, line in enumerate(src.splitlines()):
            text += line + "\n"
            lmap.append((file, offset + k + 1))
    return text, lmap


# ------------------------------------------------------------------ types

def _column(v):
    """A pandas or polars Series (or Index) as a NumPy array, else None."""
    mod = type(v).__module__ or ""
    if mod.startswith(("pandas", "polars")) and hasattr(v, "to_numpy"):
        return np.asarray(v.to_numpy())
    return None


def _kind(v, name):
    """(py2viv annotation, the value as passed to Viv) for one argument."""
    if isinstance(v, (bool, np.bool_)):
        return "bool", bool(v)
    if isinstance(v, (int, np.integer)):
        return "int", int(v)
    if isinstance(v, (float, np.floating)):
        return "float", float(v)
    col = _column(v)
    a = col if col is not None else v
    if isinstance(a, (list, tuple)):
        a = np.asarray(a)
    if isinstance(a, np.ndarray):
        if a.dtype == np.float64:
            return "np.ndarray", a
        if a.dtype.kind in "iub":
            if a.size and (int(np.max(np.abs(a.astype(np.int64)))) >= _INT_LIMIT):
                raise VivTranslationError("vp.translate: %s holds integers of 2**53 or more, which py2viv's arrays "
                                          "(doubles) cannot hold exactly" % name)
            return "np.ndarray", a.astype(np.float64)
        if a.dtype.kind == "f":
            return "np.ndarray", a.astype(np.float64)
        raise VivTranslationError("vp.translate: %s is an array of %s; py2viv translates float and int arrays"
                                  % (name, a.dtype))
    raise VivTranslationError("vp.translate: %s is a %s; a translated function takes numbers, bools and arrays "
                              "(NumPy, pandas or polars columns, lists)" % (name, type(v).__name__))


_VIV_TYPE = {"float": "Num", "int": "Int", "bool": "Bool"}


def _annotate(viv: str, name: str, kinds: dict) -> str:
    """Give each unannotated parameter of `def name(...)` its Viv type, as
    the export needs every parameter annotated."""
    m = re.search(r"^def %s\((.*?)\)" % re.escape(name), viv, re.M)
    if m is None:
        raise VivTranslationError("vp.translate: py2viv wrote no def %s" % name)
    params = []
    for p in [p.strip() for p in m.group(1).split(",") if p.strip()]:
        if ":" not in p:
            k = kinds.get(p)
            if k is None or k not in _VIV_TYPE and k != "np.ndarray":
                raise VivTranslationError("vp.translate: parameter %s of %s has no type; annotate it or call the "
                                          "function so its type is known" % (p, name))
            p = "%s: %s" % (p, "Arr" if k == "np.ndarray" else _VIV_TYPE[k])
        params.append(p)
    return viv[:m.start(1)] + ", ".join(params) + viv[m.end(1):]


# ------------------------------------------------------------- translation

def _refusal_error(fn, res, lmap):
    out, refs = [], []
    for r in res["refusals"]:
        line, col, _, _ = r.where()
        file, orig = lmap[line - 1] if 0 < line <= len(lmap) and lmap[line - 1] else ("<python>", line)
        try:
            code = open(file).read().splitlines()[orig - 1].rstrip()
        except (OSError, IndexError):
            code = ""
        refs.append((file, orig, col + 1, r.what, r.hint))
        out.append("%s:%d:%d: %s\n    %s\n  %s" % (file, orig, col + 1, r.what, code, r.hint))
    return VivTranslationError("vp.translate(%s): py2viv refuses %d construct%s (translation is exact or refused; "
                               "tools/py2viv/SUBSET.md lists what translates):\n%s"
                               % (fn.__name__, len(refs), "" if len(refs) == 1 else "s", "\n".join(out)), refs)


def _translate_text(fn, kinds, annotate=True):
    """(viv text, py2viv result) for `fn` with the parameter types `kinds`
    ({name: annotation}); parameters the user annotated keep theirs."""
    py2viv = _py2viv()
    text, lmap = _collect(fn)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            for a in node.args.args:
                if a.annotation is None and a.arg in kinds:
                    a.annotation = ast.parse(kinds[a.arg], mode="eval").body
            node.decorator_list = []
    # annotations are inserted in place, so the line numbers stay
    src = _unparse_keep_lines(text, tree, fn.__name__)
    res = py2viv.translate(src, filename=fn.__code__.co_filename)
    if res["refusals"]:
        raise _refusal_error(fn, res, lmap)
    vname = py2viv.viv_name(fn.__name__)
    return (_annotate(res["viv"], vname, kinds) if annotate else res["viv"]), res, vname


def _unparse_keep_lines(text, tree, name):
    """`text` with the target def's header rewritten to carry the inserted
    annotations, every other line unchanged (so refusals keep their lines)."""
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body_start = node.body[0].lineno
            header = "def %s(%s)%s:" % (node.name, ast.unparse(node.args),
                                        " -> " + ast.unparse(node.returns) if node.returns else "")
            n = body_start - node.lineno
            lines[node.lineno - 1:body_start - 1] = [header] + [""] * (n - 1)
            if node.body[0].lineno == node.lineno:          # def f(x): return x  on one line
                raise VivTranslationError("vp.translate: write the body of %s on its own lines" % name)
    return "\n".join(lines) + "\n"


def _equal(a, b, rtol):
    """(equal, detail) of the Python result `a` and the Viv result `b`."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)):
            return False, "Python returned %r, Viv %r" % (type(a).__name__, type(b).__name__)
        for i, (x, y) in enumerate(zip(a, b)):
            ok, why = _equal(x, y, rtol)
            if not ok:
                return False, "result %d: %s" % (i, why)
        return True, ""
    x, y = np.asarray(a, dtype=float) if not isinstance(a, np.ndarray) else a, np.asarray(b)
    if np.shape(x) != np.shape(y):
        return False, "shape %s (Python) vs %s (Viv)" % (np.shape(x), np.shape(y))
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    if rtol is not None:
        same = same | (np.abs(x - y) <= rtol * np.maximum(np.abs(x), np.abs(y)))
    if bool(np.all(same)):
        return True, ""
    idx = np.unravel_index(int(np.argmin(same.ravel())), x.shape) if x.ndim else ()
    return False, "at %s: Python %r, Viv %r%s (%d of %d differ)" % (
        idx if idx else "the value", x[idx].item() if x.ndim else x.item(), y[idx].item() if y.ndim else y.item(),
        "" if rtol is None else " (rtol %g)" % rtol, int(np.size(same) - np.sum(same)), int(np.size(same)))


def _copy(v):
    return v.copy() if hasattr(v, "copy") else list(v) if isinstance(v, list) else v


class Translated:
    """A Python function and its Viv translation, called like the function.

    `.viv_source` is the Viv it runs (for the last signature translated),
    `.python` the original, `.versions` the compiled signatures, and
    `.inexact` the operations py2viv says may differ in the last bits."""

    def __init__(self, fn, check=True, fast=False, rtol=1e-12, show=False, eager=None):
        functools.update_wrapper(self, fn)
        self.python = fn
        self.check = check
        self.fast = fast
        self.rtol = rtol
        self.show = show
        self.signature = inspect.signature(fn)
        self.versions = {}           # (kinds) -> (callable, viv text, inexact, checked, result dtypes)
        self.viv_source = None
        self.inexact = []
        params = list(self.signature.parameters.values())
        for p in params:
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise VivTranslationError("vp.translate(%s): *args, **kwargs and keyword-only parameters are not "
                                          "translated; give %s a plain parameter" % (fn.__name__, p.name))
        annotated = {p.name: _ann_kind(p.annotation) for p in params}
        if eager or (eager is None and params and all(annotated.values())):
            self._version(tuple((n, k) for n, k in annotated.items()))
        elif eager is None:
            _translate_text(fn, {}, annotate=False)    # refusals of constructs surface now; types wait for the first call

    def _version(self, key):
        v = self.versions.get(key)
        if v is not None:
            return v
        viv, res, vname = _translate_text(self.python, dict(key))
        h = hashlib.sha256((viv + "\0" + repr(key)).encode()).hexdigest()[:16]
        src_file = Path(self.python.__code__.co_filename)
        base = _core.cache_root(src_file if src_file.is_file() else Path.cwd() / "x.py") / "translate"
        base.mkdir(parents=True, exist_ok=True)
        stem = "%s_%s" % (re.sub(r"\W", "_", self.python.__name__), h)
        path = base / (stem + ".viv")
        if not path.is_file() or path.read_text() != viv:
            tmp = base / (".%s.%d.tmp" % (stem, os.getpid()))
            tmp.write_text(viv)
            os.replace(tmp, path)
        try:
            mod = _core.load(path, fast=self.fast, name=stem)
        except _core.VivCompileError as e:
            raise VivTranslationError("vp.translate(%s): the translated Viv did not compile (a py2viv bug; "
                                      "please report it): %s\n%s" % (self.python.__name__, e, viv)) from e
        fn = getattr(mod, vname, None)
        if fn is None:
            notes = "; ".join(getattr(getattr(mod, "__vipyr__", None), "notes", []) or [])
            raise VivTranslationError("vp.translate(%s): the translation exports no %s (%s)"
                                      % (self.python.__name__, vname, notes or "no note"))
        v = self.versions[key] = [fn, viv, list(res.get("inexact", [])), False, None]
        self.viv_source, self.inexact = viv, v[2]
        if self.show:
            print(viv)
        return v

    def _bind(self, args, kw):
        b = self.signature.bind(*args, **kw)
        b.apply_defaults()
        return list(b.arguments.items())

    def __call__(self, *args, **kw):
        bound = self._bind(args, kw)
        kinds, vals = [], []
        for name, v in bound:
            k, x = _kind(v, name)
            ann = _ann_kind(self.signature.parameters[name].annotation)
            kinds.append((name, ann or k))
            vals.append(x)
        v = self._version(tuple(kinds))
        fn, viv, inexact, checked, dtypes = v
        out = fn(*vals)
        if self.check and not checked:
            ref = self.python(*[_copy(x) for _, x in bound])
            ok, why = _equal(ref, out, self.rtol if (inexact or self.fast) else None)
            if not ok:
                raise VivTranslationMismatch(
                    "vp.translate(%s): the Viv translation and the Python function differ: %s.\n"
                    "This is a bug in py2viv (its contract is exact or refused); the Viv was:\n%s"
                    % (self.python.__name__, why, viv))
            v[3] = True
            # what Python returned, per result: an array's dtype, or the
            # scalar's type (py2viv's Viv gives a number for an int it did
            # not type as Int; the check has just shown the value is integral)
            v[4] = tuple(np.asarray(r).dtype if isinstance(r, np.ndarray) else
                         int if isinstance(r, (int, np.integer)) and not isinstance(r, (bool, np.bool_)) else
                         bool if isinstance(r, (bool, np.bool_)) else None
                         for r in (ref if isinstance(ref, tuple) else (ref,)))
            dtypes = v[4]
        if dtypes:
            outs = out if isinstance(out, tuple) else (out,)
            outs = tuple(d(o) if d in (int, bool) and not isinstance(o, np.ndarray) else
                         o.astype(d) if d is not None and d not in (int, bool) and isinstance(o, np.ndarray)
                         and d != o.dtype else o for o, d in zip(outs, dtypes))
            out = outs if isinstance(out, tuple) else outs[0]
        return out

    def __repr__(self):
        return "<vp.translate of %s, %d compiled signature%s>" % (
            self.python.__qualname__, len(self.versions), "" if len(self.versions) == 1 else "s")


def _ann_kind(ann):
    """A Python annotation as py2viv reads it, or None."""
    if ann is inspect.Parameter.empty or ann is None:
        return None
    if ann is float:
        return "float"
    if ann is int:
        return "int"
    if ann is bool:
        return "bool"
    if ann is np.ndarray or (isinstance(ann, str) and ann in ("np.ndarray", "numpy.ndarray")):
        return "np.ndarray"
    if isinstance(ann, str) and ann in ("float", "int", "bool"):
        return ann
    return None


def translate(fn, *, check=True, fast=False, rtol=1e-12, show=False):
    """`fn` translated to Viv by py2viv and compiled: a callable with the same
    signature. Refused constructs raise VivTranslationError (file, line,
    construct, what to write instead). `check=True` compares the first call
    of each signature with `fn` itself (bit for bit, or `rtol` where py2viv
    lists an inexact operation). `show=True` prints the Viv; `.viv_source`
    holds it. `fast=True` compiles with the fast flags (last bits may differ;
    the check then uses `rtol`)."""
    return Translated(fn, check=check, fast=fast, rtol=rtol, show=show)


def jit(fn=None, *, check=True, fast=False, rtol=1e-12, show=False):
    """`@vp.jit` (or `@vp.jit(check=False)`): translate at the first call,
    with the types of that call's arguments; one compiled version per
    signature, cached on disk by the source's hash."""
    def wrap(f):
        return Translated(f, check=check, fast=fast, rtol=rtol, show=show, eager=False)
    return wrap(fn) if fn is not None else wrap

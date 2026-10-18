# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The functions vipyr exports, made from the specs (vipyr/_spec.py).

Each function compiles its program once through vipyr's cache (the
first call of a process loads the cached library; the first call after a
source change rebuilds it), checks its keyword arguments against the
manifest before any build or run, and returns a frozen result object.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import _core, _registry, _spec, magic, pit

_lock = threading.Lock()
_modules: dict = {}          # manifest or source path -> loaded vipyr module
_records = {"path": None}    # where run records go; None: kept on the result only

_NUM = (int, float, np.integer, np.floating)
_INT = (int, np.integer)


def record_runs(path=None):
    """Append each call's run record to `path` (a JSON-lines file, the
    `.viv_runs.jsonl` format of vipyr.magic); `record_runs(None)` stops.
    `$VIPYR_RUNS` sets it for the process."""
    _records["path"] = None if path is None else str(path)


def _runs_path():
    return _records["path"] or os.environ.get("VIPYR_RUNS")


def module_for(path: str):
    """The vipyr module of a manifest or .viv file under the repository,
    built once per process (and once per change through the cache)."""
    with _lock:
        m = _modules.get(path)
        if m is None:
            m = _modules[path] = _core.load(_spec.source(path))
        return m


# -------------------------------------------------------------- validation

class ParameterError(ValueError):
    """A keyword argument the manifest refuses; `.param` names it."""

    def __init__(self, func, param, msg):
        super().__init__("vipyr.%s: %s %s" % (func, param, msg))
        self.param = param


def _bounds(p, x, func, name, what="it"):
    for key, ok, word in (("min", lambda a, b: a >= b, "at least"), ("max", lambda a, b: a <= b, "at most"),
                          ("min_exclusive", lambda a, b: a > b, "above"),
                          ("max_exclusive", lambda a, b: a < b, "below")):
        if key in p and not ok(x, p[key]):
            raise ParameterError(func, name, "must be %s %s; %s is %r" % (word, _spec._num(p[key]), what, x))


def validate(func, p, v, given):
    """`v` checked against the parameter spec `p`; returns the value to pass."""
    name, t = p["name"], p["type"]
    if p.get("choices") is not None:
        if not isinstance(v, str):
            raise TypeError("vipyr.%s: %s must be a str, one of %s; got %r" % (func, name, ", ".join(p["choices"]), v))
        key = v
        if p.get("normalize") == "lower":
            key = v.lower()
        elif p.get("normalize") == "pyscf_basis":
            key = re.sub(r"[-_ ]", "", v.lower()).replace("*", "s")
            table = {re.sub(r"[-_ ]", "", c).replace("*", "s"): c for c in p["choices"]}
            key = table.get(key, v)
        if key not in p["choices"]:
            raise ParameterError(func, name, "must be one of %s; got %r" % (", ".join(p["choices"]), v))
        return key
    if t == "Bool":
        if not isinstance(v, (bool, np.bool_)):
            raise TypeError("vipyr.%s: %s must be True or False; got %r" % (func, name, v))
        return bool(v)
    if t == "Int":
        if isinstance(v, bool) or not isinstance(v, _NUM) or (isinstance(v, float) and not float(v).is_integer()):
            raise TypeError("vipyr.%s: %s must be an int; got %r" % (func, name, v))
        v = int(v)
        _bounds(p, v, func, name)
        return v
    if t == "Num":
        if isinstance(v, bool) or not isinstance(v, _NUM):
            raise TypeError("vipyr.%s: %s must be a number; got %r" % (func, name, v))
        v = float(v)
        if not math.isfinite(v) and not p.get("nonfinite"):
            raise ParameterError(func, name, "must be finite; got %r" % v)
        _bounds(p, v, func, name)
        return v
    if t in ("Complex", "Cplx"):
        if isinstance(v, bool) or not isinstance(v, (complex, np.complexfloating) + _NUM):
            raise TypeError("vipyr.%s: %s must be a complex number; got %r" % (func, name, v))
        return complex(v)
    if t == "Str":
        if not isinstance(v, str):
            raise TypeError("vipyr.%s: %s must be a str; got %r" % (func, name, v))
        return v
    if t in ("Arr", "[Int]", "IntArr"):
        try:
            a = np.asarray(v, dtype=np.int64 if t != "Arr" else np.float64)
        except (TypeError, ValueError):
            raise TypeError("vipyr.%s: %s must be an array of numbers; got %r" % (func, name, v)) from None
        if t != "Arr" and not np.array_equal(a, np.asarray(v, dtype=float)):
            raise TypeError("vipyr.%s: %s must hold whole numbers" % (func, name))
        shape = p.get("shape")
        rank = len(shape) if shape else p.get("rank")
        if rank and a.ndim != rank:
            raise ParameterError(func, name, "must have rank %d; got shape %s" % (rank, a.shape))
        if p.get("square") and a.ndim == 2 and a.shape[0] != a.shape[1]:
            raise ParameterError(func, name, "must be square; got shape %s" % (a.shape,))
        if shape:
            for d, want in zip(a.shape, shape):
                if isinstance(want, int) and want >= 0 and d != want:
                    raise ParameterError(func, name, "must have shape (%s); got %s" % (
                        ", ".join(str(s) for s in shape), a.shape))
                if isinstance(want, str) and want.lower() in given and d != len(np.atleast_1d(given[want.lower()])):
                    raise ParameterError(func, name, "must have one row per element of %s; got shape %s" % (
                        want.lower(), a.shape))
        if t == "Arr" and not p.get("nonfinite") and a.size and not np.all(np.isfinite(a)):
            raise ParameterError(func, name, "must be finite in every element")
        if a.size:
            _bounds(p, a.min(), func, name, "its smallest element")
            _bounds(p, a.max(), func, name, "its largest element")
        return a
    return v


# ----------------------------------------------------------------- results

@dataclasses.dataclass(frozen=True)
class CheckReport:
    """What `.check()` found. True when every test passed and every stored
    hash that applies matched."""
    passed: bool
    tests: tuple            # (name, passed, detail)
    hash: dict              # {"want", "got", "match"} or {} when no hash applies
    at_deck_settings: bool  # False: this run changed a setting the checks were written for
    reference: str

    def __bool__(self):
        return self.passed

    def __repr__(self):
        bad = [t[0] for t in self.tests if not t[1]]
        return "CheckReport(passed=%s, %d tests%s%s)" % (
            self.passed, len(self.tests), ", failed: %s" % bad if bad else "",
            ", hash %s" % ("matches" if self.hash.get("match") else "differs") if self.hash else "")


class Result:
    """Base of every vipyr result. Fields are the program's results (with
    units in each field's doc), then `params` (what was asked), `tests`
    (the program's own test blocks, when the run executes them), `log`
    (printed lines), `run_id` (the content address of the run record) and
    `record` (the record itself)."""

    _fields = ()
    # iterating gives the first _unpack fields (all when 0): curve_fit's popt, pcov
    _unpack = 0

    def check(self) -> CheckReport:
        return self._checker(self)

    def digits(self, **opts):
        """Which digits of the program's printed numbers are real, for this
        run's parameters: `vipyr.digits` on the program behind the result."""
        from .digits import for_result
        return for_result(self, **opts)

    def to_dict(self):
        return {f: getattr(self, f) for f in self._fields}

    def __iter__(self):
        return iter(getattr(self, f) for f in self._fields[:self._unpack or None])

    def __len__(self):
        return len(self._fields[:self._unpack or None])

    def __repr__(self):
        def short(v):
            if isinstance(v, np.ndarray):
                return "array(shape=%s)" % (v.shape,)
            r = repr(v)
            return r if len(r) < 60 else r[:57] + "..."
        body = ", ".join("%s=%s" % (f, short(getattr(self, f))) for f in self._fields)
        return "%s(%s, run_id=%r)" % (type(self).__name__, body, self.run_id)


def make_result_class(spec):
    fields = []
    doc = ["%s: the result of vp.%s (vipyr.%s)." % (spec["result"], spec["qualname"], spec["qualname"]), "", "Fields:"]
    for r in spec["results"]:
        md = {"doc": r.get("doc", ""), "units": r.get("units")}
        fields.append((r["name"], object, dataclasses.field(metadata=md)))
        doc.append("    %s (%s): %s" % (r["name"], r["python_type"], r.get("doc", "")))
    for name in ("params", "tests", "log", "run_id", "record"):
        fields.append((name, object, dataclasses.field(default=None, repr=False, compare=False)))
    doc += ["    params, tests, log, run_id, record: see vipyr.Result", "",
            ".check() runs the program's own checks; iterating gives the fields above in order"
            + (" (the first %d)." % spec["unpack"] if spec.get("unpack") else ".")]
    ns = {"_fields": tuple(r["name"] for r in spec["results"]), "_unpack": spec.get("unpack", 0),
          "__doc__": "\n".join(doc)}
    # a registry entry's `mixin` ("module:Class") adds methods to the result,
    # as vp.stats.ols's summary()
    bases = (Result,)
    mixin = _registry.FUNCTIONS.get(spec["qualname"], {}).get("mixin")
    if mixin:
        mod, _, name = mixin.partition(":")
        bases = (getattr(importlib.import_module("vipyr." + mod), name), Result)
    cls = dataclasses.make_dataclass(spec["result"], fields, bases=bases, namespace=ns, frozen=True, eq=False,
                                     repr=False)
    cls.__repr__ = Result.__repr__
    return cls


# ----------------------------------------------------------- run records

def _record(spec, source_sha, kw, out, wall):
    rec = magic.make_record("vipyr." + spec["family"], spec["name"], source_sha, False, (), kw, out, wall)
    p = _runs_path()
    if p:
        magic.append_record(rec, p)
    return rec


# ------------------------------------------------------------------ checks

def _generated_test_module(mod):
    """The `test_<name>.py` viv export wrote beside the wrapper, with the
    wrapper as its import (for its result_hash)."""
    e = mod.__vipyr__
    path = e.path / ("test_%s.py" % e.name)
    if not path.is_file():
        return None
    saved = sys.modules.get(e.name)
    sys.modules[e.name] = mod
    try:
        sp = importlib.util.spec_from_file_location("_vipyr_test_" + e.name, path)
        t = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(t)
        return t
    finally:
        if saved is None:
            sys.modules.pop(e.name, None)
        else:
            sys.modules[e.name] = saved
        p = str(e.path)
        if p in sys.path:
            sys.path.remove(p)


@functools.lru_cache(maxsize=None)
def _viv_run_tests(source: str):
    """The test blocks of a .viv file, by the interpreter: ((name, passed, detail), ...).
    They run from the repository root, where their data paths start."""
    path = _spec.source(source)
    r = subprocess.run([_core.compiler(), "run", str(path)], capture_output=True, text=True, env=_core._env(),
                       cwd=str(_spec.repo_root()))
    tests = []
    for line in r.stdout.splitlines():
        m = re.match(r"^(PASS|FAIL) (.*)$", line)
        if m:
            tests.append((m.group(2), m.group(1) == "PASS", ""))
    if r.returncode != 0 and all(t[1] for t in tests):
        tests.append(("viv run %s exits 0" % source, False, r.stderr.strip()[-400:]))
    return tuple(tests)


def _ref_text(spec):
    return "; ".join("%s: %s" % kv for kv in _spec.ref_items(spec))


def _case_of(spec, params):
    """The name of the recorded case (the manifest's `cases` file) whose
    settings are exactly the ones given, or None."""
    f = spec.get("cases_file")
    if not f or not f.endswith(".json") or not _spec.source(f).is_file():
        return None
    asked = {k: v for k, v in params.items() if v is not None}
    cases = json.loads(_spec.source(spec["cases_file"]).read_text())
    for name, c in cases.items():
        if {k.lower(): v for k, v in c.items()} == asked:
            return name
    return None


def _program_checker(spec, mod):
    def check(res):
        tests = tuple(res.tests or ())
        h = {}
        if spec.get("deck_param"):
            at_deck = all(k == spec["deck_param"] or v is None for k, v in res.params.items())
            key = res.params.get(spec["deck_param"])
        else:
            own = all(v is None for v in res.params.values()) and "" in spec["exact_hashes"]
            key = "" if own else _case_of(spec, res.params)   # "": the program's own settings
            at_deck = key is not None
        want = spec["exact_hashes"].get(key)
        if at_deck and want and not mod._FAST:
            t = _generated_test_module(mod)
            if t is not None:
                raw = {r.get("key", r["name"]): getattr(res, r["name"]) for r in spec["results"]}
                got = t.result_hash(raw)
                h = {"want": want, "got": got, "match": got == want}
        passed = all(t[1] for t in tests) and (not h or h["match"])
        return CheckReport(passed, tests, h, at_deck, _ref_text(spec))
    return check


def _kernel_checker(spec, mod, entry, source):
    def check(res):
        tests = _viv_run_tests(source)
        h = {}
        want = spec["exact_hashes"].get("cases")
        if want and spec.get("cases_file") and not mod._FAST:
            cases = json.loads(_spec.source(spec["cases_file"]).read_text()).get(entry, [])
            args = mod.KERNELS[entry][0]
            dec = lambda k, v: complex(v[0], v[1]) if k == "Cplx" else np.array(v, float) if k == "Arr" else v
            outs = [getattr(mod, entry)(*[dec(a[1], x) for a, x in zip(args, c)]) for c in cases]
            got = mod.results_hash(entry, outs)
            h = {"want": want, "got": got, "match": got == want}
        passed = all(t[1] for t in tests) and (not h or h["match"])
        return CheckReport(passed, tests, h, True, _ref_text(spec))
    return check


def _python_checker(spec, reg):
    """A kind "python" result's check (vp.ode, vp.optimize, vp.gpt): the
    library's own test blocks (the references it is held to), then the
    module's `_check_<impl>` of this result when it has one, else that this
    call succeeded."""
    own = getattr(importlib.import_module("." + reg.get("module", "_ode"), __package__), "_check_" + reg["impl"], None)

    def check(res):
        tests = tuple(t for src in reg.get("tests", []) for t in _viv_run_tests(src))
        if own is not None:
            tests = tests + tuple(own(res))
        else:
            ok = bool(np.all(np.asarray(res.success)))
            what = reg.get("check_text", "this solve reached t1 or a terminal event")
            tests = tests + ((what, ok, "" if ok else str(getattr(res, "status", getattr(res, "info", "")))),)
        return CheckReport(all(t[1] for t in tests), tests, {}, True, _ref_text(spec))
    return check


# ---------------------------------------------------------------- builders

def _signature(spec):
    ps = []
    kind = spec["kind"]
    for p in spec["params"]:
        if p.get("required"):
            ps.append(inspect.Parameter(p["name"], inspect.Parameter.POSITIONAL_OR_KEYWORD))
        elif p["type"] == "deck":
            ps.append(inspect.Parameter(p["name"], inspect.Parameter.POSITIONAL_OR_KEYWORD, default=p["default"]))
        elif kind == "feature":
            default = None if p["default"] == "None" else eval(p["default"], {})  # a literal from inspect
            kinds = inspect.Parameter.POSITIONAL_OR_KEYWORD if p["name"] == "df" else inspect.Parameter.KEYWORD_ONLY
            ps.append(inspect.Parameter(p["name"], kinds, default=default))
        elif kind == "python":
            how = inspect.Parameter.POSITIONAL_OR_KEYWORD if p.get("positional") else inspect.Parameter.KEYWORD_ONLY
            ps.append(inspect.Parameter(p["name"], how, default=eval(p["default"], {})))
        else:
            ps.append(inspect.Parameter(p["name"], inspect.Parameter.KEYWORD_ONLY, default=None))
    return inspect.Signature(ps)


def _default_text(spec, p):
    d = p.get("default")
    if p.get("default_shape"):
        return "the program's %d x %d array" % tuple(p["default_shape"])
    if isinstance(d, dict):
        order = [k for k in spec.get("decks") or [] if k in d] or list(d)
        return ", ".join("%s %s" % (k, d[k]) for k in order)
    return None if d is None or p.get("required") else str(d)


def docstring(spec):
    """numpydoc, from the same spec the /vipyr page is drawn from."""
    lines = [spec["summary"], ""]
    if spec.get("description"):
        lines += [spec["description"].strip(), ""]
    if spec["kind"] == "coming":
        return "\n".join(lines + ["Coming soon: calling it raises NotImplementedError.", spec.get("where", "")])

    def section(title):
        lines.extend([title, "-" * len(title)])

    section("Parameters")
    for p in spec["params"]:
        d = _default_text(spec, p)
        lines.append("%s : %s%s" % (p["name"], p["python_type"], ", default %s" % d if d else ""))
        body = [x for x in (p.get("help"), ("units: %s" % p["units"]) if p.get("units") else None,
                            ("range: %s" % p["range"]) if p.get("range") else None, p.get("range_doc"),
                            ("the program's %s" % p["alias_of"].upper()) if p.get("alias_of") else None) if x]
        lines += ["    " + x for x in body]
    for n in spec.get("notes", []):
        lines.append(n)
    lines += [""]
    section("Returns")
    lines.append("%s" % spec["result"])
    for r in spec["results"]:
        lines.append("    %s : %s" % (r["name"], r["python_type"]))
        if r.get("doc"):
            lines.append("        " + r["doc"])
    if spec.get("raises"):
        lines += [""]
        section("Raises")
        for e in spec["raises"]:
            lines += [e["name"], "    " + e["text"]]
    if spec.get("see_also"):
        lines += [""]
        section("See Also")
        lines += ["vp.%s" % q for q in spec["see_also"]]
    if spec.get("notes_md"):
        lines += [""]
        section("Notes")
        lines.append(spec["notes_md"].strip())
    if spec.get("references"):
        lines += [""]
        section("References")
        for i, r in enumerate(spec["references"], 1):
            link = (" https://doi.org/" + r["doi"]) if r.get("doi") else (" " + r["url"] if r.get("url") else "")
            lines.append(".. [%d] %s%s" % (i, r["text"], link))
    ref = _ref_text(spec)
    if ref:
        lines += [""]
        section("Checked against")
        lines.append(ref)
    exs = spec.get("examples") or ([{"code": spec["example"], "output": ""}] if spec.get("example") else [])
    if exs:
        lines += [""]
        section("Examples")
        for ex in exs:
            if ex.get("caption"):
                lines += [ex["caption"], ""]
            lines += [">>> " + l if l else ">>>" for l in ex["code"].splitlines()]
            if ex.get("output"):
                lines += ex["output"].splitlines()
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def make_function(spec):
    qual, kind = spec["qualname"], spec["kind"]
    sig = _signature(spec) if kind != "coming" else inspect.Signature(
        [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)])
    holder = {}

    def result_class():
        if "cls" not in holder:
            holder["cls"] = make_result_class(spec)
        return holder["cls"]

    def bind(args, kw):
        try:
            b = sig.bind(*args, **kw)
        except TypeError as e:
            raise TypeError("vipyr.%s: %s" % (qual, e)) from None
        return b.arguments

    if kind == "coming":
        def fn(**kw):
            raise NotImplementedError("vipyr.%s is coming soon (listed, not built): %s %s" % (qual, spec["summary"], spec["where"]))
    elif kind == "program":
        def fn(*args, **kw):
            given = bind(args, kw)
            params = {p["name"]: p for p in spec["params"]}
            deck_param = spec.get("deck_param")
            deck = given.pop(deck_param, spec["decks"][0]) if deck_param else None
            if deck_param and deck not in spec["decks"]:
                raise ParameterError(qual, deck_param, "must be one of %s; got %r" % (", ".join(spec["decks"]), deck))
            vals = {}
            lower = {k: v for k, v in given.items() if v is not None}
            for k, v in lower.items():
                p = params[k]
                vals[p.get("alias_of", k)] = validate(qual, p, v, {pp.get("alias_of", n): lower.get(n) for n, pp in params.items() if lower.get(n) is not None})
            for group, members in spec.get("groups", {}).items():
                ms = [m.lower() for m in members]
                have = [m for m in ms if m in vals]
                if have and len(have) != len(ms):
                    missing = [m for m in ms if m not in vals][0]
                    raise ParameterError(qual, missing, "must be given with %s (%s)" % (", ".join(have), group))
            mod = module_for(spec["manifest"])
            t0 = time.perf_counter()
            out = mod.run(deck, **vals) if deck_param else mod.run(**vals)
            wall = time.perf_counter() - t0
            results = {r["name"]: out.get(r.get("key", r["name"])) for r in spec["results"]}
            asked = dict(vals)
            if deck_param:
                asked[deck_param] = deck
            rec = _record(spec, mod.__vipyr__.info["key"], asked, results, wall)
            full = {p.get("alias_of", n): given.get(n) for n, p in params.items() if p["type"] != "deck"}
            if deck_param:
                full[deck_param] = deck
            r = result_class()(**results, params=full, tests=tuple(out["tests"]), log=tuple(out["log"]),
                               run_id=rec["id"], record=rec)
            object.__setattr__(r, "_checker", _program_checker(spec, mod))
            object.__setattr__(r, "_spec", spec)
            return r
    elif kind in ("kernel", "inferred"):
        entry = spec["name"] if kind == "kernel" else _registry.FUNCTIONS[qual]["entry"]
        entry = _registry.FUNCTIONS[qual].get("entry", entry)
        path = spec["manifest"] if kind == "kernel" else _registry.FUNCTIONS[qual]["source"]
        source = spec["sources"][0]

        def fn(*args, **kw):
            given = bind(args, kw)
            vals = [validate(qual, p, given[p["name"]], given) for p in spec["params"]]
            mod = module_for(path)
            f = getattr(mod, entry)
            f = getattr(f, "__vipyr_raw__", f)
            t0 = time.perf_counter()
            out = f(*vals)
            wall = time.perf_counter() - t0
            names = [r["name"] for r in spec["results"]]
            outs = out if len(names) > 1 else (out,)
            results = dict(zip(names, outs))
            rec = _record(spec, mod.__vipyr__.info["key"], dict(zip([p["name"] for p in spec["params"]], vals)),
                          results, wall)
            r = result_class()(**results, params=dict(given), tests=None, log=(), run_id=rec["id"], record=rec)
            object.__setattr__(r, "_checker", _kernel_checker(spec, mod, entry, source))
            return r
    elif kind == "feature":
        entry = _registry.FUNCTIONS[qual]["entry"]

        def fn(*args, **kw):
            given = bind(args, kw)
            given = {**{n: p.default for n, p in sig.parameters.items()}, **given}
            n = given.pop("n")
            if isinstance(n, bool) or not isinstance(n, _INT) or n < 1:
                raise ParameterError(qual, "n", "must be a positive int; got %r" % (n,))
            df = given.pop("df")
            cat = _catalog()
            e = cat.ENTRIES[entry]
            if df is None:
                df = e.make(int(n), 0)
            elif not hasattr(df, "columns"):
                raise TypeError("vipyr.%s: df must be a pandas DataFrame; got %r" % (qual, type(df).__name__))
            fn_ = getattr(cat, entry)
            t0 = time.perf_counter()
            frame = fn_(df, **given)
            wall = time.perf_counter() - t0
            rec = _record(spec, _spec.sha(_spec.source(spec["sources"][0])), {"df": df, **given},
                          {"frame": frame}, wall)
            r = result_class()(frame=frame, params={"df": df, **given}, tests=None, log=(), run_id=rec["id"],
                               record=rec)
            object.__setattr__(r, "_checker", _feature_checker(spec, entry, df, given, frame))
            return r
    elif kind == "python":
        reg = _registry.FUNCTIONS[qual]
        impl = getattr(importlib.import_module("." + reg.get("module", "_ode"), __package__), "_" + reg["impl"])

        def fn(*args, **kw):
            given = bind(args, kw)
            given = {**{n: p.default for n, p in sig.parameters.items() if p.default is not inspect.Parameter.empty},
                     **given}
            t0 = time.perf_counter()
            out = impl(**given)
            wall = time.perf_counter() - t0
            results = {r["name"]: out[r["name"]] for r in spec["results"]}
            asked = {k: (v if not callable(v) else getattr(v, "__name__", repr(v))) for k, v in given.items()}
            asked["problem"] = out["_problem"]
            rec = _record(spec, out["_problem"], asked, {k: v for k, v in results.items() if k != "message"}, wall)
            r = result_class()(**results, params=dict(given), tests=None, log=(), run_id=rec["id"], record=rec)
            object.__setattr__(r, "_checker", _python_checker(spec, reg))
            return r
    else:
        raise ValueError(kind)

    fn.__name__ = spec["name"]
    fn.__qualname__ = spec["name"]
    fn.__module__ = "vipyr." + spec["family"]
    fn.__doc__ = docstring(spec)
    fn.__signature__ = sig
    fn.spec = spec
    fn.result_class = result_class
    return fn


def _catalog():
    root = str(_spec.source("catalog"))
    if root not in sys.path:
        sys.path.insert(0, root)
    import catalog
    return catalog


def _feature_checker(spec, entry, df, given, frame):
    def check(res):
        cat = _catalog()
        e = cat.ENTRIES[entry]
        ref = getattr(cat, entry + "_reference")(df, **given)
        tests = []
        same = list(ref.columns) == list(frame.columns)
        for c in ref.columns if same else []:
            x, y = np.asarray(frame[c].to_numpy()), np.asarray(ref[c].to_numpy())
            if x.dtype.kind == "M" or y.dtype.kind == "M":
                x, y = x.astype("datetime64[ns]").view(np.int64), y.astype("datetime64[ns]").view(np.int64)
            if x.dtype.kind == "f" or y.dtype.kind == "f":
                x, y = x.astype(float), y.astype(float)
                ok = np.allclose(x, y, rtol=0 if e.exact else 1e-13, atol=0, equal_nan=True) if not e.exact else \
                    bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))
            else:
                ok = bool(np.array_equal(x, y))
            tests.append(("%s equals the pandas reference" % c, ok, ""))
        if not same:
            tests.append(("the columns are the reference's", False, "%s vs %s" % (list(frame.columns), list(ref.columns))))
        if e.kind == "feature" and not given:
            try:
                pit.check_point_in_time(e.fn, df, by=e.by, ts=e.ts, outcomes=e.outcomes, cuts=2)
                tests.append(("no feature reads a row after its own time (check_point_in_time)", True, ""))
            except pit.PointInTimeError as err:
                tests.append(("no feature reads a row after its own time (check_point_in_time)", False, str(err)[:400]))
        return CheckReport(all(t[1] for t in tests), tuple(tests), {}, True, _ref_text(spec))
    return check


# --------------------------------------------------------------- families

_specs_cache = {}


def specs(path=None):
    """The stored specs of one distribution: vipyr's own (vipyr/spec.json)
    unless `path` names another's spec.json."""
    key = str(path or _spec.SPEC_FILE)
    if key not in _specs_cache:
        with open(key, encoding="utf-8") as f:
            _specs_cache[key] = json.load(f)
    return _specs_cache[key]


def populate(namespace: dict, family: str, spec_file=None):
    """Put the family's functions and result classes into a module: one of
    vipyr's own, or a distribution's (`spec_file` is its spec.json)."""
    sp = specs(spec_file)
    fns = {q: s for q, s in sp["functions"].items() if s["family"] == family}
    for s in fns.values():
        namespace[s["name"]] = make_function(s)
    names = [s["name"] for s in fns.values()]
    namespace["__all__"] = names
    namespace["__vipyr_functions__"] = fns
    namespace["__doc__"] = "\n".join(
        [f["title"] + ": " + f["about"] for f in sp["fields"] if f["id"] == family] + [""] +
        ["%s  %s" % (s["signature"], "(coming soon)" if s["kind"] == "coming" else "") for s in fns.values()])
    return names

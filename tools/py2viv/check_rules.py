#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""check_rules.py: every translation rule proves itself, or the build fails.

    python3 tools/py2viv/check_rules.py              the checks (below); the differential
                                                     tests run when NumPy, viv, cc, node and
                                                     viv.wasm are here, else the ledger is read
    python3 tools/py2viv/check_rules.py --run        run the differential tests (and fail
                                                     without their tools), write the ledger
    python3 tools/py2viv/check_rules.py --only ID    the differential tests of the rules whose
                                                     id contains ID (no ledger written)
    python3 tools/py2viv/check_rules.py --write      also regenerate SUBSET.md's rule tables

The static checks (standard library only, so CI runs them anywhere):

  1. the rule files load (rule_engine.py's own checks: fields, exactness, patterns);
  2. every rule that translates has `tests`, and every parameter a pattern binds is read by
     the rule (a parameter nothing reads would drop an argument);
  3. the precedence audit: a template that writes an argument's code where an operator
     follows or precedes it parenthesizes it (`{x:71}`), and a template that is the argument
     alone passes the argument's own precedence on (`prec = "x.prec"`);
  4. SUBSET.md's rule tables are what `rule_engine.py --table` generates now;
  5. the ledger (results/rule_tests.json) was written from these very rules, translator and
     generator (a hash of the files), and every case in it passed.

The differential tests (tools: NumPy, a built viv, cc, node and viv.wasm, as diff.py): each
rule's `tests` are its seed inputs. For every spelling of every pattern, the generator writes
cases from them, systematically:

  seed        each test as written;
  expression  each argument given as an expression, and the call inside one, `-f(x - 0.5)`
              and `f(x - 0.5) * 3` (a template that drops an argument's precedence prints the
              wrong number);
  values      each number argument at 0, -0.0, its negative, NaN, inf, -inf, as an int, as a
              Bool and as a complex number; each array with a NaN, an infinity, a -0.0, all
              zeros, negated, as ints, as Bools, as complex numbers, as a Python list, and at
              another rank; each size negative, zero, a float; each dtype in turn;
  keywords    each parameter Python accepts by keyword, passed by keyword (and a default
              passed explicitly);
  arity       one positional argument too many, one keyword the function does not take, and
              the last required argument missing.

Each case is run in CPython with NumPy (in this process: the result, its type, or the
exception), and translated. The verdicts:

  PASS        translated; the three Viv engines print the same values (bit for bit for an
              `exact` rule and a case whose translation records no inexact operation, within
              the rule's rtol and atol otherwise) with the same type (int, float, Bool,
              complex, array rank and dtype, list or array); or Python raised and Viv stops
              with the same exception line;
  REFUSED     the translator refused the case at the call, with a message (the strict
              answer); a rule's own seed test may not be refused unless Python raises;
  STOP        Viv stopped with a `py2viv:` message where it cannot follow Python (a run-time
              refusal);
  SKIP        the case's setup is not translatable, or this NumPy lacks the spelling;
  FAIL        anything else: the translator crashed, refused somewhere other than the call,
              printed another value or type, the engines disagree, or the run ended otherwise.

A `refuse` rule's cases (each spelling, with no arguments, one, two, and a keyword) must be
refused with the rule's message. Runs are batched per rule (a batch that fails is rerun case
by case) and cached by content in tools/py2viv/.rule_cache/.
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

LEDGER = os.path.join(HERE, "results", "rule_tests.json")
REPORT = os.path.join(HERE, "results", "rule_tests.txt")
CACHE = os.path.join(HERE, ".rule_cache")
SUBSET = os.path.join(HERE, "SUBSET.md")
BEGIN = "<!-- BEGIN GENERATED: python3 tools/py2viv/rule_engine.py --table (check_rules.py --write) -->"
END = "<!-- END GENERATED -->"
# the files whose content the ledger is valid for
INPUTS = ["rule_engine.py", "py2viv.py", "funcs.py", "views.py", "classes.py", "maps.py", "infer.py", "strs.py", "ints.py", "check_rules.py", "diff.py",
          "rules/strings.toml", "rules/builtins.toml", "rules/numpy.toml", "rules/helpers.toml", "rules/modules.toml"] + [
    "../../lib/nprandom/%s.viv" % m for m in ("bits", "mt19937", "legacy", "seedseq", "pcg64", "ziggurat_tables", "generator")]


def without_spdx(data: bytes) -> bytes:
    """A file's bytes without its SPDX lines (LICENSING.md): the license
    header is not an input of the translator, so it does not void the ledger."""
    data = re.sub(rb"\n\n(?:# SPDX-[^\n]*\n)+\Z", b"\n", data)
    return re.sub(rb"(?m)^# SPDX-[^\n]*\n", b"", data)


def inputs_hash():
    h = hashlib.sha256()
    for f in INPUTS:
        h.update(f.encode())
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(without_spdx(fh.read()))
    return h.hexdigest()


# ================================================================ static checks


def load_engine():
    import rule_engine
    import py2viv
    return rule_engine, py2viv.RULES


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def check_tests_and_params(rules):
    """Every translating rule has tests; every bound parameter is read somewhere."""
    errs = []
    for r in rules:
        if r.exactness != "refuse" and not r.tests:
            errs.append("rule %s: no tests (a translating rule states the inputs of its differential test)" % r.id)
        text = " ".join(list(_strings(r.cases)) + ([r.when] if r.when else []))
        for p in r.patterns:
            for nm, how, default, optional in p.params:
                if not re.search(r"(?<![\w.])%s(?![\w])" % re.escape(nm), text):
                    errs.append("rule %s: parameter `%s` of `%s` is bound but never read (the argument would be "
                                "dropped)" % (r.id, nm, p.text))
            for extra in (p.varargs, p.varkw):
                if extra and not re.search(r"(?<![\w.])%s(?![\w])" % re.escape(extra), text) and r.id not in ("sqrt",):
                    pass
    return errs


_HOLE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
_SIMPLE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?::(\d+)|\|(.*))?$")
_OPERATOR = re.compile(r"(?:^|\s)(?:[-+*/@^%]|==|!=|<=|>=|<|>|and|or)(?:\s|$)")


def audit_template(rule, case, names):
    """The precedence audit of one `viv` template. Returns a list of problems.

    The template is split at its holes; parentheses and brackets are tracked in the text
    between them. An argument hole at depth 0 is in operator position when the template has
    any other text at depth 0 (an operator, a call around it being depth 1): it must carry a
    precedence (`{x:71}`) at least the case's result precedence. A template that is one hole
    and nothing else must pass the argument's precedence on (`prec = "x.prec"`), and a case
    whose depth-0 text has an operator must declare its own precedence below 100."""
    tpl = case["viv"]
    prec = case.get("prec", 100)
    out = []
    depth = 0
    pieces = []  # (kind, text, depth): kind "text" or "hole"
    pos = 0
    for m in _HOLE.finditer(tpl):
        pieces.append(("text", tpl[pos:m.start()], None))
        pieces.append(("hole", m.group(0), None))
        pos = m.end()
    pieces.append(("text", tpl[pos:], None))
    toplevel_text = []
    holes0 = []
    after_pow = False
    for kind, text, _ in pieces:
        if kind == "text":
            after_pow = depth == 0 and text.rstrip().endswith("^")
            for ch in text:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                elif depth == 0:
                    toplevel_text.append(ch)
            continue
        if text in ("{{", "}}"):
            continue
        inner = text[1:-1].strip()
        sm = _SIMPLE.match(inner)
        if depth == 0:
            if sm and sm.group(1) in names:
                # the right operand of ^ (right-associative, above unary minus) may be unary
                holes0.append((sm.group(1), sm.group(2), 80 if after_pow else None))
            elif ".code" in inner:
                holes0.append((inner, None, None))
            else:
                toplevel_text.append("x")
    top = "".join(toplevel_text).strip()
    only_hole = len(holes0) == 1 and top == "" and tpl.strip().startswith("{") and tpl.strip().endswith("}")
    if only_hole:
        nm = holes0[0][0]
        if str(prec) != "%s.prec" % nm:
            out.append("rule %s: the template `%s` is the argument alone, so its precedence is the argument's: "
                       "declare prec = \"%s.prec\" (now %s)" % (rule.id, tpl, nm, prec))
        return out
    has_op = bool(_OPERATOR.search(" " + re.sub(r"x", " x ", top) + " ")) or bool(holes0)
    if has_op and (prec == 100 or prec == "100"):
        out.append("rule %s: the template `%s` has an operator outside parentheses but declares no precedence"
                   % (rule.id, tpl))
    for nm, need, floor in holes0:
        if need is None:
            out.append("rule %s: `%s` writes the argument `%s` next to an operator without parenthesizing it "
                       "(write {%s:N})" % (rule.id, tpl, nm, nm))
        else:
            try:
                p = int(prec)
            except (TypeError, ValueError):
                continue
            if int(need) < (floor if floor is not None else p):
                out.append("rule %s: `%s` parenthesizes `%s` below its own precedence %s" % (rule.id, tpl, nm, prec))
    return out


def check_precedence(rules):
    errs = []
    n = 0
    for r in rules:
        names = set()
        for p in r.patterns:
            names |= {q[0] for q in p.params}
            if p.receiver:
                names.add(p.receiver)
        for c in r.cases:
            if "viv" in c:
                n += 1
                if r.kind == "slice":
                    continue  # a range of a subscript: each bound between the colons is a whole expression
                errs.extend(audit_template(r, c, names))
    return errs, n


def generated_tables(engine):
    import rule_engine
    return rule_engine.table(engine)


def check_subset(engine, write):
    text = open(SUBSET).read()
    if BEGIN not in text or END not in text:
        return ["SUBSET.md has no generated rule tables (%s ... %s)" % (BEGIN, END)]
    a = text.index(BEGIN) + len(BEGIN)
    b = text.index(END)
    want = "\n" + generated_tables(engine) + "\n"
    if text[a:b] == want:
        return []
    if write:
        with open(SUBSET, "w") as f:
            f.write(text[:a] + want + text[b:])
        return []
    return ["SUBSET.md's rule tables are not what the rules generate: run `python3 tools/py2viv/check_rules.py --write`"]


def check_ledger():
    if not os.path.exists(LEDGER):
        return ["no ledger (%s): run `python3 tools/py2viv/check_rules.py --run`" % os.path.relpath(LEDGER, ROOT)]
    led = json.load(open(LEDGER))
    errs = []
    if led.get("inputs") != inputs_hash():
        errs.append("the ledger was written for other rules or another translator: run `python3 tools/py2viv/check_rules.py --run`")
    if led.get("fail", 1):
        errs.append("the ledger records %s failing case(s)" % led.get("fail"))
    for rid, s in led.get("rules", {}).items():
        if s.get("fail"):
            errs.append("rule %s: %d failing case(s) in the ledger" % (rid, s["fail"]))
    return errs


# ================================================================ the generator


class Case:
    """One generated call: its setup lines (Python), the expression, where it came from."""

    def __init__(self, rule, spelling, kind, label, setup, expr, stmt=None, seed=False, targets=None, post=None,
                 expect=None):
        self.post = post or []    # lines after the call or statement, before the print (a write)
        self.expect = expect      # "pass": the case must translate and pass (a use that is exact)
        self.rule = rule
        self.spelling = spelling
        self.kind = kind          # seed, expression, values, keywords, arity, refuse
        self.label = label
        self.setup = setup        # list of Python lines
        self.expr = expr          # the call (an expression), or None for a statement case
        self.stmt = stmt          # a statement form (a.sort(); q, r = divmod(...))
        self.seed = seed
        self.targets = targets    # names a statement case prints
        self.verdict = None
        self.detail = ""
        self.py = None            # ("ok", value) or ("raise", line)
        self.inexact = False

    def lines(self, tag):
        """The case as Python lines, its result printed (names prefixed with tag)."""
        out = [l.replace("@", tag) for l in self.setup]
        if self.stmt is not None:
            out.append(self.stmt.replace("@", tag))
            out.extend(l.replace("@", tag) for l in self.post)
            out.append("print(%s)" % ", ".join(t.replace("@", tag) for t in self.targets))
        else:
            out.append("%sr = %s" % (tag, self.expr.replace("@", tag)))
            out.extend(l.replace("@", tag) for l in self.post)
            out.append("print(%sr)" % tag)
        return [x for l in out for x in l.split("\n")]

    def call_lines(self):
        """The 1-based lines (after PRELUDE) of the call or statement, and of the lines after it."""
        n_setup = sum(len(l.split("\n")) for l in self.setup)
        n_stmt = len((self.stmt if self.stmt is not None else "x").split("\n"))
        n_post = sum(len(l.split("\n")) for l in self.post)
        first = n_setup + 1
        return set(range(first, first + n_stmt)), set(range(first + n_stmt, first + n_stmt + n_post))


class PyStr(str):
    """A rule's test value that is a Python string (`{ str = "..." }` in the rule files; a plain
    string there names a number: "inf", "nan", a complex number)."""


def pyvals(v):
    """A rule's test value with its `{ str = ... }` tables read as PyStr."""
    if isinstance(v, dict) and set(v) == {"str"}:
        return PyStr(v["str"])
    if isinstance(v, dict):
        return {k: pyvals(x) for k, x in v.items()}
    if isinstance(v, list):
        return [pyvals(x) for x in v]
    return v


def is_strs(v):
    return isinstance(v, list) and bool(v) and all(isinstance(x, PyStr) for x in v)


# The strings each string argument is also tried at (value_variants): empty, one character,
# beyond ASCII (two-byte, combining, non-BMP), every kind of space, separators, case mappings
# that change the length, control characters, quotes and backslashes, digits.
STR_VARIANTS = [("empty", ""), ("one character", "a"), ("accented", "héllo wörld"), ("non-BMP", "𝔘𝔫𝔦 😀"),
                ("combining", "e\u0301le\u0300ve"), ("spaces", " \t\n\u00a0\u2003 x \u3000y \x1c"),
                ("separators", "a,b,,c;d"), ("case", "Straße ΑΣ İ ǅ"), ("controls", "\x07\x1b\x7f"),
                ("quotes", "it's \"q\" \\"), ("digits", " 1_000.5 "), ("line breaks", "a\r\nb\u2028c\x0cd")]


def _flat(v):
    if isinstance(v, list):
        for x in v:
            yield from _flat(x)
    else:
        yield v


def _map(v, f):
    return [_map(x, f) for x in v] if isinstance(v, list) else f(v)


def _num(v):
    if isinstance(v, PyStr):
        return v
    if isinstance(v, str) and v in ("inf", "-inf", "nan"):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
    if isinstance(v, str) and v.endswith("j"):
        try:
            return complex(v)  # a complex number, written "1.5+0.5j"
        except ValueError:
            return v  # a string argument ("ij")
    return v


def pylit(v):
    """A Python literal for a scalar test value."""
    if isinstance(v, PyStr):
        return repr(str(v))
    v = _num(v)
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, complex):
        return "(%r + %rj)" % (v.real, v.imag) if v.imag >= 0 else "(%r - %rj)" % (v.real, -v.imag)
    if isinstance(v, float):
        if v != v:
            return "float('nan')"
        if v == math.inf:
            return "float('inf')"
        if v == -math.inf:
            return "float('-inf')"
        return repr(v)
    return repr(v)


def arr_lit(v, container="nd", dtype=None):
    body = repr(_map(v, lambda x: _Lit(pylit(x))))
    if container == "list" or (v == [] and container == "nd"):
        return body  # a test's `[]` is a Python list (a list to grow); "np" makes it an array
    if dtype == "bool":
        return "(np.array(%s) > 0.5)" % body
    return "np.array(%s)" % body


def arr_setup(name, v, container="nd", dtype=None):
    """Lines binding name to the array v. A rank-3 array is built element by element (Viv's
    array literals are vectors and matrices)."""
    if not (isinstance(v, list) and v and isinstance(v[0], list) and v[0] and isinstance(v[0][0], list)) \
            or container == "list":
        return ["%s = %s" % (name, arr_lit(v, container, dtype))]
    d = (len(v), len(v[0]), len(v[0][0]))
    cplx = any(isinstance(_num(x), complex) for x in _flat(v))
    isint = all(isinstance(x, int) and not isinstance(x, bool) for x in _flat(v))
    out = ["%s = np.zeros(%r%s)" % (name, d, ", dtype=complex" if cplx else (", dtype=int" if isint else ""))]
    for i in range(d[0]):
        for j in range(d[1]):
            for k in range(d[2]):
                out.append("%s[%d, %d, %d] = %s" % (name, i, j, k, pylit(v[i][j][k])))
    if dtype == "bool":
        out = [l.replace(name, name + "f", 1) for l in out]
        out.append("%s = %sf > 0.5" % (name, name))
    return out


class _Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def is_arr(v):
    return isinstance(v, list)


def value_variants(v):
    """(label, value, container, dtype) variants of one argument value."""
    out = []
    if isinstance(v, PyStr):
        # a string at each kind of text, and a value of another type (Python raises TypeError)
        return [("the string " + lab, PyStr(x), "nd", None) for lab, x in STR_VARIANTS if x != v] + \
            [("a number", 1.5, "nd", None), ("an int", 3, "nd", None), ("a list of strings", [PyStr("a"), PyStr("b")], "list", None)]
    if is_strs(v):
        return [("empty", [], "list", None), ("one", [PyStr("x")], "list", None),
                ("beyond ASCII", [PyStr("é"), PyStr(""), PyStr("𝔘"), PyStr("a b")], "list", None),
                ("repeated", [PyStr("b"), PyStr("a"), PyStr("b")], "list", None),
                ("a string", PyStr("ab"), "nd", None), ("numbers", [1.5, 2.0], "list", None)]
    if v == []:
        return [("a NumPy array", [1.0, 2.0], "np", None), ("a non-empty list", [1.0, 2.0], "list", None)]
    if is_arr(v) and any(isinstance(_num(x), complex) for x in _flat(v)):
        flat = [_num(x) for x in _flat(v)]
        return [("negated", _map(v, lambda x: -_num(x)), "nd", None),
                ("a NaN entry", [complex(math.nan, 0.0)] + flat[1:] if not isinstance(v[0], list) else v, "nd", None),
                ("as a list", v, "list", None),
                ("the real parts", _map(v, lambda x: complex(_num(x)).real), "nd", None)]
    if isinstance(_num(v), complex):
        x = _num(v)
        return [("negated", -x, "nd", None), ("real", x.real, "nd", None), ("an array", [x, -x], "nd", None)]
    if is_arr(v):
        flat = [_num(x) for x in _flat(v)]
        isint = all(isinstance(x, int) and not isinstance(x, bool) for x in flat)

        def put(k, new):
            it = iter(range(len(flat)))
            return _map(v, lambda x: new if next(it) == k else _num(x))
        if not isint:
            out.append(("a NaN entry", put(0, math.nan), "nd", None))
            out.append(("an inf entry", put(len(flat) - 1, math.inf), "nd", None))
            out.append(("a -inf entry", put(0, -math.inf), "nd", None))
            out.append(("a -0.0 entry", put(0, -0.0), "nd", None))
            out.append(("as ints", _map(v, lambda x: int(round(_num(x))) if math.isfinite(_num(x)) else 0), "nd", None))
        else:
            out.append(("as floats", _map(v, lambda x: float(_num(x))), "nd", None))
            out.append(("a negative entry", put(0, -7), "nd", None))
        out.append(("all zeros", _map(v, lambda x: 0 if isint else 0.0), "nd", None))
        out.append(("negated", _map(v, lambda x: -_num(x) if not isint or _num(x) != 0 else 0), "nd", None))
        out.append(("as Bools", _map(v, lambda x: 1.0 if _num(x) > 0 else 0.0), "nd", "bool"))
        out.append(("as complex", _map(v, lambda x: complex(_num(x), 0.5)), "nd", None))
        out.append(("as a list", v, "list", None))
        if isinstance(v[0], list):
            out.append(("flattened", [x for x in _flat(v)], "nd", None))
        elif len(v) % 2 == 0 and len(v) >= 2:
            h = len(v) // 2
            out.append(("as a matrix", [v[:h], v[h:]], "nd", None))
        else:
            out.append(("as a matrix", [v, v], "nd", None))
        return out
    x = _num(v)
    if isinstance(x, bool):
        return [("the other Bool", not x, "nd", None), ("as an int", int(x), "nd", None), ("as a float", float(x), "nd", None)]
    if isinstance(x, int):
        out = [("zero", 0, "nd", None), ("negated", -x if x else -3, "nd", None), ("-1", -1, "nd", None),
               ("as a float", float(x) + 0.5, "nd", None), ("a Bool", True, "nd", None), ("an array", [x, x + 1], "nd", None)]
        return out
    out = [("zero", 0.0, "nd", None), ("-0.0", -0.0, "nd", None)]
    if x == x and x != 0:
        out.append(("negated", -x, "nd", None))
    out += [("NaN", math.nan, "nd", None), ("inf", math.inf, "nd", None), ("-inf", -math.inf, "nd", None)]
    if math.isfinite(x):
        out += [("as an int", int(round(x)) or 2, "nd", None), ("as complex", complex(x, 0.5), "nd", None)]
    out += [("a Bool", True, "nd", None), ("an array", [x, -x, 0.5], "nd", None)]
    return out


def _rank(v):
    r = 0
    while isinstance(v, list) and v:
        r += 1
        v = v[0]
    return r


def rank_variants(v):
    """The values of array v at ranks 1, 2 and 3 (and an empty axis): for an axis= test."""
    flat = [_num(x) for x in _flat(v)]
    while len(flat) < 12:
        flat = flat + [x * -0.5 if not isinstance(x, int) or isinstance(x, bool) else x - 3 for x in flat]
    flat = flat[:12]
    return [flat[:5], [flat[0:4], flat[4:8], flat[8:12]], [[flat[0:3], flat[3:6]], [flat[6:9], flat[9:12]]],
            [flat[0:3]], [[x] for x in flat[:4]]]


DTYPES = ["float", "int", "complex", "bool", "np.float32", "np.int32", "np.float64"]

# np.random (families np-random-legacy and np-random-generator): the seeds each rule's cases
# are drawn from, and what a case prints after a call that is a statement (a seed, a shuffle)
# or returns a Generator.
LEGACY_SEEDS = ["0", "1", "42", "2**32 - 1", "[1, 2, 3]", "[4294967295, 0, 7]"]
GENERATOR_SEEDS = ["0", "1", "42", "2**32 - 1", "2**53 - 1", "2**64 + 5", "[1, 2, 3]", "[2**40, 5]"]


def random_prints(rule):
    """What a case of this np.random rule prints after the call, when the call is a statement
    (np.random.seed, shuffle) or its value a Generator (default_rng); None otherwise."""
    return {"np.random.seed": ["np.random.rand(3)", "np.random.randint(0, 2**40)"],
            "np.random.shuffle": ["@x"],
            "Generator.shuffle": ["@x", "@g.random()"],
            "np.random.default_rng": ["@r.random(3)", "@r.integers(0, 2**40)", "@r.integers(0, 9, size=3)"]}.get(rule.id)


class Gen:
    def __init__(self, rules, np_mod):
        self.rules = rules
        self.np = np_mod

    # ------------------------------------------------ spellings
    @staticmethod
    def spellings(pat):
        k = pat.key
        if pat.glob:
            return [k.replace("*", {"np.random": ".poisson", "random.": "random", "np.emath.": "sqrt", "np.lib.scimath.": "sqrt",
                                     "np.ma.": "sqrt"}.get(k.split("*")[0], "x"))]
        return [k]

    # NumPy leaves these arrays' entries as the memory held them: the test compares the shape
    UNINITIALIZED = ("np.empty", "np.empty_like")

    def cases(self, rule):
        if isinstance(rule, InferRule):
            return infer_cases(rule)
        if isinstance(rule, StrRule):
            return string_cases(rule)
        if isinstance(rule, ObjRule):
            return map_cases(rule) if rule.id.startswith("maps.") else object_cases(rule)
        out = []
        for pat in rule.patterns:
            for sp in self.spellings(pat):
                if rule.exactness == "refuse":
                    out.extend(self.refuse_cases(rule, pat, sp))
                    continue
                if pat.kind == "slice":
                    cs = self.slice_cases(rule, pat)
                elif pat.kind == "binop":
                    cs = self.binop_cases(rule, pat, sp)
                elif pat.kind == "newaxis":
                    cs = self.newaxis_cases(rule, pat)
                elif pat.kind in ("index", "index-store"):
                    cs = self.index_cases(rule, pat)
                elif pat.kind == "item":
                    cs = self.item_cases(rule, pat, sp)
                else:
                    cs = self.rule_cases(rule, pat, sp)
                if rule.family in ("np-random-legacy", "np-random-generator"):
                    cs = cs + self.random_cases(rule, pat, sp)
                if sp in self.UNINITIALIZED:
                    for c in cs:
                        c.expr = "(%s).shape" % c.expr
                out.extend(cs)
        return out

    # ------------------------------------------------ refuse rules
    def refuse_cases(self, rule, pat, sp):
        setup = ["@v = np.array([1.0, -2.0, 3.5])", "@H = np.array([[2.0, 1.0], [1.0, 3.0]])"]
        out = []
        if pat.kind == "unpack":
            for args in ["@H", "@H, 1"]:
                out.append(Case(rule, sp, "refuse", "(%s)" % args, list(setup), None,
                                stmt="@w, @V = %s(%s)" % (sp, args), targets=["@w"]))
            return out
        if pat.kind == "const":
            out.append(Case(rule, sp, "refuse", "the attribute", list(setup), sp))
            return out
        if pat.kind in ("method", "stmt"):
            for args in ["", "1", "@v", "@v, 1"]:
                call = "@v.%s(%s)" % (sp, args)
                if pat.kind == "stmt":
                    out.append(Case(rule, sp, "refuse", "(%s)" % args, list(setup), None, stmt=call, targets=["@v"]))
                else:
                    out.append(Case(rule, sp, "refuse", "(%s)" % args, list(setup), call))
            return out
        for args in ["", "@v", "@v, 1", "@v, foo=1"]:
            out.append(Case(rule, sp, "refuse", "(%s)" % args, list(setup), "%s(%s)" % (sp, args)))
        return out

    # ------------------------------------------------ translating rules
    def rule_cases(self, rule, pat, sp):
        out = []
        params = pat.params
        kinds = {nm: rule.params.get(nm, "expr") for nm, *_ in params}
        if pat.receiver:
            kinds[pat.receiver] = rule.params.get(pat.receiver, "expr")  # `list`: a Python list (funcs.py's rules)
        tests = [{k: pyvals(v) for k, v in t.items() if k != "only"} for t in (rule.tests or [{}])
                 if "only" not in t or sp in t["only"]]
        for ti, test in enumerate(tests):
            base = self.build(rule, pat, sp, test, kinds)
            if base is None:
                continue
            base.kind = "seed"
            base.seed = True
            base.label = "test %d %s" % (ti + 1, json.dumps(test)[:80])
            out.append(base)
            if pat.kind in ("const",):
                continue
            # the call inside an expression, and each argument as one
            if pat.kind != "stmt" and pat.kind != "unpack" and not random_prints(rule):
                exprs = self.expr_args(test, kinds)
                c = self.build(rule, pat, sp, test, kinds, exprs=exprs)
                if c is not None:
                    out.append(self._wrap(c, "-(%s)", "the call negated, each argument an expression"))
                    c = self.build(rule, pat, sp, test, kinds, exprs=exprs)
                    out.append(self._wrap(c, "%s * 3", "the call times 3, each argument an expression"))
            if ti > 0:
                continue  # the variants of the first test only (the other tests are seeds)
            # values (the receiver's too)
            for nm, how, default, optional in ([(pat.receiver, "recv", None, False)] if pat.receiver else []) + list(params):
                if nm not in test or isinstance(test[nm], str) and test[nm].startswith("@"):
                    continue
                kind = kinds.get(nm, "expr")
                if kind == "dtype":
                    for dt in DTYPES:
                        if dt != test[nm]:
                            t2 = dict(test, **{nm: dt})
                            c = self.build(rule, pat, sp, t2, kinds)
                            if c is not None:
                                c.kind, c.label = "values", "%s = %s" % (nm, dt)
                                out.append(c)
                    continue
                if kind == "seq":
                    v = test[nm]
                    first = v[0]
                    other = [[1.0, 2.0]] if _rank(first) == 1 else [1.0, 2.0]
                    for lab, val in [("as ints", [_map(e, lambda x: int(round(_num(x)))) for e in v]),
                                     ("as floats", [_map(e, lambda x: float(_num(x))) for e in v]),
                                     ("as complex", [_map(e, lambda x: complex(_num(x), 0.5)) for e in v]),
                                     ("one array", [first]), ("three arrays", v + [first]), ("no arrays", []),
                                     ("an array of another rank", v + [other]), ("a number", v + [2.5]),
                                     ("an empty vector", v + ["zeros0"]),
                                     ("a Python list", [(first,)] + v[1:]),
                                     ("real and complex", [first, _map(v[1] if len(v) > 1 else first, lambda x: complex(_num(x), 1.0))]),
                                     ("a NaN and a -0.0", [_map(e, lambda x: float(_num(x))) for e in v][:1] + [[math.nan, -0.0]] if _rank(first) == 1 else v)]:
                        c = self.build(rule, pat, sp, dict(test, **{nm: val}), kinds)
                        if c is not None:
                            c.kind, c.label = "values", "%s %s" % (nm, lab)
                            out.append(c)
                    c = self.build(rule, pat, sp, test, kinds)
                    if c is not None:
                        c.setup.append("@L = [%s]" % ", ".join("@%s%d" % (nm, k) for k in range(len(test[nm]))))
                        c.expr = re.sub(r"\(@%s0.*?\)" % nm, "@L", c.expr, count=1)
                        c.kind, c.label = "values", "%s held in a variable" % nm
                        out.append(c)
                    continue
                if kind == "shape":
                    for lab, val in [("zero", 0), ("negative", -1), ("a float", 2.5), ("a tuple", [2, 3]),
                                     ("a tuple with a negative", [2, -1]), ("a Bool", True)]:
                        t2 = dict(test, **{nm: val})
                        c = self.build(rule, pat, sp, t2, kinds)
                        if c is not None:
                            c.kind, c.label = "values", "%s %s" % (nm, lab)
                            out.append(c)
                    continue
                if nm == "axis" and "a" in test:
                    # every axis of the array, negative ones, one past each end, None; the array
                    # at ranks 1 to 3 with each of its axes
                    arrs = [test["a"]] + [x for x in rank_variants(test["a"]) if x != test["a"]]
                    for av in arrs:
                        rk = _rank(av)
                        for ax in list(range(-rk - 1, rk + 1)) + ["None"]:
                            for kd in ([None, True] if "keepdims" in [p[0] for p in params] else [None]):
                                t2 = dict(test, a=av, axis=ax)
                                t2.pop("keepdims", None)
                                if kd is not None:
                                    t2["keepdims"] = kd
                                c = self.build(rule, pat, sp, t2, kinds, raw={"axis"} if ax == "None" else set())
                                if c is not None:
                                    c.kind, c.label = "values", "rank %d, axis=%s%s" % (rk, ax, ", keepdims" if kd else "")
                                    out.append(c)
                    continue
                for lab, val, cont, dt in value_variants(test[nm]):
                    t2 = dict(test, **{nm: val})
                    if kind == "list" and is_arr(val) and cont == "nd":
                        cont = "list"  # a list method's receiver: every variant as a Python list
                    c = self.build(rule, pat, sp, t2, kinds, containers={nm: (cont, dt)})
                    if c is not None:
                        c.kind, c.label = "values", "%s %s" % (nm, lab)
                        out.append(c)
            # keywords: each parameter Python takes by keyword, by keyword
            for nm, how, default, optional in params:
                if how == "pos":
                    continue
                if nm in test:
                    c = self.build(rule, pat, sp, test, kinds, by_kw={nm})
                    if c is not None:
                        c.kind, c.label = "keywords", "%s= by keyword" % nm
                        out.append(c)
                elif default is not None or optional:
                    dv = pat.defaults.get(nm)
                    c = self.build(rule, pat, sp, dict(test, **{nm: dv if dv is not None else "None"}), kinds,
                                   by_kw={nm}, raw={nm} if dv is None else set())
                    if c is not None:
                        c.kind, c.label = "keywords", "%s=%r explicitly" % (nm, dv)
                        out.append(c)
            # arity
            if pat.kind in ("call", "method"):
                c = self.build(rule, pat, sp, test, kinds, extra="1")
                if c is not None:
                    c.kind, c.label = "arity", "one positional argument too many"
                    out.append(c)
                c = self.build(rule, pat, sp, test, kinds, extra_kw="foo=1")
                if c is not None:
                    c.kind, c.label = "arity", "an unknown keyword"
                    out.append(c)
                req = [nm for nm, how, d, opt in params if not opt and nm in test]
                if req:
                    t2 = {k: v for k, v in test.items() if k != req[-1]}
                    c = self.build(rule, pat, sp, t2, kinds)
                    if c is not None:
                        c.kind, c.label = "arity", "`%s` missing" % req[-1]
                        out.append(c)
        return out

    # ------------------------------------------------ np.random: seeds, sizes, interleavings
    def random_cases(self, rule, pat, sp):
        """The np.random cases beside the generic ones: each test from several seeds (np.random.seed
        of ints and of lists; default_rng of ints up to 2**64 + 5 and of lists); each size at 1,
        (1, 1), (0, 3), (3, 0) and 1000 draws; and each test between other draws, which share
        the state (a Generator's integers below 2**32 keep half of a 64-bit draw for the next
        one: the draws after the call show whether it took that half)."""
        out = []
        kinds = {nm: rule.params.get(nm, "expr") for nm, *_ in pat.params}
        legacy = rule.family == "np-random-legacy"
        tests = [{k: v for k, v in t.items() if k != "only"} for t in (rule.tests or [{}])]
        if rule.id == "np.random.seed":
            for v, raw in [(0, False), (1, False), (42, False), (4294967295, False), (4294967296, False), (-1, False),
                           ([1, 2, 3], False), ([4294967295, 0, 7], False), ("[]", True), ("2**32", True), ("[2**32]", True),
                           ("[-1, 2]", True), ("[[1, 2], [3, 4]]", True), ("2.5", True), ("True", True)]:
                c = self.build(rule, pat, sp, {"seed": v}, kinds, raw={"seed"} if raw else set())
                if c is not None:
                    c.kind, c.label = "seeds", "np.random.seed(%s)" % (v,)
                    out.append(c)
            return out
        if rule.id == "np.random.default_rng":
            for v in ["0", "1", "42", "2**32 - 1", "2**32", "2**53 - 1", "2**64 + 5", "2**128 - 1", "[1, 2, 3]", "[2**40, 5]",
                      "[0]", "(7, 8)", "-1", "-(2**40)", "[1, -2]", "[]", "2.5", "[1.5]", "True"]:
                c = self.build(rule, pat, sp, {"seed": v}, kinds, raw={"seed"})
                if c is not None:
                    c.kind, c.label = "seeds", "default_rng(%s)" % v
                    out.append(c)
            c = self.build(rule, pat, sp, {"seed": 5}, kinds)
            if c is not None:
                c.setup.append("@n = 5")
                c.stmt = c.stmt.replace("(@seed)", "(@n * 3)")
                c.kind, c.label = "seeds", "default_rng of an int expression"
                out.append(c)
            return out
        seeds = LEGACY_SEEDS if legacy else GENERATOR_SEEDS
        first = rule.setup[0] if rule.setup else None
        for ti, test in enumerate(tests):
            for sd in seeds:
                c = self.build(rule, pat, sp, test, kinds)
                if c is None or first is None:
                    continue
                c.setup[0] = "np.random.seed(%s)" % sd if legacy else "@g = np.random.default_rng(%s)" % sd
                c.kind, c.label = "seeds", "test %d, seed %s" % (ti + 1, sd)
                out.append(c)
            # between other draws
            c = self.build(rule, pat, sp, test, kinds)
            if c is None:
                continue
            if legacy:
                pre, post = ["@x0 = np.random.randint(0, 100)"], ["@x1 = np.random.rand()", "@x2 = np.random.randint(0, 2**40)"]
            else:
                pre, post = ["@x0 = @g.integers(0, 100)"], ["@x1 = @g.integers(0, 100)", "@x2 = @g.random()"]
            c.setup = c.setup + pre
            if c.stmt is None:
                c.stmt, c.expr = "@r = " + c.expr, None
                c.targets = ["@x0", "@r", "@x1", "@x2"]
            else:
                c.targets = ["@x0"] + c.targets + ["@x1", "@x2"]
            c.post = post
            c.kind, c.label = "interleave", "test %d between other draws" % (ti + 1)
            out.append(c)
        # the ziggurat's tail: default_rng(0)'s draw 303 reaches it, and the run must stop there
        if rule.id in ("Generator.standard_normal", "Generator.normal"):
            c = self.build(rule, pat, sp, dict(tests[0], size=400), kinds)
            if c is not None:
                c.setup[0] = "@g = np.random.default_rng(0)"
                c.kind, c.label, c.solo = "tail", "400 draws from default_rng(0): draw 303 is in the tail", True
                out.append(c)
        # sizes
        if "size" in kinds and tests:
            base = dict(tests[-1])
            for label, v in [("1", 1), ("(1, 1)", [1, 1]), ("(0, 3)", [0, 3]), ("(3, 0)", [3, 0]), ("1000", 1000)]:
                c = self.build(rule, pat, sp, dict(base, size=v), kinds)
                if c is not None:
                    c.kind, c.label = "sizes", "size %s" % label
                    if v == 1000:
                        c.solo = True
                    out.append(c)
        return out

    # ------------------------------------------------ slices: `a[lo:hi:step]` on each axis
    def slice_cases(self, rule, pat):
        """The cases of a slice rule. From each test (an array `a` and the bounds `lo`, `hi`,
        `step` it states), the range is put on each axis, beside ranges and plain indices on the
        others (literal, negative, a variable, out of bounds, a float); its bounds are varied
        (omitted, 0, inside, at and past either end, negative, far negative, reversed, None, a
        variable, a float, a Bool) and its step (1, 2, 3, 0, -1, a variable); the array is
        given as ints, Bools, complex numbers and a Python list; and the slice is used as a
        value (an expression, a reduction) and bound to a name, with the writes that would
        show NumPy's view (each must be refused) and the reads that would not (each must
        pass)."""
        out = []
        sp = "[:]"

        def rank_of(v):
            r = 0
            while isinstance(v, list):
                r += 1
                v = v[0] if v else None
            return r

        def dims(v):
            d = []
            while isinstance(v, list):
                d.append(len(v))
                v = v[0] if v else None
            return d

        def rng(lo, hi, st):
            txt = "%s:%s" % ("" if lo is None else lo, "" if hi is None else hi)
            if st is not None:
                txt += ":%s" % st
            return txt

        def subscript(r, axis, item, other=":"):
            items = [other] * r
            items[axis] = item
            # trailing whole axes may be left out (NumPy and Viv take them whole)
            while len(items) > axis + 1 and items[-1] == ":":
                items.pop()
            return ", ".join(items)

        def case(kind, label, a, sub, setup=(), cont="nd", dtype=None, wrap="%s", post=None, stmt=None, targets=None,
                 expect=None, seed=False):
            st = arr_setup("@a", a, cont, dtype) + list(setup)
            if stmt is not None:
                return Case(rule, sp, kind, label, st, None, stmt=stmt.replace("{S}", "@a[%s]" % sub),
                            targets=targets, post=post, expect=expect, seed=seed)
            return Case(rule, sp, kind, label, st, wrap % ("@a[%s]" % sub), post=post, expect=expect, seed=seed)

        for ti, test in enumerate(rule.tests):
            a = test["a"]
            r = rank_of(a)
            lo, hi, st = test.get("lo"), test.get("hi"), test.get("step")
            item = rng(lo, hi, st)
            n0 = dims(a)[0]
            # seeds: the test's range on each axis
            for axis in range(r):
                out.append(case("seed", "test %d on axis %d: a[%s]" % (ti + 1, axis, subscript(r, axis, item)), a,
                                subscript(r, axis, item), seed=True))
            out.append(case("expression", "test %d negated" % (ti + 1), a, subscript(r, 0, item), wrap="-(%s)"))
            out.append(case("expression", "test %d times 3" % (ti + 1), a, subscript(r, 0, item), wrap="%s * 3"))
            if ti > 0 and r == 1:
                continue
            # the bounds
            n = dims(a)[0]
            bounds = [None, 0, 1, n - 1, n, n + 3, -1, -2, -(n + 3)]
            for b_lo in bounds:
                for b_hi in bounds:
                    if b_lo is None and b_hi is None:
                        continue
                    lab = "a[%s]" % rng(b_lo, b_hi, None)
                    out.append(case("values", lab, a, subscript(r, 0, rng(b_lo, b_hi, None))))
            for (vlo, vhi) in [(2, 1), (-1, 1), (1, 1)]:
                out.append(case("values", "reversed or empty a[%d:%d]" % (vlo, vhi), a, subscript(r, 0, rng(vlo, vhi, None))))
            for lab, sub, setup in [("None bounds", "None:None", ()), ("a[None:2]", "None:2", ()),
                                    ("variable bounds", "@lo:@hi", ("@lo = 1", "@hi = %d" % n)),
                                    ("negative variable bounds", "@lo:@hi", ("@lo = -2", "@hi = -1")),
                                    ("variable bounds past the ends", "@lo:@hi", ("@lo = -%d" % (n + 5), "@hi = %d" % (n + 5))),
                                    ("bound expressions", "@lo - 1:@hi + 1", ("@lo = 2", "@hi = 1")),
                                    ("a float bound", "1.0:", ()), ("a float variable bound", ":@hi", ("@hi = 2.0",)),
                                    ("a Bool bound", "True:", ()), ("an array bound", "@lo:", ("@lo = np.array([1])",)),
                                    ("step 1", "::1", ()), ("step 2", "1::2", ()), ("step 3 from the end", "-4::3", ()),
                                    ("step 2 bounded", "0:%d:2" % n, ()), ("step 0", "::0", ()), ("step -1", "::-1", ()),
                                    ("step -2 bounded", "%d:0:-2" % n, ()), ("a variable step", "::@st", ("@st = 2",)),
                                    ("a float step", "::2.0", ())]:
                out.append(case("values", lab, a, subscript(r, 0, sub), setup=setup))
            # the array's dtype and container
            flat = list(_flat(a))
            isint = all(isinstance(x, int) and not isinstance(x, bool) for x in flat)
            out.append(case("values", "a as ints" if not isint else "a as floats",
                            _map(a, lambda x: float(x) if isint else int(round(x))), subscript(r, 0, item)))
            out.append(case("values", "a as Bools", _map(a, lambda x: 1.0 if x > 0 else 0.0), subscript(r, 0, item), dtype="bool"))
            out.append(case("values", "a as complex", _map(a, lambda x: complex(x, 0.5)), subscript(r, 0, item)))
            out.append(case("values", "a as a Python list", a, subscript(r, 0, item), cont="list"))
            out.append(case("values", "a with a NaN and a -0.0", _map(a, lambda x: x), subscript(r, 0, item),
                            setup=["@a%s = float('nan')" % ("[0]" * r), "@a%s = -0.0" % ("[-1]" * r)] if not isint else []))
            if r == 1:
                out.append(case("values", "a number", a, item, setup=["@a = 2.5"]))
                out.append(case("values", "two ranges into a vector", a, "1:, 1:"))
                out.append(case("values", "a subscript of a slice", a, item + "][0"))
                out.append(case("values", "an empty array", [], subscript(r, 0, item), setup=["@a = np.zeros(0)"]))
            # the other axes: ranges and plain indices
            if r >= 2:
                last = r - 1
                for oth, setup in [(":", ()), ("0", ()), ("-1", ()), ("@j", ("@j = 1",)), ("@j", ("@j = -1",)),
                                   ("%d" % (n0 + 2), ()), ("-%d" % (n0 + 2), ()), ("1.0", ()), ("True", ()),
                                   ("None", ()), ("...", ()), ("1:-1", ())]:
                    items = [oth] * r
                    items[last] = item
                    out.append(case("values", "a[%s]" % ", ".join(items), a, ", ".join(items), setup=setup))
                    items = [oth] * r
                    items[0] = item
                    out.append(case("values", "a[%s]" % ", ".join(items), a, ", ".join(items), setup=setup))
                if r >= 3:
                    out.append(case("values", "a[0, %s]" % item, a, "0, " + item))
                    out.append(case("values", "a[:, 1, %s]" % item, a, ":, 1, " + item))
                    out.append(case("values", "a[%s, :, -1]" % item, a, item + ", :, -1"))
                out.append(case("values", "a row, then a range: a[1][%s]" % item, a, "1][" + item))
                out.append(case("values", "too many indices", a, ", ".join([item] * (r + 1))))
            # the slice as a value
            for fn in ("np.sum(%s)", "np.mean(%s)", "np.max(%s)", "%s.copy()", "np.sqrt(np.abs(%s))", "len(%s)", "%s.shape"):
                out.append(case("expression", fn % "S", a, subscript(r, 0, item), wrap=fn))
            # the slice bound to a name: writes that show NumPy's view are refused; reads pass
            S = subscript(r, 0, item)
            z = "0" if r == 1 else ", ".join(["0"] * r)
            ok_reads = [
                ("read, then arithmetic", ["@c = @b * 2.0 + 1.0"], "@c"),
                ("read by a def that only reads", ["def @rd(v):\n    return v * 3.0", "@c = @rd(@b)"], "@c"),
                ("the array rebound after", ["@a = np.zeros(3)"], "@b"),
                ("the slice's name rebound, then written", ["@b = np.zeros(%s)" % ("3" if r == 1 else "(%s)" % ", ".join(["2"] * r)), "@b[%s] = 5.0" % z], "@b"),
            ]
            for lab, post, tgt in ok_reads:
                out.append(case("views", lab, a, S, stmt="@b = {S}", post=post, targets=[tgt, "@a"], expect="pass"))
            writes = [
                ("a store through the slice", ["@b[%s] = 9.5" % z], ["@a"]),
                ("`+=` of the slice", ["@b += 1.0"], ["@a"]),
                ("`*=` of the slice", ["@b *= 2.0"], ["@a"]),
                ("a store into the array", ["@a[%s] = 9.5" % z.replace("0", "-1", 1)], ["@b"]),
                ("`+=` of the array", ["@a += 1.0"], ["@b"]),
                ("a store into a slice of the array", ["@a[:] = 0.0"], ["@b"]),
                ("a second name for the array, written", ["@a2 = @a", "@a2[%s] = 9.5" % z], ["@b"]),
                ("a def that writes its argument, given the slice",
                 ["def @w(v):\n    v[%s] = 9.5\n    return 0" % z, "@k = @w(@b)"], ["@a"]),
                ("a def that writes its argument, given the array",
                 ["def @w(v):\n    v[%s] = 9.5\n    return 0" % z, "@k = @w(@a)"], ["@b"]),
                ("a def that writes the module's array",
                 ["def @w():\n    @a[%s] = 9.5\n    return 0" % z, "@k = @w()"], ["@b"]),
            ]
            if r == 1:
                writes.append(("`.sort()` of the slice", ["@b.sort()"], ["@a"]))
                writes.append(("`.fill()` of the array", ["@a.fill(0.5)"], ["@b"]))
            for lab, post, tgts in writes:
                out.append(case("views", lab, a, S, stmt="@b = {S}", post=post, targets=tgts))
            # the slice used in place, beside a write
            out.append(case("views", "a def that writes the array, in the same statement", a, S,
                            setup=["def @w(v):\n    v[%s] = 9.5\n    return 1.0" % z.replace("0", "-1", 1)],
                            stmt="@c = {S} * 1.0 + @w(@a)", targets=["@c"]))
            out.append(case("views", "passed to a def that writes it", a, S,
                            setup=["def @w(v):\n    v[%s] = 9.5\n    return 0" % z], stmt="@k = @w({S})", targets=["@a"]))
            out.append(case("views", "returned by a def, then the array written", a, S,
                            setup=["def @mid(v):\n    return v[%s]" % S], stmt="@b = @mid(@a)",
                            post=["@a[%s] = 9.5" % z.replace("0", "-1", 1)], targets=["@b"]))
            out.append(case("views", "returned by a def, only read", a, S,
                            setup=["def @mid(v):\n    return v[%s]" % S], stmt="@b = @mid(@a) * 2.0", targets=["@b"],
                            expect="pass"))
            out.append(case("views", "a loop over the slice that writes the array", a, S,
                            stmt="for @x in {S}:\n    @a[%s] = 7.5" % z, targets=["@a"]))
            if r == 1:
                out.append(case("views", "a loop over the slice that only reads", a, S, setup=["@t = 0.0"],
                                stmt="for @x in {S}:\n    @t = @t + @x", targets=["@t"], expect="pass"))
                out.append(case("views", "a store into a slice", a, S, setup=["@v = np.array(%s)" % repr([0.25] * 20)],
                                stmt="@a[%s] = @v[:len(@a[%s])]" % (S, S), targets=["@a"], expect="pass"))
                out.append(case("views", "a store of a number into a slice", a, S, stmt="@a[%s] = 0.5" % S,
                                targets=["@a"], expect="pass"))
                out.append(case("views", "`+=` into a slice", a, S, stmt="@a[%s] += 1.25" % S, targets=["@a"], expect="pass"))
                out.append(case("views", "a store of another length into a slice", a, S, stmt="@a[%s] = np.ones(7)" % S,
                                targets=["@a"]))
                out.append(case("views", "a store of one element broadcast into a slice", a, S,
                                stmt="@a[%s] = np.ones(1)" % S, targets=["@a"]))
            else:
                out.append(case("views", "a store of a row", a, S, stmt="@a[1] = @a[0] * 2.0", targets=["@a"], expect="pass"))
                out.append(case("views", "a store into a column", a, S,
                                stmt="@a[%s] = @a[%s] + 1.0" % (subscript(r, r - 1, "0"), subscript(r, r - 1, "1")),
                                targets=["@a"], expect="pass"))
                out.append(case("views", "a store broadcast along an axis", a, S,
                                stmt="@a[%s] = @a[0]" % subscript(r, 0, "1:"), targets=["@a"]))
        return out

    # ------------------------------------------------ broadcasting: `x + y` of two arrays
    FLOATS = [1.5, -2.0, 3.25, 0.5, -0.0, 4.0, -7.5, 2.0, 0.25, -1.0, 6.0, 8.5, -3.0, 0.75, 5.0, -0.5,
              9.0, 1.25, -4.5, 7.0, 0.125, -6.0, 2.5, 3.0]
    # shapes that broadcast, and shapes that do not (NumPy raises ValueError; the run stops)
    SHAPES = [((3,), (3,)), ((2, 3), (3,)), ((3,), (2, 3)), ((2, 3), (2, 1)), ((2, 1), (1, 3)), ((2, 3), (1, 3)),
              ((1,), (4,)), ((4,), (1,)), ((1, 1), (3,)), ((0,), (1,)), ((2, 0), (2, 1)), ((1, 0), (3, 1)),
              ((2, 1, 3), (4, 1)), ((2, 3, 4), (3, 4)), ((1, 3, 1), (2, 1, 4)), ((2, 2, 2), (2, 2, 2)),
              ((3,), (4,)), ((2, 3), (3, 2)), ((2, 3), (2,)), ((0,), (3,)), ((2, 1, 3), (2, 3, 1, 1))]

    @classmethod
    def filled(cls, shape, start=0, kind="float", specials=()):
        """A nested list of the given shape: floats (or ints) from a fixed pool, with the
        specials (index into the flat order, value) put in."""
        n = 1
        for d in shape:
            n *= d
        if kind == "int":
            flat = [((k * 7 + start * 3) % 11) - 5 for k in range(n)]
        elif kind == "bool":
            flat = [1.0 if (k + start) % 3 else 0.0 for k in range(n)]
        else:
            flat = [cls.FLOATS[(k + start) % len(cls.FLOATS)] for k in range(n)]
        for k, v in specials:
            if k < n:
                flat[k] = v
        it = iter(flat)

        def build(d):
            if len(d) == 1:
                return [next(it) for _ in range(d[0])]
            return [build(d[1:]) for _ in range(d[0])]
        return build(list(shape)) if shape else next(it)

    @staticmethod
    def setup_of(name, shape, value, kind="float", cont="nd"):
        """Lines binding name to an array of this shape (np.zeros for an empty one; a rank-3
        array element by element)."""
        if 0 in shape:
            return ["%s = np.zeros(%r%s)" % (name, shape if len(shape) > 1 else shape[0], ", dtype=int" if kind == "int" else "")]
        if len(shape) > 3:
            flat = [pylit(x) for x in _flat(value)]
            return ["%s = np.array([%s]).reshape(%r)" % (name, ", ".join(flat), shape)]
        return arr_setup(name, value, cont, "bool" if kind == "bool" else None)

    def binop_cases(self, rule, pat, op):
        """The cases of an operator rule: each test as written, the operands as expressions and
        the result inside one; every pair of SHAPES (they broadcast, or the run must stop as
        NumPy raises), with NaN, infinities and -0.0 among the entries; ints, Bools, complex
        numbers and a Python list as operands; an array with itself; an operand that calls a
        def; and for `+` and `-`, `x += y` (NumPy writes into x: a result larger than x
        raises), with x held by another name, and a store broadcast into a slice."""
        out = []
        sp = op

        def case(kind, label, setup, expr=None, stmt=None, targets=None, seed=False, expect=None):
            if stmt is not None:
                return Case(rule, sp, kind, label, setup, None, stmt=stmt, targets=targets, seed=seed, expect=expect)
            return Case(rule, sp, kind, label, setup, expr, seed=seed, expect=expect)

        for ti, test in enumerate(rule.tests):
            x, y = test["x"], test["y"]
            st = arr_setup("@x", x) + arr_setup("@y", y)
            out.append(case("seed", "test %d" % (ti + 1), st, "@x %s @y" % op, seed=True))
            out.append(case("expression", "test %d negated" % (ti + 1), st, "-(@x %s @y)" % op))
            out.append(case("expression", "test %d times 3, operands as expressions" % (ti + 1), st,
                            "((@x - 1) %s (@y * 2)) * 3" % op))
        arith = op in ("+", "-", "*", "/")
        for k, (sx, sy) in enumerate(self.SHAPES):
            specials_x = [(0, math.nan), (1, -0.0)] if k % 3 == 0 else [(1, math.inf)] if k % 3 == 1 else []
            specials_y = [(0, 0.0), (1, -math.inf)] if k % 2 == 0 else [(2, -0.0)]
            vx = self.filled(sx, k, specials=specials_x)
            vy = self.filled(sy, k + 5, specials=specials_y)
            st = self.setup_of("@x", sx, vx) + self.setup_of("@y", sy, vy)
            out.append(case("values", "shapes %s %s %s" % (sx, op, sy), st, "@x %s @y" % op))
        # dtypes and containers
        sx, sy = (2, 3), (3,)
        for lab, kx, ky in [("ints", "int", "int"), ("int and float", "int", "float"), ("Bools and floats", "bool", "float"),
                            ("two Bool arrays", "bool", "bool")]:
            st = self.setup_of("@x", sx, self.filled(sx, 1, kx), kx) + self.setup_of("@y", sy, self.filled(sy, 2, ky), ky)
            out.append(case("values", lab, st, "@x %s @y" % op))
        st = arr_setup("@x", self.filled((2, 3), 3)) + ["@y = %s" % arr_lit(self.filled((3,), 4), "list")]
        out.append(case("values", "an array and a Python list", st, "@x %s @y" % op))
        st = ["@x = %s" % arr_lit(self.filled((3,), 3), "list"), "@y = %s" % arr_lit(self.filled((3,), 4), "list")]
        out.append(case("values", "two Python lists", st, "@x %s @y" % op))
        cx = ["@x = np.zeros((2, 1), dtype=complex)", "@x[0, 0] = (1.5 + 0.5j)", "@x[1, 0] = (-2.0 - 0.0j)"]
        cy = _map(self.filled((3,), 2), lambda v: complex(v, -1.5))
        for lab, stx, sty in [("complex and complex", cx, arr_setup("@y", cy)),
                              ("complex and real", cx, arr_setup("@y", self.filled((3,), 2))),
                              ("real and complex", arr_setup("@x", self.filled((2, 1), 1)), arr_setup("@y", cy))]:
            out.append(case("values", lab + ", its second row", stx + sty, "(@x %s @y)[1]" % op))
        c1 = ["@x = np.array([(1.5 + 0.5j)])"]
        for lab, stx, sty in [("complex and complex", c1, arr_setup("@y", cy)),
                              ("complex and real", c1, arr_setup("@y", self.filled((3,), 2))),
                              ("real and complex", ["@x = np.array([-2.5])"], arr_setup("@y", cy)),
                              ("complex vectors of other lengths", ["@x = np.array([1j, 2j])"], arr_setup("@y", cy))]:
            out.append(case("values", lab + ", vectors", stx + sty, "@x %s @y" % op))
        # the same array on both sides; an operand that calls a def
        st = arr_setup("@x", self.filled((2, 3), 2, specials=[(0, math.nan), (4, -0.0)]))
        out.append(case("values", "an array with itself", st, "@x %s @x" % op))
        st2 = st + ["def @f(v):\n    return v * 2.0"]
        out.append(case("values", "operands that call a def", st2, "@f(@x) %s @f(@x)" % op))
        out.append(case("values", "rank 4 and rank 1", ["@x = np.zeros((2, 1, 2, 1))", "@y = np.array([1.0, 2.0])"], "@x %s @y" % op))
        # operands whose shapes are equal by their form keep Viv's operator: ranges that trim as
        # much (at lengths where they are empty too), np.roll, arithmetic with a number; ranges
        # that do not (NumPy raises where the lengths differ)
        for n in (0, 1, 2, 3, 5):
            st = ["@x = np.array(%r)" % [float(k) * 0.5 - 1.0 for k in range(n)] if n else "@x = np.zeros(0)"]
            out.append(case("values", "a stencil of length %d" % n, st, "@x[2:] %s (2.0 * @x[1:-1] %s @x[:-2])" % (op, op)))
            out.append(case("values", "ranges of other lengths, length %d" % n, st, "@x[:3] %s @x[1:4]" % op))
        st = ["@x = np.array([1.5, -2.0, 0.25, -0.0])", "@y = np.array([[1.0, 2.0], [3.0, 4.0]])"]
        out.append(case("values", "np.roll of the array", st, "(np.roll(@x, 1) - 0.5) %s np.abs(@x)" % op))
        out.append(case("values", "rows of a matrix", st, "@y[0] %s @y[1]" % op))
        out.append(case("values", "columns of a matrix, trimmed", st, "@y[1:, 0] %s @y[:-1, 1]" % op))
        if arith and op in ("+", "-"):
            aug = op + "="
            for lab, sx, sy in [("broadcast into x", (2, 3), (3,)), ("a length 1 of y", (2, 3), (2, 1)),
                                ("a result larger than x", (3,), (2, 3)), ("a result larger than x, same rank", (1, 3), (2, 3)),
                                ("shapes that do not broadcast", (2, 3), (2,))]:
                st = self.setup_of("@x", sx, self.filled(sx, 1)) + self.setup_of("@y", sy, self.filled(sy, 6))
                out.append(case("values", "`x %s y`, %s" % (aug, lab), st, stmt="@x %s @y" % aug, targets=["@x"]))
                out.append(case("values", "`x %s y` of an x another name holds, %s" % (aug, lab), st + ["@b = @x"],
                                stmt="@x %s @y" % aug, targets=["@x", "@b"]))
            st = self.setup_of("@x", (2, 3), self.filled((2, 3), 1, "int"), "int") + arr_setup("@y", self.filled((3,), 2))
            out.append(case("values", "`x %s y` of an int x and a float y" % aug, st, stmt="@x %s @y" % aug, targets=["@x"]))
            st = self.setup_of("@x", (2, 3), self.filled((2, 3), 1, "int"), "int") + ["@y = 0.5"]
            out.append(case("values", "`x %s 0.5` of an int x" % aug, st, stmt="@x %s @y" % aug, targets=["@x"]))
            st = arr_setup("@x", self.filled((2, 3), 1)) + arr_setup("@y", self.filled((3,), 7))
            out.append(case("values", "`x[1:] %s y` broadcast into a slice" % aug, st, stmt="@x[1:] %s @y" % aug, targets=["@x"]))
            out.append(case("values", "`x[:, 0] %s y` of another length" % aug, st, stmt="@x[:, 0] %s @y" % aug, targets=["@x"]))
            if op == "+":
                for lab, sub, val in [("a row broadcast into a slice", "0:2", "@y"), ("a number into a slice", "1:", "2.5"),
                                      ("a vector into a column", ":, 1", "np.array([7.0, 8.0])"),
                                      ("a column into a matrix slice", ":, 1:", "np.array([[7.0], [8.0]])"),
                                      ("a vector of another length into a slice", "0", "np.array([1.0, 2.0])"),
                                      ("a matrix into a row", "0", "np.ones((2, 3))")]:
                    out.append(case("values", "store: %s" % lab, st, stmt="@x[%s] = %s" % (sub, val), targets=["@x"]))
        return out

    # ------------------------------------------------ `None` (np.newaxis) in a subscript
    def newaxis_cases(self, rule, pat):
        """The cases of the newaxis rule: `None` and np.newaxis before, between and after the
        axes of each test array (and of it at ranks 1 to 3), beside ranges and plain indices;
        the result in an outer operation; bound to a name and only read (must pass), written
        through or its array written (must be refused); a store through `None`; a Python list;
        more than four axes."""
        out = []
        sp = "[None]"

        def case(kind, label, a, sub, wrap="%s", setup=(), post=None, stmt=None, targets=None, expect=None, seed=False,
                 cont="nd"):
            st = arr_setup("@a", a, cont) + list(setup)
            if stmt is not None:
                return Case(rule, sp, kind, label, st, None, stmt=stmt.replace("{S}", "@a[%s]" % sub),
                            targets=targets, post=post, expect=expect, seed=seed)
            return Case(rule, sp, kind, label, st, wrap % ("@a[%s]" % sub), post=post, expect=expect, seed=seed)

        for ti, test in enumerate(rule.tests):
            a = test["a"]
            out.append(case("seed", "test %d: a[None]" % (ti + 1), a, "None", seed=True))
            out.append(case("seed", "test %d: a[:, None]" % (ti + 1), a, ":, None", seed=True))
            out.append(case("expression", "test %d negated" % (ti + 1), a, ":, None", wrap="-(%s)"))
            out.append(case("expression", "test %d times 3" % (ti + 1), a, "None", wrap="%s * 3"))
            if ti > 0:
                continue
            for av in [a] + [v for v in rank_variants(a)[:3] if v != a]:
                r = _rank(av)
                subs = ["None", "np.newaxis", "None, :", ":, None", "None, None", "0, None", "None, 0", "1:, None",
                        "None, 1:", "-1, None", "None, ..."]
                if r >= 2:
                    subs += [":, None, :", ":, :, None", "0, :, None", "None, :, 1", ":, 0, None", "1:, None, :-1"]
                if r >= 3:
                    subs += [":, :, :, None", "None, :, :, :", ":, None, :, :"]
                for sub in subs:
                    out.append(case("values", "rank %d: a[%s]" % (r, sub), av, sub))
                out.append(case("values", "rank %d: more than four axes" % r, av, ", ".join(["None"] * (5 - r + 1))))
            flat = list(_flat(a))
            isint = all(isinstance(x, int) and not isinstance(x, bool) for x in flat)
            out.append(case("values", "a as ints" if not isint else "a as floats",
                            _map(a, lambda x: float(x) if isint else int(round(x))), ":, None"))
            out.append(case("values", "a as Bools", _map(a, lambda x: 1.0 if x > 0 else 0.0), ":, None", cont="nd"))
            out[-1].setup = [l.replace("@a =", "@af =", 1) if l.startswith("@a =") else l for l in out[-1].setup] + ["@a = @af > 0.5"]
            out.append(case("values", "a as complex", _map(a, lambda x: complex(x, 0.5)), ":, None"))
            out.append(case("values", "a Python list", a, ":, None", cont="list"))
            out.append(case("values", "a number", a, "None", setup=["@a = 2.5"]))
            if _rank(a) == 1:
                out.append(case("expression", "the outer difference", a, ":, None", wrap="%s - @a[None, :]"))
                out.append(case("expression", "an outer product with a comparison", a, ":, None", wrap="(%s * @a) > 1.0"))
            z = ", ".join(["0"] * (_rank(a) + 1))
            out.append(case("views", "bound, then only read", a, ":, None", stmt="@b = {S}", post=["@c = @b * 2.0"],
                            targets=["@c", "@a"], expect="pass"))
            out.append(case("views", "bound, then written through", a, ":, None", stmt="@b = {S}",
                            post=["@b[%s] = 9.5" % z], targets=["@a"]))
            out.append(case("views", "bound, then its array written", a, ":, None", stmt="@b = {S}",
                            post=["@a[%s] = 9.5" % ", ".join(["0"] * _rank(a))], targets=["@b"]))
            out.append(case("views", "a store through None", a, "None", stmt="{S} = 1.0", targets=["@a"]))
        return out

    # ------------------------------------------------ masks and integer indices
    def index_cases(self, rule, pat):
        """The cases of the mask and integer-index rules. From each test (`a`, its index arrays
        `i`, `j`, `k`, and for a store the value `v`), for each pattern whose axes fit a: the
        index array repeated, negative, out of bounds (NumPy raises IndexError), empty, of
        rank 2, a Python list, floats (refused); a mask all true, all false, of another length
        (IndexError), of the array's shape; a scalar index beside an array; index arrays that
        broadcast together or do not; a as ints, Bools, complex numbers, a Python list, empty.
        A read is a copy: bound to a name and written, the array must not change (must pass).
        A store: a number, a vector of the selected length, one entry, another length (NumPy
        raises), repeated indices (the last write wins), `+=` through repeated indices (added
        once); through a name bound to a slice, and beside a slice of the array (both refused,
        as the view check refuses them)."""
        out = []
        sp = pat.key
        store = pat.kind == "index-store"
        items = pat.key.strip("[]").split(",")
        names = [p[0] for p in pat.params]
        inames = names[:-1] if store else names

        def sub_of(ix):
            it = iter(ix)
            return ", ".join(":" if x == ":" else next(it) for x in items)

        def case(kind, label, a, idx, v=None, setup=(), acont="nd", akind=None, stmt=None, wrap="%s", post=None,
                 targets=None, expect=None, seed=False):
            """idx: {name: value or ('raw', code)}; the subscript's index names in order."""
            st = []
            if isinstance(a, tuple) and a and a[0] == "raw":
                st.append("@a = %s" % a[1])
            else:
                st.extend(arr_setup("@a", a, acont, "bool" if akind == "bool" else None))
            codes = []
            for nm in inames:
                val = idx[nm]
                if isinstance(val, tuple) and val[0] == "raw":
                    st.append("@%s = %s" % (nm, val[1]))
                elif isinstance(val, tuple) and val[0] == "list":
                    st.append("@%s = %s" % (nm, arr_lit(val[1], "list")))
                elif is_arr(val):
                    st.extend(arr_setup("@%s" % nm, val, "nd", "bool" if val and all(isinstance(x, bool) for x in _flat(val)) else None))
                else:
                    st.append("@%s = %s" % (nm, pylit(val)))
                codes.append("@%s" % nm)
            st.extend(setup)
            s_ = "@a[%s]" % sub_of(codes)
            if store:
                if isinstance(v, tuple) and v[0] == "raw":
                    st.append("@v = %s" % v[1])
                elif is_arr(v):
                    st.extend(arr_setup("@v", v))
                else:
                    st.append("@v = %s" % pylit(v))
                return Case(rule, sp, kind, label, st, None, stmt=(stmt or "{S} = @v").replace("{S}", s_),
                            targets=targets or ["@a"], post=post, expect=expect, seed=seed)
            if stmt is not None:
                return Case(rule, sp, kind, label, st, None, stmt=stmt.replace("{S}", s_), targets=targets, post=post,
                            expect=expect, seed=seed)
            return Case(rule, sp, kind, label, st, wrap % s_, post=post, expect=expect, seed=seed)

        axis = items.index("#") if "#" in items else 0
        for ti, test in enumerate(rule.tests):
            if "only" in test and pat.key not in test["only"]:
                continue
            a = test["a"]
            r = _rank(a)
            if len(items) > r and "#" not in items[r:]:
                continue
            idx = {nm: test[nm] for nm in inames if nm in test}
            if len(idx) != len(inames):
                continue
            v = test.get("v")
            if len(items) > r:
                out.append(case("values", "test %d: more indices than axes" % (ti + 1), a, idx, v))
                continue
            out.append(case("seed", "test %d" % (ti + 1), a, idx, v, seed=True))
            if not store:
                out.append(case("expression", "test %d negated" % (ti + 1), a, idx, wrap="-(%s)"))
                out.append(case("expression", "test %d times 3" % (ti + 1), a, idx, wrap="%s * 3"))
                out.append(case("views", "test %d bound, then written (a copy)" % (ti + 1), a, idx, stmt="@b = {S}",
                                post=["@b += 1"], targets=["@a", "@b"], expect="pass"))
            dims = []
            q = a
            while isinstance(q, list):
                dims.append(len(q))
                q = q[0] if q else None
            n_ax = dims[axis] if axis < len(dims) else 1
            flat = list(_flat(a))
            isint = all(isinstance(x, int) and not isinstance(x, bool) for x in flat)
            first = inames[0]
            i0 = idx[first]
            ismask = is_arr(i0) and all(isinstance(x, bool) for x in _flat(i0))
            variants = []
            if len(inames) == 1:
                if ismask:
                    variants += [("all true", [True] * n_ax), ("all false", [False] * n_ax),
                                 ("one entry too many", [True] * (n_ax + 1)), ("one entry too few", [True] * max(n_ax - 1, 0)),
                                 ("as a Python list", ("list", i0)),
                                 ("as 0/1 ints", [1 if x else 0 for x in i0] if _rank(i0) == 1 else i0)]
                    if r >= 2 and axis == 0:
                        variants.append(("of the array's shape", ("raw", "@a > 2")))
                else:
                    variants += [("repeated", [0, 0, 1, 0][:max(1, min(4, n_ax + 2))]), ("negative", [-1, -n_ax]),
                                 ("past the end", [0, n_ax]), ("before the start", [-(n_ax + 1)]),
                                 ("empty", ("raw", "np.zeros(0, dtype=int)")), ("of rank 2", [[0, n_ax - 1], [n_ax - 1, 0]]),
                                 ("as a Python list", ("list", i0)), ("floats", [0.0, 1.0]),
                                 ("a mask", [True] + [False] * (n_ax - 1))]
            else:
                variants += [("broadcast together", None), ("that do not broadcast", None), ("an int beside an array", None),
                             ("negative", None), ("past the end", None), ("a mask beside an array", None), ("floats", None)]
            for lab, val in variants:
                if len(inames) == 1:
                    ix = dict(idx, **{first: val})
                else:
                    ix = dict(idx)
                    rows = dims[0]
                    if lab == "broadcast together":
                        ix[inames[0]] = [[0], [rows - 1]]
                        ix[inames[1]] = [0, -1]
                    elif lab == "that do not broadcast":
                        ix[inames[0]] = [0, 1]
                        ix[inames[1]] = [0, 1, 0]
                    elif lab == "an int beside an array":
                        ix[inames[0]] = rows - 1
                    elif lab == "negative":
                        ix[inames[0]] = [-1, -rows]
                    elif lab == "past the end":
                        ix[inames[-1]] = [0, dims[len(inames) - 1]]
                    elif lab == "a mask beside an array":
                        ix[inames[0]] = [True] + [False] * (rows - 1)
                    elif lab == "floats":
                        ix[inames[0]] = [0.0]
                    if lab == "an int beside an array" and store and is_arr(v) and _rank(v) > 1:
                        continue
                vv = v
                if store and is_arr(v) and lab not in ("all true", "all false", "one entry too many", "one entry too few"):
                    vv = 2.5 if not isint else 3  # a number fits any selection
                out.append(case("values", "indices %s" % lab, a, ix, vv))
            if not store:
                out.append(case("values", "a as ints" if not isint else "a as floats",
                                _map(a, lambda x: float(x) if isint else int(round(x))), idx))
                out.append(case("values", "a as Bools", _map(a, lambda x: 1.0 if x > 0 else 0.0), idx, akind="bool"))
                out.append(case("values", "a as complex", _map(a, lambda x: complex(x, 0.5)), idx))
                out.append(case("values", "a as a Python list", a, idx, acont="list"))
                out.append(case("values", "a with a NaN and a -0.0", a, idx,
                                setup=["@a%s = float('nan')" % ("[0]" * r), "@a%s = -0.0" % ("[-1]" * r)] if not isint else []))
                if r >= 2 and len(inames) == 1:
                    out.append(case("values", "after a row: a[1][i]", a, idx, stmt="@c = @a[1][@%s]" % first if axis == len(items) - 1 else "@c = @a[1][@%s]" % first,
                                    targets=["@c"]))
            else:
                one = 2.5 if not isint else 3
                for lab, val in [("a number", one), ("one entry", [one]), ("an int value", 3),
                                 ("a value of another length", [one] * 7), ("a complex value", ("raw", "np.array([1j, 2j])")),
                                 ("a Bool value", True), ("a matrix value", [[one, one], [one, one]])]:
                    out.append(case("values", "store %s" % lab, a, idx, val))
                out.append(case("values", "store through a name bound to a slice", a, idx, v, setup=["@s = @a[:]"],
                                stmt="@s%s = @v" % ("[%s]" % sub_of(["@%s" % nm for nm in inames])), targets=["@a"]))
                sv = case("views", "a slice of the array, then a store through the indices", a, idx, v,
                          post=["@c = @s * 1.0"], targets=["@a", "@c"])
                sv.stmt, sv.post = "@s = @a[1:]", [sv.stmt] + sv.post
                out.append(sv)
                out.append(case("values", "`+=` through the indices", a, idx, v if not is_arr(v) else (2.5 if not isint else 3),
                                stmt="{S} += @v"))
                if len(inames) == 1 and not ismask:
                    out.append(case("values", "`+=` through repeated indices (added once)", a, dict(idx, **{first: [0, 0, 0]}),
                                    1.5 if not isint else 1, stmt="{S} += @v"))
                    out.append(case("values", "`=` through repeated indices (the last write wins)", a,
                                    dict(idx, **{first: [0, 0, 0]}), ("raw", "np.array([%s])" % ", ".join(
                                        "[%s]" % ", ".join([str(x)] * (dims[1] if r >= 2 and axis == 0 else 1)) if r >= 2 and axis == 0 else str(x)
                                        for x in ([1.0, 2.0, 3.0] if not isint else [1, 2, 3])))))
                    out.append(case("values", "`*=` through the indices", a, idx, 2.0 if not isint else 2, stmt="{S} *= @v"))
                    out.append(case("values", "`/=` through the indices", a, idx, 2.0, stmt="{S} /= @v"))
                    out.append(case("values", "`+=` through indices that call a def", a, idx, 1.0,
                                    setup=["def @g(t):\n    return t"], stmt="@a[@g(@%s)] += @v" % first if items == ["#"] else "{S} += @v"))
                out.append(case("values", "a Python list", a, idx, v, acont="list"))
                out.append(case("values", "a Bool array", _map(a, lambda x: 1.0 if x > 0 else 0.0), idx, v, akind="bool"))
        return out

    # ------------------------------------------------ np.nonzero(a)[k]
    def item_cases(self, rule, pat, sp):
        """The cases of an item rule (`np.nonzero(a)[k]`): each axis of the test array and of it
        at ranks 1 to 3, a negative k, a k past the last axis (IndexError), k a variable; the
        array all zeros, with NaN (nonzero) and -0.0 (zero), as ints, Bools, complex numbers, a
        Python list, empty, a number."""
        out = []
        pname = pat.params[0][0]
        for ti, test in enumerate(rule.tests):
            a = test[pname]
            k = test["k"]

            def case(kind, label, av, kk, setup=(), cont="nd", dt=None, seed=False):
                st = (arr_setup("@a", av, cont, dt) if not (isinstance(av, tuple) and av[0] == "raw") else ["@a = %s" % av[1]]) + list(setup)
                return Case(rule, sp, kind, label, st, "%s(@a)[%s]" % (sp, kk), seed=seed)
            out.append(case("seed", "test %d" % (ti + 1), a, k, seed=True))
            if ti > 0:
                continue
            out.append(Case(rule, sp, "expression", "the result as an index", arr_setup("@a", a),
                            "@a[%s(@a)[%s]] * 3" % (sp, k)))
            for av in [a] + [v for v in rank_variants(a)[:3] if v != a]:
                rk = _rank(av)
                for kk in list(range(-rk - 1, rk + 1)):
                    out.append(case("values", "rank %d, k = %d" % (rk, kk), av, kk))
            out.append(case("values", "k a variable", a, "@k", setup=["@k = 0"]))
            out.append(case("values", "all zeros", _map(a, lambda x: 0.0), 0))
            out.append(case("values", "a NaN and a -0.0", _map(a, lambda x: x), 0,
                            setup=["@a[0] = float('nan')", "@a[1] = -0.0"]))
            out.append(case("values", "as ints", _map(a, lambda x: int(round(x))), 0))
            out.append(case("values", "as Bools", _map(a, lambda x: 1.0 if x > 0 else 0.0), 0, dt="bool"))
            out.append(case("values", "as complex", _map(a, lambda x: complex(x, 0.5)), 0))
            out.append(case("values", "a Python list", a, 0, cont="list"))
            out.append(case("values", "empty", ("raw", "np.zeros(0)"), 0))
            out.append(case("values", "a number", ("raw", "2.5"), 0))
        return out

    @staticmethod
    def _wrap(c, fmt, label):
        c.expr = fmt % c.expr
        c.kind = "expression"
        c.label = label
        return c

    @staticmethod
    def expr_args(test, kinds):
        """Arguments written as expressions: a number or number array minus 0.5 (minus 1 for
        ints), so a dropped parenthesis changes the value."""
        out = {}
        for nm, v in test.items():
            if kinds.get(nm, "expr") != "expr" or isinstance(v, str) and (isinstance(v, PyStr) or v not in ("inf", "-inf", "nan")):
                continue
            if is_strs(v):
                continue
            flat = list(_flat(v)) if is_arr(v) else [v]
            if any(isinstance(x, bool) for x in flat):
                continue
            ints = all(isinstance(x, int) for x in flat)
            out[nm] = " - 1" if ints else " - 0.5"
        return out

    def build(self, rule, pat, sp, test, kinds, exprs=None, containers=None, by_kw=(), extra=None, extra_kw=None, raw=()):
        """The case for these argument values: setup lines binding each value to a name, and
        the call (or statement) passing the names. `@` is the case's name prefix."""
        setup = []
        args = []
        kws = []
        exprs = exprs or {}
        containers = containers or {}
        positional_ok = True
        recv = None

        def bind(nm, v):
            kind = kinds.get(nm, "expr")
            if nm in raw:
                return str(v)
            if isinstance(v, str) and v.startswith("@"):
                return v  # a name the rule's setup binds (a Generator)
            if kind == "dtype":
                return str(v)
            if kind == "literal":
                return pylit(v)  # a value the rule reads from the syntax (a format spec, indexing='ij')
            if nm == "axis" and isinstance(v, int) and not isinstance(v, bool):
                return str(v)  # an axis is written as a literal (the rules read it statically)
            if nm == "keepdims" and isinstance(v, bool):
                return str(v)  # so is keepdims
            if kind == "seq":
                # a tuple of arrays, each bound to its own name (a number, "zeros0" an empty
                # vector, a (value, "list") pair a Python list)
                names = []
                for k, e in enumerate(v):
                    en = "@%s%d" % (nm, k)
                    if e == "zeros0":
                        setup.append("%s = np.zeros(0)" % en)
                    elif isinstance(e, tuple):
                        setup.append("%s = %s" % (en, arr_lit(e[0], "list")))
                    elif is_arr(e):
                        setup.extend(arr_setup(en, e))
                    else:
                        setup.append("%s = %s" % (en, pylit(e)))
                    names.append(en)
                return "(%s)" % ", ".join(names) + ("," if len(names) == 1 else "")
            if kind == "shape":
                if isinstance(v, list):
                    return "(%s)" % ", ".join(pylit(x) for x in v) + ("," if len(v) == 1 else "")
                setup.append("@%s = %s" % (nm, pylit(v)))
                return "@%s" % nm
            cont, dt = containers.get(nm, ("list" if kind == "list" else "nd", None))
            if is_arr(v):
                setup.extend(arr_setup("@%s" % nm, v, cont, dt))
            else:
                setup.append("@%s = %s" % (nm, pylit(v)))
            return "@%s%s" % (nm, exprs.get(nm, ""))
        if pat.receiver:
            if pat.receiver not in test:
                return None
            recv = bind(pat.receiver, test[pat.receiver])
            if exprs.get(pat.receiver):
                recv = "(%s)" % recv
        for nm, how, default, optional in pat.params:
            if nm not in test:
                positional_ok = False
                continue
            v = test[nm]
            if nm == pat.varargs:
                continue
            code = bind(nm, v)
            if how == "kw" or nm in by_kw or (not positional_ok and how == "poskw"):
                if how == "pos":
                    return None
                kws.append("%s=%s" % (nm, code))
            elif not positional_ok:
                return None
            else:
                args.append(code)
        if pat.varargs and pat.varargs in test:
            for k, v in enumerate(test[pat.varargs]):
                setup.append("@%s%d = %s" % (pat.varargs, k, pylit(v)))
                args.append("@%s%d" % (pat.varargs, k))
        if extra is not None:
            args.append(extra)
        if extra_kw is not None:
            kws.append(extra_kw)
        call_args = ", ".join(args + kws)
        setup = list(getattr(rule, "setup", [])) + setup
        prints = random_prints(rule)
        if prints and pat.kind in ("call", "method"):
            call = "%s(%s)" % (sp, call_args) if pat.kind == "call" else "%s.%s(%s)" % (recv, sp, call_args)
            if rule.id == "np.random.default_rng":
                call = "@r = " + call
            return Case(rule, sp, "seed", "", setup, None, stmt=call, targets=prints)
        if pat.kind == "call":
            return Case(rule, sp, "seed", "", setup, "%s(%s)" % (sp, call_args))
        if pat.kind == "method":
            return Case(rule, sp, "seed", "", setup, "%s.%s(%s)" % (recv, sp, call_args))
        if pat.kind == "attr":
            return Case(rule, sp, "seed", "", setup, "%s.%s" % (recv, sp))
        if pat.kind == "const":
            return Case(rule, sp, "seed", "", setup, sp)
        if pat.kind == "stmt":
            return Case(rule, sp, "seed", "", setup, None, stmt="%s.%s(%s)" % (recv, sp, call_args), targets=[recv])
        if pat.kind == "unpack":
            return Case(rule, sp, "seed", "", setup, None, stmt="@q, @s = %s(%s)" % (sp, call_args), targets=["@q", "@s"])
        return None


# ================================================================ running a case


PRELUDE = "import numpy as np\nimport math\n"


# ================================================================ objects (classes.py)
#
# Classes, dataclasses, named tuples and dicts with literal keys are not library calls, so they
# have no rule in rules/*.toml; classes.py translates them. Each construct is a pseudo-rule here,
# whose generated cases run and are judged as a rule's: every value kind in a field (ints, -0.0,
# NaN, infinities, 2**52, Bools, int and float arrays, a matrix, a list), each read (a field, a
# method, a property, printing the object, an f-string), a copy that writes into an array field,
# a field rebound in place, through a def (state threading) and through a second name (refused),
# replace, unpacking, and a dict's len, `in`, get and loops. A case where Python raises must be
# refused or stop as Python does; a case that translates must print what Python prints.

class ObjRule:
    """A construct classes.py translates, standing in for a rule in the generated tests."""

    def __init__(self, id, construct):
        self.id = id
        self.construct = construct
        self.exactness = "exact"
        self.cases = []
        self.rtol = self.atol = None
        self.family = None
        self.patterns = []
        self.tests = []
        self.params = {}


class ObjCase(Case):
    """A case whose placeholder for the batch's name prefix is `$` (a decorator's `@` stays)."""

    def lines(self, tag):
        out = [l.replace("$", tag) for l in self.setup]
        if self.stmt is not None:
            out.append(self.stmt.replace("$", tag))
            out.append("print(%s)" % ", ".join(t.replace("$", tag) for t in self.targets))
        else:
            out.append("%sr = %s" % (tag, self.expr.replace("$", tag)))
            out.append("print(%sr)" % tag)
        return [x for l in out for x in l.split("\n")]


OBJ_CONSTRUCTS = ("class", "dataclass", "frozen", "NamedTuple", "namedtuple", "dict")
OBJ_VALUES = [("int", "3"), ("zero", "0"), ("negint", "-7"), ("2**52", "2 ** 52"), ("float", "1.5"), ("-0.0", "-0.0"),
              ("tiny", "1e-300"), ("nan", "np.nan"), ("inf", "np.inf"), ("-inf", "-np.inf"), ("True", "True"),
              ("False", "False"), ("floats", "np.array([1.5, -0.0, 2.25])"), ("ints", "np.array([1, -2, 3])"),
              ("matrix", "np.array([[1.0, 2.0], [3.0, 4.0]])"), ("list", "[0.5, 2.0]")]


# a second value of each kind (the same Python type, and the same shape for an array)
OBJ_ALT = {"int": "-4", "zero": "5", "negint": "0", "2**52": "-(2 ** 52)", "float": "-0.25", "-0.0": "0.0", "tiny": "-1e-310",
           "nan": "np.inf", "inf": "np.nan", "-inf": "1e308", "True": "False", "False": "True",
           "floats": "np.array([-1.0, 0.5, np.nan])", "ints": "np.array([7, 8, -9])",
           "matrix": "np.array([[0.5, -0.0], [1e-3, 4.5]])", "list": "[-1.0, 3.5]"}


def object_rules():
    return [ObjRule("objects.%s" % c, c) for c in OBJ_CONSTRUCTS]


def _obj_setup(construct, a, b):
    """The lines that define `$C` (or nothing, for a dict) and make `$x` from the values a and b."""
    if construct == "class":
        return ["class $C:\n    def __init__(self, a, b):\n        self.a = a\n        self.b = b\n"
                "    def get_a(self):\n        return self.a\n    @property\n    def pb(self):\n        return self.b\n"
                "    def set_a(self, v):\n        self.a = v",
                "$x = $C(%s, %s)" % (a, b)]
    if construct in ("dataclass", "frozen"):
        deco = "@dataclass(frozen=True)" if construct == "frozen" else "@dataclass"
        return ["from dataclasses import dataclass, replace",
                "%s\nclass $C:\n    a: object\n    b: object\n    def get_a(self):\n        return self.a" % deco,
                "$x = $C(%s, b=%s)" % (a, b)]
    if construct == "NamedTuple":
        return ["from typing import NamedTuple",
                "class $C(NamedTuple):\n    a: object\n    b: object\n    def get_a(self):\n        return self.a",
                "$x = $C(%s, %s)" % (a, b)]
    if construct == "namedtuple":
        return ["from collections import namedtuple", "$C = namedtuple(\"$C\", [\"a\", \"b\"])", "$x = $C(%s, %s)" % (a, b)]
    return ["$x = {\"a\": %s, \"b\": %s}" % (a, b)]


def object_cases(rule):
    k = rule.construct
    out = []
    fa = (lambda o: '%s["a"]' % o) if k == "dict" else (lambda o: "%s.a" % o)
    fb = (lambda o: '%s["b"]' % o) if k == "dict" else (lambda o: "%s.b" % o)
    setfa = (lambda o, v: '%s["a"] = %s' % (o, v)) if k == "dict" else (lambda o, v: "%s.a = %s" % (o, v))

    def add(kind, label, a, b, expr=None, stmt=None, targets=None, seed=False, extra=()):
        c = ObjCase(rule, k, kind, label, list(extra) + _obj_setup(k, a, b), expr, stmt=stmt, seed=seed, targets=targets)
        out.append(c)

    for name, a in OBJ_VALUES:
        b = "2.5"
        arr = name in ("floats", "ints", "matrix")
        seed = name in ("int", "float")
        add("seed" if seed else "values", "read a field (a = %s)" % name, a, b, expr=fa("$x"), seed=seed)
        add("values", "read the other field (a = %s)" % name, b, a, expr=fb("$x"))
        add("values", "print the object (a = %s)" % name, a, b, stmt="$z = 0", targets=["$x"])
        add("values", "an f-string of the object (a = %s)" % name, a, b, stmt="$z = 0", targets=['f"<{$x}>"'])
        if k in ("class", "dataclass", "frozen", "NamedTuple"):
            add("values", "a method (a = %s)" % name, a, b, expr="$x.get_a()")
        if k == "class":
            add("values", "a property (b = %s)" % name, b, a, expr="$x.pb")
        if arr:
            idx = "[0, 0]" if name == "matrix" else "[0]"
            add("values", "a copy writes into the array field (a = %s)" % name, a, b,
                stmt="$y = $x\n%s%s = 9" % (fa("$y"), idx), targets=[fa("$x"), fa("$y")])
            add("values", "a def writes into the array field (a = %s)" % name, a, b,
                extra=["def $w(o):\n    %s%s = -4\n    return 0" % (fa("o"), idx)], stmt="$w($x)", targets=[fa("$x")])
        # a field rebound: in place (Python raises for a frozen dataclass and a named tuple), to a
        # value of another kind (refused: a record's field keeps one type) and of the same kind
        alt = OBJ_ALT[name]
        add("values", "the field rebound (a = %s)" % name, b, a, stmt=setfa("$x", a), targets=[fa("$x"), fb("$x")])
        add("values", "the field rebound to the same kind (a = %s)" % name, a, b, stmt=setfa("$x", alt), targets=[fa("$x"), fb("$x")])
        add("values", "the field rebound through a def (a = %s)" % name, a, b,
            extra=["def $f(o, v):\n    %s\n    return 0" % setfa("o", "v")], stmt="$f($x, %s)" % alt, targets=[fa("$x")])
        if k == "class":
            add("values", "the field rebound through a method (a = %s)" % name, a, b, stmt="$x.set_a(%s)" % alt, targets=[fa("$x")])
        add("values", "the field rebound through a second name (a = %s)" % name, a, b,
            stmt="$y = $x\n%s" % setfa("$y", alt), targets=[fa("$x")])
        if k in ("NamedTuple", "namedtuple"):
            add("values", "_replace (a = %s)" % name, b, b, stmt="$q = $x._replace(a=%s)" % a, targets=["$q"])
            add("values", "unpacking (a = %s)" % name, a, b, stmt="$u, $v = $x", targets=["$u", "$v"])
            add("values", "an index (a = %s)" % name, a, b, expr="$x[0]")
        if k in ("dataclass", "frozen"):
            add("values", "dataclasses.replace (a = %s)" % name, b, b, stmt="$q = replace($x, a=%s)" % a, targets=["$q"])
        if k == "dict":
            add("values", "len, in and get (a = %s)" % name, a, b, stmt="$n = len($x)\n$i = \"a\" in $x\n$g = $x.get(\"c\", 0.5)",
                targets=["$n", "$i", "$g", '$x.get("a")'])
            add("values", "a loop over the items (a = %s)" % name, a, b,
                stmt="$n = 0\nfor $k, $v in $x.items():\n    $n += 1", targets=["$n", "$v"])
    return out


# ================================================================ maps (maps.py)
#
# A dict whose keys are computed at run time is a Viv map. Each way of making one is a
# pseudo-rule, and each key kind goes through every use: store and read, `in`, `len`, `get`,
# `pop`, `setdefault`, `del`, printing, a missing key (KeyError), 0.0 beside -0.0 and 1 beside
# 1.0, removal then reinsertion (the order), a loop over the items, and a loop that changes the
# size (RuntimeError). A NaN key stops with py2viv's message (Python stores it; a map holds no
# NaN), which the verdict STOP records.

MAP_CONSTRUCTS = ("dict", "literal", "comprehension", "defaultdict", "Counter")
# (label, key, a second key of the same kind)
MAP_KEYS = [("int", "3", "-7"), ("zero", "0", "5"), ("2**52", "2 ** 52", "-(2 ** 52)"),
            ("2**53-1", "9007199254740991", "-9007199254740991"), ("float", "1.5", "-0.25"), ("-0.0", "-0.0", "0.5"),
            ("0.0", "0.0", "-2.5"), ("tiny", "1e-300", "5e-324"), ("inf", "np.inf", "-np.inf"),
            ("np.float64", "np.float64(0.1)", "np.float64(-3.0)"), ("nan", "np.nan", "1.0")]


def map_rules():
    return [ObjRule("maps.%s" % c, c) for c in MAP_CONSTRUCTS]


def _map_make(construct, k):
    """The lines that make `$d`, holding 2.5 under the key k (1 for a Counter)."""
    if construct == "dict":
        return ["$d = {}", "$d[%s] = 2.5" % k]
    if construct == "literal":
        return ["$d = {%s: 2.5}" % k, "$j = 0", "$d[$j + 100] = 0.5"]
    if construct == "comprehension":
        return ["$d = {$q: 2.5 for $q in [%s]}" % k]
    if construct == "defaultdict":
        return ["from collections import defaultdict", "$d = defaultdict(float)", "$d[%s] += 2.5" % k]
    return ["from collections import Counter", "$d = Counter([%s])" % k]


def map_cases(rule):
    k = rule.construct
    out = []

    def add(kind, label, setup, expr=None, stmt=None, targets=None, seed=False):
        out.append(ObjCase(rule, k, kind, label, setup, expr, stmt=stmt, seed=seed, targets=targets))

    printable = k in ("dict", "literal", "comprehension")
    for name, a, b in MAP_KEYS:
        mk = _map_make(k, a)
        seed = name in ("int", "float")
        add("seed" if seed else "values", "store and read (key %s)" % name, mk, expr="$d[%s]" % a, seed=seed)
        add("values", "len, in, not in (key %s)" % name, mk, stmt="$n = len($d)\n$i = %s in $d\n$o = %s not in $d" % (a, b),
            targets=["$n", "$i", "$o"])
        add("values", "get with a default (key %s)" % name, mk, stmt="$g = $d.get(%s, -1.0)\n$h = $d.get(%s, -1.0)" % (a, b),
            targets=["$g", "$h"])
        add("values", "a second key, then the keys and values (key %s)" % name, mk,
            stmt="$d[%s] = 4.0\n$ks = list($d)\n$vs = list($d.values())" % b, targets=["$ks", "$vs", "len($d)"])
        if k != "Counter":
            add("values", "a missing key (key %s)" % name, mk, expr="$d[%s] + $d[%s]" % (a, b))
        add("values", "+= under the key (key %s)" % name, mk, stmt="$d[%s] += 1.0" % a, targets=["$d[%s]" % a])
        add("values", "pop, then pop with a default (key %s)" % name, mk, stmt="$p = $d.pop(%s)\n$r = $d.pop(%s, 9.5)" % (a, a),
            targets=["$p", "$r", "len($d)"])
        add("values", "setdefault (key %s)" % name, mk, stmt="$s = $d.setdefault(%s, 7.0)\n$t = $d.setdefault(%s, 8.0)" % (a, b),
            targets=["$s", "$t", "len($d)"])
        add("values", "del, then store again: the key goes last (key %s)" % name, mk,
            stmt="$d[%s] = 1.0\ndel $d[%s]\n$d[%s] = 3.0\n$ks = list($d)" % (b, a, a), targets=["$ks"])
        add("values", "a loop over the items (key %s)" % name, mk,
            stmt="$n = 0.0\nfor $k, $v in $d.items():\n    $n += $v", targets=["$n"])
        add("values", "a loop that adds a key (key %s)" % name, mk,
            stmt="for $k in $d:\n    $d[%s] = 1.0" % b, targets=["len($d)"])
        add("values", "a loop that removes a key (key %s)" % name, mk,
            stmt="$d[%s] = 1.0\nfor $k in $d:\n    $x = $d.pop(%s, 0.0)" % (b, a), targets=["len($d)"])
        if printable:
            add("values", "print the dict (key %s)" % name, mk, stmt="$d[%s] = -1.5" % b, targets=["$d"])
            add("values", "an f-string of the dict (key %s)" % name, mk, stmt="$z = 0", targets=['f"<{$d}>"'])
    # keys that are one key in Python: 0.0 and -0.0 (the first is kept), 1 and 1.0
    for x, y in (("0.0", "-0.0"), ("-0.0", "0.0"), ("1", "1.0"), ("1.0", "1")):
        mk = _map_make(k, x)
        add("values", "the keys %s and %s are one" % (x, y), mk, stmt="$d[%s] = 5.0\n$ks = list($d)" % y,
            targets=["len($d)", "$d[%s]" % x, "$ks"])
        if printable:
            add("values", "print the keys %s and %s" % (x, y), mk, stmt="$d[%s] = 5.0" % y, targets=["$d"])
    return out


# ================================================================ inference (infer.py)
#
# A def's parameters are typed by its calls, and a def called with arguments of several types is
# translated once per tuple of types (infer.py). Each family is a pseudo-rule whose generated
# cases run and are judged as a rule's, with one more demand: the printed text must be CPython's
# byte for byte (an int printed as 3.0 fails the case).
#
#   infer.clone    a def called at two places with a value of each pair of kinds (ints, floats,
#                  -0.0, NaN, infinities, 2**52, Bools, int and float arrays, a matrix, lists of
#                  floats and of ints), for bodies that return the value, double it, add it to
#                  itself, negate it, or print it;
#   infer.list     a def that changes a list parameter (append, pop, insert, extend), called with
#                  new lists of each form (a literal, list(), sorted(), a comprehension, a slice,
#                  a def's new list) and with lists the caller still holds (refused);
#   infer.columns  rows mixing ints and floats: items read by constant and computed indices, rows
#                  looped over, arithmetic on an int column, a store and a print (refused);
#   infer.hints    annotated parameters (int, float, bool, list[float], list[int], np.ndarray)
#                  called with each kind of value;
#   infer.maps     a def called with a dict (each way of making a map, int and float keys) at one
#                  site and with each kind of value at another: bodies that take its length,
#                  loop over it, sum what they iterate, return it, store into it through a
#                  parameter of the clone (the dict's changes seen by the caller), and a
#                  parameter annotated `dict[int, float]` (maps.py's map arguments with
#                  infer.py's clones).

class InferRule(ObjRule):
    pass


class InferCase(ObjCase):
    exact_text = True


def infer_rules():
    return [InferRule("infer.%s" % c, c) for c in ("clone", "list", "columns", "hints", "maps")]


INFER_VALUES = [("int", "3"), ("negint", "-7"), ("2**52", "2 ** 52"), ("float", "1.5"), ("-0.0", "-0.0"),
                ("nan", "np.nan"), ("inf", "np.inf"), ("True", "True"), ("floats", "np.array([1.5, -0.0, 2.25])"),
                ("ints", "np.array([1, -2, 3])"), ("matrix", "np.array([[1.0, 2.0], [3.0, 4.0]])"),
                ("list", "[0.5, 2.0]"), ("intlist", "[1, 2]")]
INFER_BODIES = [("returns it", "return x"), ("doubles it", "return x * 2"), ("adds it to itself", "return x + x"),
                ("negates it", "return -x"), ("prints it", "print(\"x:\", x)\n    return 0")]


def infer_cases(rule):
    k = rule.construct
    out = []

    def add(label, setup, stmt, targets, seed=False):
        out.append(InferCase(rule, k, "seed" if seed else "values", label, list(setup), None, stmt=stmt, seed=seed,
                             targets=targets))

    if k == "clone":
        for bl, body in INFER_BODIES:
            for na, a in INFER_VALUES:
                for nb, b in INFER_VALUES:
                    seed = (na, nb) in (("int", "float"), ("float", "floats"))
                    # the def is part of the case (a refusal in its body is the case's refusal)
                    add("a def that %s, called with %s and %s" % (bl, na, nb), [],
                        "def $f(x):\n    %s\n$p = $f(%s)\n$q = $f(%s)" % (body, a, b), ["$p", "$q"],
                        seed=seed and bl != "prints it")
    elif k == "list":
        bodies = [("appends", "v.append(1.5)\n    return v"), ("pops", "v.pop()\n    return v"),
                  ("inserts", "v.insert(0, 2.5)\n    return len(v)"), ("extends", "v.extend([3.0])\n    return v"),
                  ("pops the first", "w = []\n    w.append(v.pop(0))\n    return w")]
        args = [("a literal", "[1.0, 2.0]", True), ("list(...)", "list([1.0, 4.0])", True),
                ("sorted(...)", "sorted([2.0, 1.0])", True), ("a comprehension", "[t * 2.0 for t in range(3)]", True),
                ("a slice of a list", "$s[1:]", True), ("a def's new list", "$g()", True),
                ("a name the caller prints", "$x", False), ("a NumPy array", "np.array([1.0, 2.0])", False),
                ("a list of ints", "[1, 2]", True)]
        for bl, body in bodies:
            for al, a, fresh in args:
                setup = ["def $g():\n    return [0.5, 0.25]", "$s = [9.0, 8.0, 7.0]", "$x = [1.0, 2.0]"]
                add("a def that %s, called with %s" % (bl, al), setup, "def $f(v):\n    %s\n$r = $f(%s)" % (body, a),
                    ["$r", "$x", "$s"],
                    seed=fresh and bl == "appends" and al == "a literal")
    elif k == "columns":
        rows = [("ints beside floats", "[[1, 2.5], [3, 4.0]]"), ("an int column", "[[3, 0.5], [4, 1.5], [5, -0.0]]"),
                ("a float row", "[[1.5, 2], [2.5, 3]]")]
        reads = [("an item by constant indices", "$r = $M[1][0] * 3", ["$r"]),
                 ("an int column by a computed index", "$r = 0\nfor $i in range(len($M)):\n    $r = $r + $M[$i][0] // 2", ["$r"]),
                 ("a float column", "$r = $M[0][1] + $M[1][1]", ["$r"]),
                 ("rows looped over", "$r = 0.0\nfor $w in $M:\n    $r = $r + $w[1]", ["$r"]),
                 ("the whole list printed", "$r = 0", ["$M"]),
                 ("an int stored into the float column", "$M[0][1] = 7", ["$M[0][1]"]),
                 ("a float stored into the int column", "$M[0][0] = 7.5", ["$M[0][0]"])]
        for rl, lit in rows:
            for dl, stmt, tg in reads:
                add("rows of %s: %s" % (rl, dl), ["$M = %s" % lit], stmt, tg, seed=(rl, dl) == ("an int column", "a float column"))
    elif k == "hints":
        hints = [("int", "int"), ("float", "float"), ("bool", "bool"), ("list[float]", "list[float]"),
                 ("list[int]", "list[int]"), ("np.ndarray", "np.ndarray")]
        for hl, h in hints:
            for na, a in INFER_VALUES:
                add("a parameter annotated %s, called with %s" % (hl, na), [],
                    "def $f(x: %s):\n    return x\n$p = $f(%s)" % (h, a), ["$p"], seed=(hl, na) == ("float", "int"))
    elif k == "maps":
        bodies = [("takes its length", "def $f(x):\n    return len(x)", "$f($d)", "$f(%s)", ["$p", "$q"]),
                  ("counts its items in a loop", "def $f(x):\n    n = 0\n    for k in x:\n        n += 1\n    return n",
                   "$f($d)", "$f(%s)", ["$p", "$q"]),
                  ("sums what it iterates", "def $f(x):\n    s = 0.0\n    for k in x:\n        s += k\n    return s",
                   "$f($d)", "$f(%s)", ["$p", "$q"]),
                  ("returns it", "def $f(x):\n    return x", "$f($d)", "$f(%s)", ["$p", "$q"]),
                  ("stores into the dict under its second argument",
                   "def $f(m, y):\n    m[y] = 2.5\n    return y * 2", "$f($d, 3)", "$f($d, %s)", ["$p", "$q", "len($d)"]),
                  ("is annotated dict[int, float]", "def $f(x: dict[int, float]):\n    return len(x)",
                   "$f($d)", "$f(%s)", ["$p", "$q"])]
        for c in MAP_CONSTRUCTS:
            for kl, key in (("int", "3"), ("float", "1.5")):
                for bl, df, first, second, tg in bodies:
                    for na, a in INFER_VALUES:
                        add("a def that %s, called with a %s (key %s) and with %s" % (bl, c, kl, na), _map_make(c, key),
                            "%s\n$p = %s\n$q = %s" % (df, first, second % a), tg,
                            seed=(c, kl, bl, na) == ("dict", "int", "takes its length", "list"))
    return out


# ================================================================ strings (strs.py)
#
# The string operations py2viv translates in code rather than by a rule (operators, subscripts,
# the text of values, f-strings, str.format and %-formatting, lists of strings, loops, join over
# generators, printing, files) are pseudo-rules `strings.*`. Their cases run each operation on
# every kind of text (STR_VARIANTS: empty, beyond ASCII, non-BMP, combining, every kind of
# space and line break, control characters, quotes) and on the edge numbers (-0.0, NaN, the
# infinities, 1e16, 1e-7, 5e-324, 2**52), and require CPython's printed text byte for byte.

class StrRule(ObjRule):
    pass


class StrCase(ObjCase):
    exact_text = True


STR_CONSTRUCTS = ("operators", "subscripts", "text", "lists", "print", "files")
STR_NUMS = [("-0.0", "-0.0"), ("0.1 + 0.2", "0.1 + 0.2"), ("1e16", "1e16"), ("1e-7", "1e-7"), ("5e-324", "5e-324"),
            ("nan", "float('nan')"), ("inf", "float('inf')"), ("-inf", "-float('inf')"), ("2**52", "2 ** 52"),
            ("-7", "-7"), ("0", "0"), ("True", "True"), ("1.5", "1.5")]


def string_rules():
    return [StrRule("strings.%s" % c, c) for c in STR_CONSTRUCTS]


def string_cases(rule):
    k = rule.construct
    out = []

    def add(label, setup, stmt, targets, seed=False):
        out.append(StrCase(rule, k, "seed" if seed else "values", label, list(setup), None, stmt=stmt, seed=seed,
                           targets=targets))
    texts = [(lab, repr(x)) for lab, x in STR_VARIANTS]
    if k == "operators":
        for lab, a in texts:
            seed = lab == "accented"
            add("+, * and len of the string %s" % lab, ["$s = %s" % a],
                "$u = $s + \"|\" + $s\n$v = $s * 3 + $s * 0 + $s * -2\n$n = len($s) + len($u)", ["$u", "$v", "$n"], seed=seed)
            add("comparisons of the string %s" % lab, ["$s = %s" % a, "$t = 'héllo'"],
                "$c = [$s < $t, $s <= $t, $s == $t, $s != $t, $s > $t, $s >= $t, $s == $s, $s < $s + 'a']", ["$c"])
            add("in and truth of the string %s" % lab, ["$s = %s" % a],
                "$c = ['a' in $s, $s in 'xx' + $s, '' in $s, 'zz' not in $s, len($s) > 0]\n"
                "$e = 'yes' if $s else 'no'", ["$c", "$e"])
            add("+= and *= of the string %s" % lab, ["$s = %s" % a], "$u = $s\n$u += '!'\n$u *= 2", ["$u", "$s"])
    elif k == "subscripts":
        for lab, a in texts:
            seed = lab == "accented"
            add("indices of the string %s" % lab, ["$s = %s" % a, "$k = 1"],
                "$c = [$s[0], $s[-1], $s[$k], $s[-$k]]", ["$c"], seed=seed)
            add("slices of the string %s" % lab, ["$s = %s" % a, "$k = 2"],
                "$c = [$s[1:3], $s[::-1], $s[::2], $s[-2:], $s[5:1:-1], $s[:100], $s[-100:], $s[$k:$k + 3], $s[::-$k], "
                "$s[3:], $s[:-1], $s[100:-100:-1], $s[None:2]]", ["$c"], seed=seed)
            add("an index past the end of the string %s" % lab, ["$s = %s" % a], "$c = $s[len($s)]", ["$c"])
            add("a zero step on the string %s" % lab, ["$s = %s" % a, "$z = 0"], "$c = $s[::$z]", ["$c"])
            add("the characters of the string %s in a loop" % lab, ["$s = %s" % a],
                "$c = []\nfor $ch in $s:\n    $c.append($ch + '.')\n$d = [$ch2 * 2 for $ch2 in reversed($s)]\n"
                "$e = ''.join(c + str(i) for i, c in enumerate($s))", ["$c", "$d", "$e"])
    elif k == "text":
        for lab, a in STR_NUMS:
            seed = lab == "1.5"
            add("str, repr and f-strings of %s" % lab, ["$x = %s" % a],
                "$t = str($x) + '|' + repr($x) + '|' + f'{$x}|{$x!r}|{$x!s}|{$x=}'", ["$t"], seed=seed)
            if lab != "True":
                add("format specs of %s" % lab, ["$x = %s" % a],
                    "$t = f'{$x:>12}|{$x:*<12}|{$x:^12}|{$x:12}|{$x:.3e}|{$x:10.2f}|{$x:g}|{$x:.4g}' + format($x, '>9')",
                    ["$t"])
                add("%%-formatting of %s" % lab, ["$x = %s" % a],
                    "$t = '%s|%r|%e|%10.3f|%-10.2f|%g|%d' % ($x, $x, $x, $x, $x, $x, $x)", ["$t"])
                add("str.format of %s" % lab, ["$x = %s" % a],
                    "$t = '{}|{!r}|{:>10}'.format($x, $x, $x) + '{0}|{0!r}|{0:>10}|{0:.2e}'.format($x)", ["$t"])
        for lab, a in texts:
            add("str, repr and specs of the string %s" % lab, ["$s = %s" % a],
                "$t = str($s) + '|' + repr($s) + f'|{$s}|{$s!r}|{$s:>12}|{$s:*^9}|{$s:.3}|{$s:<4.2}|' + format($s, '_>6')",
                ["$t"])
            add("str.format and %%-formatting of the string %s" % lab, ["$s = %s" % a],
                "$t = '[{}] [{!r}] [{:>8}]'.format($s, $s, $s) + '<%s|%r|%6s|%-6s|%.2s>' % ($s, $s, $s, $s, $s)", ["$t"])
        add("the text of lists", ["$v = [1.5, -0.0, 2.0]", "$m = [1, -2, 30]", "$w = ['a', \"b'c\", 'é']"],
            "$t = str($v) + str($m) + repr($w) + str($w) + f'{$v}{$m!r}{$w}'", ["$t"], seed=True)
    elif k == "lists":
        for lab, a in texts:
            seed = lab == "separators"
            add("split, join and sorted of the string %s" % lab, ["$s = %s" % a],
                "$w = $s.split(',')\n$v = $s.split()\n$j = '|'.join($w) + '/' + ' '.join(sorted($v))\n$r = sorted($w, reverse=True)",
                ["$w", "$v", "$j", "$r"], seed=seed)
            add("a list grown from the string %s" % lab, ["$s = %s" % a],
                "$w = []\nfor $p in $s.split():\n    $w.append($p.upper())\n$w.append($s)\n"
                "$n = len($w)\n$f = $w[0]\n$l = $w[-1]\n$i = $w.index($s)\n$c = $w.count($s)\n$h = $s in $w",
                ["$w", "$n", "$f", "$l", "$i", "$c", "$h"])
            add("list methods beside the string %s" % lab, ["$s = %s" % a],
                "$w = ['b', 'a']\n$w.insert(1, $s)\n$w.append('z')\n$w.sort()\n$x = $w.pop()\n$w.reverse()\n"
                "$w.extend([$s, 'q'])\n$w.remove($s)\n$y = $w.pop(0)",
                ["$w", "$x", "$y"])
            add("a comprehension over the string %s" % lab, ["$s = %s" % a],
                "$w = [c for c in $s if c != ' ']\n$u = [c.upper() + str(i) for i, c in enumerate($s)]\n"
                "$j = ', '.join(repr(c) for c in $s)\n$m = list($s)", ["$w", "$u", "$j", "$m"])
        add("join of numbers written as text", ["$v = [1.5, -0.0, 2.0, float('nan')]", "$m = [3, -1, 2 ** 52]"],
            "$a = ', '.join(str(x) for x in $v)\n$b = '-'.join(map(str, $m))\n$c = ''.join([f'{x:.1f}' for x in $v])",
            ["$a", "$b", "$c"], seed=True)
    elif k == "print":
        for lab, a in texts:
            seed = lab == "accented"
            add("printing the string %s" % lab, ["$s = %s" % a],
                "print($s)\nprint($s, 3, $s, sep='|')\nprint(repr($s), [$s], $s + 'x')\nprint(f'[{$s}]', 2.5, sep='')", ["$s"],
                seed=seed)
    elif k == "files":
        for i, (lab, a) in enumerate(texts):
            seed = lab == "accented"
            add("a file written and read back holding the string %s" % lab, ["from pathlib import Path", "$s = %s" % a],
                "$n = Path('$f%d.txt').write_text($s + '\\n' + $s)\n$t = open('$f%d.txt').read()\n"
                "$l = open('$f%d.txt').readlines()\n$m = [x for x in open('$f%d.txt')]" % (i, i, i, i),
                ["$n", "$t", "$l", "$m"], seed=seed)
            add("a file written in a with block holding the string %s" % lab, ["$s = %s" % a],
                "with open('$g%d.txt', 'w') as $fh:\n    $fh.write($s)\n    $fh.write('\\n' + $s + '\\n')\n"
                "with open('$g%d.txt') as $fr:\n    $t = $fr.read()" % (i, i), ["$t"], seed=seed)
    return out


def python_outcome(case):
    """Run the case's Python in this process: ("ok", printed values) or ("raise", line)."""
    import numpy as np
    src = PRELUDE + "\n".join(case.lines("c_")) + "\n"
    printed = []
    g = {"print": lambda *a, **k: printed.append(a)}
    here = os.getcwd()
    work = tempfile.mkdtemp(prefix="case_")  # a case may write files (strings.files): not into the checkout
    os.chdir(work)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                exec(compile(src, "<case>", "exec"), g)
    except BaseException as e:  # noqa: BLE001
        import traceback
        line = traceback.format_exception_only(type(e), e)[-1].rstrip("\n")
        return ("raise", line, type(e).__name__)
    finally:
        os.chdir(here)
        shutil.rmtree(work, ignore_errors=True)
    return ("ok", printed[-1] if printed else (), None)


def type_of(v):
    """The static type py2viv should give a Python value: (kind, py, rank, container)."""
    import numpy as np
    if isinstance(v, (bool, np.bool_)):
        return ("bool", None, 0, None)
    if isinstance(v, (int, np.integer)):
        return ("num", "int", 0, None)
    if isinstance(v, (float, np.floating)):
        return ("num", "float", 0, None)
    if isinstance(v, (complex, np.complexfloating)):
        return ("cnum", None, 0, None)
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return type_of(v[()])
        k = v.dtype.kind
        if k == "c":
            return ("carr", None, v.ndim, "nd")
        return ("arr", {"f": "float", "i": "int", "u": "int", "b": "bool"}.get(k, k), v.ndim, "nd")
    if isinstance(v, str):
        return ("str", None, 0, None)
    if isinstance(v, list) and (not v or all(isinstance(e, str) for e in v)) and (not v or isinstance(v[0], str)):
        return ("strs", None, 0, None) if v else ("emptylist", None, 1, "list")
    if isinstance(v, list):
        rank = 1
        x = v
        while x and isinstance(x[0], list):
            rank += 1
            x = x[0]
        flat = list(_flat(v))
        if all(isinstance(e, (bool, np.bool_)) for e in flat):
            py = "bool"
        elif all(isinstance(e, (int, np.integer)) and not isinstance(e, bool) for e in flat):
            py = "int"
        elif any(isinstance(e, (complex, np.complexfloating)) for e in flat):
            return ("carr", None, rank, "list")
        else:
            py = "float"
        return ("arr", py, rank, "list")
    if isinstance(v, tuple):
        return ("tuple", None, len(v), None)
    return (type(v).__name__, None, 0, None)


def static_type(tj):
    k = tj.get("kind")
    if k == "num":
        return ("num", tj.get("py"), 0, None)
    if k in ("bool", "cnum"):
        return (k, None, 0, None)
    if k in ("arr", "carr"):
        cont = tj.get("container")
        return (k, tj.get("dtype") if k == "arr" else None, tj.get("rank"), "list" if cont == "list" else "nd")
    return (k, None, 0, None)


def type_matches(pyt, st):
    if pyt == st:
        return True
    if pyt[0] == "emptylist":
        return st[0] == "strs" or st[0] in ("arr", "carr") and st[3] == "list"  # [] of any kind
    # a shape prints as a tuple; an np.float64 list entry is a float
    if st[0] == "arr" and st[3] == "nd" and pyt[0] == "tuple":
        return True
    return False


def translate_case(case):
    """Translate the case alone: ("ok", res) or ("refused", message, located at the call) or
    ("crash", text)."""
    import py2viv
    lines = PRELUDE.splitlines() + case.lines("c_")
    src = "\n".join(lines) + "\n"
    npre = len(PRELUDE.splitlines())
    stmt_lines, after = case.call_lines()
    stmt_lines = {l + npre for l in stmt_lines}
    after = {l + npre for l in after}  # a write after it
    call_line = min(stmt_lines)  # the r = ... line (or the statement's first), 1-based
    try:
        res = py2viv.translate(src, "case.py")
    except Exception as e:  # noqa: BLE001
        import traceback
        return ("crash", "%s: %s" % (type(e).__name__, e), traceback.format_exc(limit=3))
    if res["refusals"]:
        lines_of = [r.where()[0] for r in res["refusals"] if not getattr(r, "cascade", False)]
        whats = [r.what for r in res["refusals"] if not getattr(r, "cascade", False)]
        setup_lines = set(range(npre + 1, call_line))
        if any(l in setup_lines for l in lines_of):
            return ("setup", res["messages"][0])
        if stmt_lines & set(lines_of):
            i = next(k for k, l in enumerate(lines_of) if l in stmt_lines)
            return ("refused", res["messages"][i], whats[i])
        if after & set(lines_of):
            i = next(k for k, l in enumerate(lines_of) if l in after)
            return ("refused", res["messages"][i], "(a line after it) " + whats[i])
        if max(stmt_lines | after) + 1 in lines_of:
            i = lines_of.index(max(stmt_lines | after) + 1)
            return ("refused", res["messages"][i], "(its print) " + whats[i])
        return ("refused-elsewhere", res["messages"][0], whats[0] if whats else "")
    return ("ok", res)


# ================================================================ the differential runs


def _cache_key(py_src, tol, tools):
    h = hashlib.sha256()
    h.update(py_src.encode())
    h.update(repr(tol).encode())
    h.update(tools.encode())
    return h.hexdigest()


def tools_id():
    import diff
    h = hashlib.sha256()
    for p in (diff.default_viv(), diff.default_wasm()):
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    with open(os.path.join(HERE, "diff.py"), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run_program(py_src, tol, workdir, tools):
    """diff.py's differential run of one program (cached). Returns the report's essentials."""
    import diff
    import py2viv
    # the key: the program, its translation (so a rule change reruns only what it changed),
    # the tolerance, and the tools
    key = _cache_key(py_src + "\n#viv\n" + py2viv.translate(py_src, "case.py")["viv"], tol, tools)
    cpath = os.path.join(CACHE, key + ".json")
    if os.path.exists(cpath):
        try:
            return json.load(open(cpath))
        except ValueError:
            pass
    d = tempfile.mkdtemp(prefix="rule_", dir=workdir)
    path = os.path.join(d, "case.py")
    with open(path, "w") as f:
        if tol:
            f.write("# py2viv: rtol=%r atol=%r\n" % tol)
        f.write(py_src)
    rep = diff.differential(path, os.path.join(d, "out"), timeout=120)
    ran = [e for e, r in rep.get("engines", {}).items() if not r.get("skipped")]
    all_stopped = bool(ran) and all(rep["engines"][e]["ran"] and rep["engines"][e]["exit"] not in (0, None) for e in ran)
    stop_lines = {}
    for e in ran:
        p_ = os.path.join(d, "out", e + ".out")
        if os.path.exists(p_):
            err = open(p_).read().split("\n--- stderr ---\n")
            stop_lines[e] = (err[1].splitlines() or [""])[0][:160] if len(err) > 1 else ""
    out = {"verdict": rep["verdict"], "stop": rep.get("stop"), "engines_agree": rep.get("engines_agree"),
           "all_stopped": all_stopped, "stop_lines": stop_lines,
           "engine_diffs": rep.get("engine_diffs"), "values": [
               {"line": v["line"], "verdict": v["verdict"], "python": v["python_first"], "viv": v["viv_first"],
                "why": v.get("why")} for v in rep.get("values", []) if v["verdict"] != "bitwise"],
           "messages": rep.get("messages"),
           "text_bad": [[r.get("python"), r.get("viv")] for r in rep.get("lines", []) if r.get("verdict") != "identical"][:3]}
    shutil.rmtree(d, ignore_errors=True)
    # a timeout says how busy the machine was, not what the program does, and so does an
    # engine killed from outside (SIGTERM, SIGKILL: exit -15, -9): it is reported (a FAIL)
    # but never cached, so the next run tries the case again
    seen = json.dumps([out["engine_diffs"], out["stop_lines"]])
    killed = any(r.get("exit") in (-15, -9) for r in rep.get("engines", {}).values()) or \
        "-15]" in seen or "-9]" in seen
    if "timeout after" not in seen and not killed:
        os.makedirs(CACHE, exist_ok=True)
        with open(cpath, "w") as f:
            json.dump(out, f)
    return out


def judge_run(case, rep):
    v = rep["verdict"]
    stop = rep.get("stop") or {}
    viv_stop = stop.get("viv") or ""
    if getattr(case, "type_error", None) and not viv_stop.startswith("py2viv:"):
        return "FAIL", case.type_error + ("" if v.startswith("PASS") else "; " + v)
    if v.startswith("PASS"):
        if case.py[0] == "raise":
            return "PASS", "stops as Python: %s" % case.py[1]
        if getattr(case, "exact_text", False) and rep.get("text_bad", []):
            # the inference cases: the printed text must be CPython's byte for byte (an int
            # printed as 3.0 is a wrong translation even though the number is equal)
            return "FAIL", "text differs: %s" % rep["text_bad"]
        return "PASS", v
    if viv_stop.startswith("py2viv:") and rep.get("engines_agree"):
        return "STOP", viv_stop
    if case.py[0] == "raise" and rep.get("all_stopped"):
        # Python raises, and every engine stops the run (with Viv's own error line)
        return "STOP", "Python: %s; Viv stops: %s" % (case.py[1], rep.get("stop_lines"))
    detail = v
    if rep.get("values"):
        detail += "; " + "; ".join("line %s: python %s viv %s %s" % (x["line"], x["python"], x["viv"], x.get("why") or "")
                                   for x in rep["values"][:3])
    if rep.get("engine_diffs"):
        detail += "; engines: %s" % json.dumps(rep["engine_diffs"])[:300]
    if stop:
        detail += "; ends: python %r, viv %r" % (stop.get("python"), stop.get("viv"))
    return "FAIL", detail


def expected_refusal(rule, sp):
    msg = next((c["refuse"] for c in rule.cases if "refuse" in c), "")
    short = sp.split(".")[-1]
    return msg.replace("{name}", sp).replace("{short}", short)


def evaluate_rule(rule, cases, workdir, tools, log):
    """Classify every case of one rule; run the translatable ones."""
    batch = {}  # tolerance -> [case]
    absent = set()
    for c in cases:
        # a refuse rule's cases are only translated (running `input()` or `open(...)` here
        # would block or touch files)
        c.py = python_outcome(c) if rule.exactness != "refuse" else ("ok", (), None)
        if c.seed and c.py[0] == "raise" and c.py[2] == "AttributeError" and "module 'numpy' has no attribute" in c.py[1]:
            absent.add(c.spelling)
    for c in cases:
        if c.spelling in absent:
            c.verdict, c.detail = "SKIP", "not in this NumPy (%s)" % (__import__("numpy").__version__)
            continue
        tr = translate_case(c)
        if tr[0] == "crash":
            c.verdict, c.detail = "FAIL", "the translator crashed: %s" % tr[1]
            continue
        if tr[0] == "setup":
            c.verdict, c.detail = "SKIP", "setup not translatable: %s" % tr[1].splitlines()[0]
            continue
        if rule.exactness == "refuse":
            if tr[0] == "refused" and tr[2] == expected_refusal(rule, c.spelling):
                c.verdict, c.detail = "REFUSED", tr[2]
            elif tr[0] == "refused" and "{" in expected_refusal(rule, c.spelling):
                c.verdict, c.detail = "REFUSED", tr[2]
            else:
                c.verdict = "FAIL"
                c.detail = "expected the refusal %r, got %s" % (expected_refusal(rule, c.spelling),
                                                                 tr[2] if tr[0] != "ok" else "a translation")
            continue
        if tr[0] == "refused-elsewhere":
            c.verdict, c.detail = "FAIL", "refused away from the call: %s" % tr[1].splitlines()[0]
            continue
        if tr[0] == "refused":
            if (c.seed or c.expect == "pass") and c.py[0] == "ok":
                c.verdict, c.detail = "FAIL", "%s is refused: %s" % ("the rule's own test" if c.seed else "a use that is exact", tr[2])
            else:
                c.verdict, c.detail = "REFUSED", tr[2]
            continue
        res = tr[1]
        c.inexact = bool(res["inexact"])
        # the result's static type against Python's
        if c.py[0] == "ok" and c.stmt is None:
            sites = res["sites"]
            if sites:
                parts = [p for p in sites[-1]["parts"] if p["kind"] == "val"]
                if parts and c.py[1]:
                    pyt = type_of(c.py[1][0])
                    st = static_type(parts[0]["type"])
                    if not type_matches(pyt, st):
                        # run it anyway: a run-time refusal (STOP) answers it; anything else fails
                        c.type_error = "type: Python %s, translation %s" % (pyt, st)
                        c.solo = True
        if c.py[0] == "raise":
            c.solo = True
        tol = (rule_tol(rule, res) if c.inexact else None)
        if tol and c.kind == "expression" and c.expr.endswith(" * 3"):
            tol = (tol[0] + 2.3e-16, tol[1] * 3)  # the product's own rounding
        batch.setdefault(tol, []).append(c)
    # runs: one program per tolerance class; a failing batch is rerun case by case
    jobs = []
    for tol, cs in batch.items():
        solo = [c for c in cs if getattr(c, "solo", False)]
        group = [c for c in cs if not getattr(c, "solo", False)]
        for c in solo:
            jobs.append((tol, [c]))
        if group:
            jobs.append((tol, group))
    return jobs


def rule_tol(rule, res):
    """The tolerance a program using this rule declares: the rule's, and any case override's."""
    rtol = rule.rtol or 0.0
    atol = rule.atol or 0.0
    for c in rule.cases:
        if "rtol" in c:
            rtol = max(rtol, c["rtol"])
            atol = max(atol, c["atol"])
    if rtol == 0.0 and atol == 0.0:
        # an exact rule whose case records an inexact operation it borrowed (a complex abs)
        rtol = 2.2e-16
    return (rtol, atol)


def program_of(cases):
    lines = []
    for k, c in enumerate(cases):
        lines.extend(c.lines("c%d_" % k))
    return PRELUDE + "\n".join(lines) + "\n"


def run_job(job, workdir, tools):
    tol, cases = job
    rep = run_program(program_of(cases), tol, workdir, tools)
    text_ok = not (rep.get("text_bad") and any(getattr(c, "exact_text", False) for c in cases))
    if rep["verdict"].startswith("PASS") and text_ok or len(cases) == 1:
        for c in cases:
            c.verdict, c.detail = judge_run(c, rep)
        return
    for c in cases:
        r1 = run_program(program_of([c]), tol, workdir, tools)
        c.verdict, c.detail = judge_run(c, r1)


# ================================================================ driver


def have_tools():
    import diff
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "NumPy"
    if not os.path.exists(diff.default_viv()):
        return "viv (VIV=<path>)"
    if not diff.default_wasm():
        return "viv.wasm (VIV_WASM=<path>)"
    if not shutil.which("node"):
        return "node"
    if not shutil.which(os.environ.get("CC", "cc")):
        return "a C compiler"
    return None


def differential(engine, only=None, jobs_n=4, log=print):
    import numpy as np
    rules = [r for r in list(engine.rules) + object_rules() + map_rules() + infer_rules() + string_rules()
             if only is None or only in r.id]
    gen = Gen(rules, np)
    tools = tools_id()
    workdir = tempfile.mkdtemp(prefix="check_rules_")
    all_cases = {}
    jobs = []
    for r in rules:
        cs = gen.cases(r)
        all_cases[r.id] = cs
        jobs.extend(evaluate_rule(r, cs, workdir, tools, log))
    log("%d rules, %d cases generated; %d differential runs (batched; cached in %s)" % (
        len(rules), sum(len(c) for c in all_cases.values()), len(jobs), os.path.relpath(CACHE, ROOT)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_n) as ex:
        futs = [ex.submit(run_job, j, workdir, tools) for j in jobs]
        for k, f in enumerate(concurrent.futures.as_completed(futs)):
            f.result()
            if (k + 1) % 50 == 0:
                log("  %d/%d runs" % (k + 1, len(jobs)))
    shutil.rmtree(workdir, ignore_errors=True)
    return rules, all_cases


def summarize(rules, all_cases):
    summary = {}
    lines = []
    for r in rules:
        cs = all_cases[r.id]
        cnt = {}
        by_kind = {}
        for c in cs:
            v = c.verdict or "FAIL"
            cnt[v] = cnt.get(v, 0) + 1
            by_kind.setdefault(c.kind, {}).setdefault(v, 0)
            by_kind[c.kind][v] += 1
        seeds_ok = all(c.verdict in ("PASS", "STOP", "SKIP") or (c.verdict == "REFUSED" and c.py and c.py[0] == "raise")
                       for c in cs if c.seed)
        if r.exactness != "refuse" and not any(c.seed and c.verdict == "PASS" for c in cs):
            seeds_ok = False
        summary[r.id] = {"exactness": r.exactness, "cases": len(cs), "pass": cnt.get("PASS", 0),
                         "refused": cnt.get("REFUSED", 0), "stop": cnt.get("STOP", 0), "skip": cnt.get("SKIP", 0),
                         "fail": cnt.get("FAIL", 0) + (0 if seeds_ok else 1), "by_kind": by_kind}
        lines.append("%-28s %-9s %4d cases: %4d pass, %4d refused, %3d stop, %3d skip, %3d FAIL%s" % (
            r.id, r.exactness, len(cs), cnt.get("PASS", 0), cnt.get("REFUSED", 0), cnt.get("STOP", 0),
            cnt.get("SKIP", 0), cnt.get("FAIL", 0), "" if seeds_ok else "  (no passing seed test)"))
        for c in cs:
            if c.verdict == "FAIL" or c.verdict is None:
                lines.append("    FAIL %s [%s] %s: %s\n         %s" % (c.spelling, c.kind, c.label,
                                                                    (c.stmt or c.expr or "").replace("@", ""), c.detail[:600]))
    return summary, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Every py2viv rule proves itself (see the docstring).")
    ap.add_argument("--run", action="store_true", help="run the differential tests and write the ledger")
    ap.add_argument("--only", help="the differential tests of rules whose id contains this (no ledger)")
    ap.add_argument("--write", action="store_true", help="regenerate SUBSET.md's rule tables")
    ap.add_argument("--static", action="store_true", help="the static checks and the ledger only")
    ap.add_argument("-j", type=int, default=4, help="parallel runs (default 4)")
    ap.add_argument("--verbose", action="store_true", help="list every case")
    args = ap.parse_args(argv)
    errs = []
    try:
        rule_engine, engine = load_engine()
    except Exception as e:  # noqa: BLE001 - a malformed rule file is the first failure
        print("rules: %s" % e)
        return 1
    print("rules: %d load (%s)" % (len(engine.rules), ", ".join("%s %d" % kv for kv in sorted(engine.summary().items()))))
    errs += check_tests_and_params(engine.rules)
    perrs, ntpl = check_precedence(engine.rules)
    errs += perrs
    print("precedence audit: %d templates, %d problems" % (ntpl, len(perrs)))
    errs += check_subset(engine, args.write)
    missing = have_tools()
    run = args.run or args.only or (not args.static and missing is None)
    if run:
        if missing:
            print("the differential tests need %s" % missing)
            return 1
        rules, all_cases = differential(engine, args.only, args.j)
        summary, lines = summarize(rules, all_cases)
        total = {k: sum(s[k] for s in summary.values()) for k in ("cases", "pass", "refused", "stop", "skip", "fail")}
        print("\n".join(lines))
        print("differential: %(cases)d cases: %(pass)d pass, %(refused)d refused, %(stop)d stop at run time, "
              "%(skip)d skipped, %(fail)d FAIL" % total)
        if args.verbose:
            for r in rules:
                for c in all_cases[r.id]:
                    print("  %-7s %-24s %-10s %-40s %s" % (c.verdict, r.id, c.kind, c.label[:40], c.detail[:120]))
        if not args.only:
            import numpy
            led = {"inputs": inputs_hash(), "python": sys.version.split()[0], "numpy": numpy.__version__,
                   "fail": total["fail"], "totals": total, "rules": summary}
            with open(LEDGER, "w") as f:
                json.dump(led, f, indent=1, sort_keys=True)
            with open(REPORT, "w") as f:
                f.write("check_rules.py: the generated differential test of every rule (Python %s, NumPy %s)\n\n"
                        % (led["python"], led["numpy"]))
                f.write("\n".join(lines) + "\n\n")
                f.write(by_class(summary) + "\n")
            print(by_class(summary))
        if total["fail"]:
            errs.append("%d generated case(s) fail" % total["fail"])
    else:
        if missing and not args.static:
            print("differential tests not run here (needs %s): checking the ledger" % missing)
        errs += check_ledger()
    for e in errs:
        print("ERROR: " + e)
    print("check_rules: %s" % ("OK" if not errs else "%d problem(s)" % len(errs)))
    return 1 if errs else 0


def by_class(summary):
    rows = {}
    for rid, s in summary.items():
        row = rows.setdefault(s["exactness"], {"rules": 0, "cases": 0, "pass": 0, "refused": 0, "stop": 0, "skip": 0, "fail": 0})
        row["rules"] += 1
        for k in ("cases", "pass", "refused", "stop", "skip", "fail"):
            row[k] += s[k]
    out = ["per rule class:", "%-10s %5s %6s %6s %8s %5s %5s %5s" % ("class", "rules", "cases", "pass", "refused", "stop", "skip", "FAIL")]
    for k in ("exact", "tolerance", "refuse"):
        if k in rows:
            r = rows[k]
            out.append("%-10s %5d %6d %6d %8d %5d %5d %5d" % (k, r["rules"], r["cases"], r["pass"], r["refused"], r["stop"], r["skip"], r["fail"]))
    kinds = {}
    for s in summary.values():
        for kind, cnt in s["by_kind"].items():
            d = kinds.setdefault(kind, {})
            for v, n in cnt.items():
                d[v] = d.get(v, 0) + n
    out.append("")
    out.append("per kind of generated case:")
    for kind in ("seed", "expression", "values", "keywords", "arity", "refuse", "seeds", "interleave", "sizes", "tail"):
        if kind in kinds:
            d = kinds[kind]
            out.append("  %-10s %5d: %s" % (kind, sum(d.values()), ", ".join("%s %d" % kv for kv in sorted(d.items()))))
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())

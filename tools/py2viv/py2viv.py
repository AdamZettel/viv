#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv: translate a numeric-Python subset to Viv source.

    python3 tools/py2viv/py2viv.py file.py > file.viv
    python3 tools/py2viv/py2viv.py file.py -o file.viv --manifest file.py2viv.json

The subset, and what to write instead of everything outside it, is in
tools/py2viv/SUBSET.md. Anything outside the subset is refused with the
Python file, line and column, a caret under the construct, and a hint. The
translator collects every refusal in the file before it stops, so one run
lists all of them.

Standard library only: CPython's own `ast` parses the file, so the translator
accepts exactly the Python the running interpreter accepts.

The contract: a translated program behaves exactly as CPython (and NumPy) run the
Python, or py2viv refuses it with a location. Where Python raises and a double
would not (a zero divisor, a math domain error, an int past 2^53), the translation
calls a guard that stops the run with the traceback's last line. Operations whose
last bits come from a different library (libm, LAPACK, BLAS, the FFT) are kept and
listed in the manifest's `inexact`; diff.py compares bit for bit unless the program
states a tolerance (`# py2viv: rtol=...`).

How printing works. Each Python `print(...)` becomes one Viv `print(...)`: the
text, and each value as `fmt` writes it, with the space
Viv's print puts between arguments where Python's does, and `sep=""` where
Python writes none (`print_call`): a float as
Python's repr, an int with "d", a format spec of the supported table as
itself. So for numbers, lists and supported specs the Viv output is CPython's
text byte for byte. A bool prints `true`, a complex number `(re,im)`, a NumPy
array as a list: diff.py reads those back and prints them as Python would. The
manifest (`--manifest`) lists each print site's parts, so the line can be read
back. `--print plain` emits `print`/`show` of six digits, for reading.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_engine  # noqa: E402
import views as views_mod  # noqa: E402
import classes as classes_mod  # noqa: E402  (classes, dataclasses, named tuples, dicts: Viv records)
import funcs  # noqa: E402  (functions and lists: comprehensions, lambdas, nested defs, list methods)
import maps as maps_mod  # noqa: E402  (dicts with keys computed at run time: Viv maps)
import infer  # noqa: E402  (defs typed by their call sites, specialized per argument types)
import strs as strs_mod  # noqa: E402  (Python's str and lists of str: Viv strings)
import ints as ints_mod  # noqa: E402  (int annotations as Viv Ints, enum.Enum, Optional)
py_funcs = funcs  # (grow_refusal's parameter `funcs` hides the module)

VERSION = "0.2.0"

# ---------------------------------------------------------------- names

VIV_KEYWORDS = {
    "def", "return", "if", "else", "while", "for", "to", "break", "continue", "narrate", "show",
    "test", "rung", "parallel", "fn", "record", "and", "or", "not", "true", "false", "from", "by",
    "reversed", "in",
}

# Builtins of Viv. A Python name equal to one of these is renamed (name + "_") so that a
# builtin the translation emits is never captured by a user def or variable.
VIV_BUILTINS = set("""
exp log ln sqrt pow abs fabs sin cos tan asin acos atan atan2 sinh cosh tanh erf gamma floor ceil
round min max mod band bor bxor bnot shl shr sar u32 i32 zeros len linspace fill sort transpose sum
histogram roll tridiag_solve all any where slab setslab take put eig_sym spectrum_sym lin_solve
lin_solve_multi cholesky eig_general eig_near simpson trapezoid deriv apply release_mode complex re
im conj czeros fft ifft fft3 ifft3 eig_herm srand rand plot scatter bars plot2d plot3d quiver scene
wire verts figure xtick xscale yscale aspect fill_between rangebars grid share yright style ytick
origin colormap clim levels checkpoint stage axes curve dots circle rect segment poly frame label
hline vline draw fade_in fade_out morph move_to say wait together apart print assert_near
assert_true assert_gt assert_lt assert_allclose numel shape grad value_and_grad integrate
find_root text stop fmt copy concat hcat vcat eye diag outer trace
""".split())

MAX_EXACT_INT = 2 ** 53


def _printable(ch):
    """Written as itself in a Viv literal: a printable character, by Python's rule. The source
    stays readable, and no character the lexer might read as a line break (U+2028, U+0085) or
    that an editor hides is written raw."""
    return ch == " " or ch.isprintable()


def viv_str(text):
    """A Viv string literal of any text (the lexer's escapes: \\\\ \\" \\n \\t \\r and \\u{...}).
    A lone surrogate, which a Viv string (UTF-8) cannot hold, raises ValueError."""
    out = ['"']
    for ch in text:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif 0xD800 <= o <= 0xDFFF:
            raise ValueError("a lone surrogate (U+%04X)" % o)
        elif _printable(ch):
            out.append(ch)
        else:
            out.append("\\u{%x}" % o)
    out.append('"')
    return "".join(out)


_VIV_LIT = re.compile(r'"(?:[^"\\]|\\.)*"')


def _sep_simulate(items):
    """Viv's parser on print(a, b, ...) (lang/viv/src/parser.rs, `print_call`):
    the space between two arguments joins the text before it, else the text
    after it, else stands alone. items are (is_text, text or code)."""
    out = []
    for k, (lit, c) in enumerate(items):
        if k > 0:
            if out and out[-1][0]:
                out[-1] = (True, out[-1][1] + " ")
            elif lit:
                c = " " + c
            else:
                out.append((True, " "))
        out.append((lit, c))
    return out


def print_call(args):
    """The Viv `print` call that writes `args` one after another: Viv's print
    puts a space between its arguments, so a space is taken out of the text at
    each junction (Python's own separator, usually), and a call whose
    junctions have none keeps its arguments with sep="" (tools/migrate_print_sep.py
    is the same rule)."""
    old = [(True, unquote_viv(a)) if _VIV_LIT.fullmatch(a) else (False, a) for a in args]
    plan, ok, k = [], True, 0  # [is_text, text, kept]
    while k < len(old):
        lit, c = old[k]
        entry = [lit, c, True]
        k += 1
        kept = [p for p in plan if p[2]]
        if not kept:
            plan.append(entry)
            continue
        last = kept[-1]
        if last[0] and last[1].endswith(" "):
            last[1] = last[1][:-1]
            if last[1] == "" and not lit and not (len(kept) >= 2 and kept[-2][0]):
                last[2] = False
        elif not last[0] and lit and c == " " and k < len(old) and not old[k][0]:
            entry[2] = False
            plan.extend([entry, [False, old[k][1], True]])
            k += 1
            continue
        elif not last[0] and lit and c.startswith(" "):
            entry[1] = c[1:]
        else:
            ok = False
            break
        plan.append(entry)
    if ok:
        new = [(p[0], p[1]) for p in plan if p[2]]
        if _sep_simulate(new) == old:
            return "print(%s)" % ", ".join(viv_str(c) if lit else c for lit, c in new)
    kept = [a for a in args if a != '""'] or ['""']
    if len(kept) == 1:
        return "print(%s)" % kept[0]
    return 'print(%s, sep="")' % ", ".join(kept)


def unquote_viv(lit):
    """The text of a Viv string literal viv_str wrote."""
    body = lit[1:-1]
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            c = body[i + 1]
            if c == "u" and body[i + 2:i + 3] == "{":
                end = body.index("}", i + 3)
                out.append(chr(int(body[i + 3:end], 16)))
                i = end + 1
                continue
            out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(c, c))
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def _host_error(src):
    """The last line of the traceback this Python prints for `src` (evaluated here), which a
    guard in the translation writes with `stop` when the same operation would raise."""
    import math  # noqa: F401  (for the evaluated source)
    try:
        eval(src, {"math": __import__("math")})
    except Exception as e:  # noqa: BLE001
        m = str(e)
        return "%s: %s" % (type(e).__name__, m) if m else type(e).__name__
    return None


# The exceptions the translation's guards raise, worded as THIS CPython words them (CPython
# rewords them between versions, so they are evaluated here rather than written down).
HOST_ERRORS = {
    "div_int": _host_error("1 / 0"),
    "div_float": _host_error("1.0 / 0.0"),
    "floordiv_int": _host_error("1 // 0"),
    "floordiv_float": _host_error("1.0 // 0.0"),
    "mod_int": _host_error("1 % 0"),
    "mod_float": _host_error("1.0 % 0.0"),
    "sqrt": _host_error("math.sqrt(-1.0)"),
    "log_zero": _host_error("math.log(0.0)"),
    "log_neg": _host_error("math.log(-1.0)"),
    "asin": _host_error("math.asin(2.0)"),
    "acos": _host_error("math.acos(2.0)"),
    "exp_range": _host_error("math.exp(1000.0)"),
    "trig_inf": _host_error("math.sin(float('inf'))"),
    "sinh_range": _host_error("math.sinh(1000.0)"),
    "gamma_pole": _host_error("math.gamma(0.0)"),
    "gamma_range": _host_error("math.gamma(200.0)"),
    "pow_zero_neg": _host_error("0.0 ** -1.0"),
    "pow_range": _host_error("10.0 ** 400"),
    "mpow_domain": _host_error("math.pow(0.0, -1.0)"),
    "mpow_range": _host_error("math.pow(10.0, 400.0)"),
    "int_nan": _host_error("int(float('nan'))"),
    "int_inf": _host_error("int(float('inf'))"),
    "round_nan": _host_error("round(float('nan'))"),
    "round_inf": _host_error("round(float('inf'))"),
    "floor_nan": _host_error("math.floor(float('nan'))"),
    "floor_inf": _host_error("math.floor(float('inf'))"),
    "fmod": _host_error("math.fmod(1.0, 0.0)"),
    "neg_shift": _host_error("1 << -1"),
    # list methods (funcs.py)
    "pop_empty": _host_error("[].pop()"),
    "pop_index": _host_error("[1].pop(5)"),
    "remove_missing": _host_error("[1].remove(2)"),
}

# The Python whose semantics the translation follows is the one running py2viv (its builtin
# `sum` of floats is compensated from 3.12 on).
HOST_PY = sys.version_info[:2]

_SPEC = None


def viv_spec(spec, is_int):
    """A Python format spec as Viv's fmt takes it, or None when fmt
    cannot write it the way Python does."""
    import re as _re
    m = _re.fullmatch(r"(\d*)(?:\.(\d+))?([gefd]?)", spec)
    if not m:
        return None
    width, prec, typ = m.group(1), m.group(2), m.group(3)
    if width and (width.startswith("0") or int(width) > 100):
        return None  # zero padding is Python's `0` flag, not in fmt
    if prec is not None and int(prec) > 40:
        return None
    if typ == "d":
        if prec is not None or not is_int:
            return None  # Python refuses `d` on a float, and a precision with `d`
        return spec
    if typ == "" and prec is not None:
        return None  # `.3`: Python's rule is neither g nor f
    if typ == "" and is_int:
        return width + "d"  # format(3, "8") writes the int, not 3.0
    return spec


def viv_name(name):
    """The Viv spelling of a Python identifier."""
    if name.startswith("py_") or name.startswith("viv__") or name.lower().startswith("npr_"):
        return "u_" + name
    if name in VIV_KEYWORDS or name in VIV_BUILTINS:
        return name + "_"
    return name


# ---------------------------------------------------------------- static types


class T:
    """A static type. kind: num, bool, arr, carr, cnum, str, none, func, tuple, module, unknown,
    and (ints.py) int (a Viv Int: a Python int annotated `int`), enum (a member of an enum.Enum
    class, name = the class's name), opt (an Optional, elts = [the type inside]).

    py: for num, 'int' or 'float' (Python's type, for printing); for arr, the NumPy dtype
    ('int', 'float', 'bool') or a list's element type. container: 'nd' or 'list' for arrays.
    """

    __slots__ = ("kind", "rank", "py", "container", "elts", "name")

    def __init__(self, kind, rank=0, py=None, container=None, elts=None, name=None):
        self.kind = kind
        self.rank = rank
        self.py = py
        self.container = container
        self.elts = elts
        self.name = name

    def key(self):
        return (self.kind, self.rank, self.py, self.container,
                tuple(e.key() for e in self.elts) if self.elts else None, self.name)

    def __eq__(self, other):
        return isinstance(other, T) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        if self.kind in ("arr", "carr"):
            return "%s%d[%s,%s]" % (self.kind, self.rank, self.py, self.container)
        if self.kind == "num":
            return "num(%s)" % self.py
        return self.kind

    def is_arr(self):
        return self.kind in ("arr", "carr")

    def scalar(self):
        return self.kind in ("num", "bool", "cnum")


NUM_I = T("num", py="int")
NUM_F = T("num", py="float")
NUM_MIXED = T("num", py="mixed")
BOOL = T("bool", py="bool")
UNK = T("unknown")
# a value of unknown type because the construct that made it was refused (a binding, a def's
# body, the argument a caller passed): what it reaches is a cascade of that refusal, reported
# as one (infer.py), not a construct of its own
UNK_C = T("unknown", name="cascade")
NONE = T("none")
STR = T("str")
STRS = T("strs", 1, "str", "list")  # a list of strings (immutable in Viv; strs.py)
CNUM = T("cnum", py="complex")
INT = T("int", py="int")  # a Viv Int (ints.py): a value of a Python int annotated `int`


def row_type(t):
    """The type of one row of t (an array or a list of rows, rank >= 2): a list of NumPy arrays
    ("ndrows") holds NumPy arrays, each printed and computed with as NumPy's."""
    if t.name == "ndrows":
        return T(t.kind, t.rank - 1, t.py, "nd")
    return T(t.kind, t.rank - 1, t.py, t.container, elts=t.elts, name=t.name)


def tup(elts):
    return T("tuple", elts=list(elts))


def arr(rank, py="float", container="nd"):
    return T("arr", rank, py, container)


def num_of(a, b=None):
    """The Python type of an arithmetic result: int only if every operand is int."""
    if b is None:
        return NUM_I if a.py == "int" else (NUM_MIXED if a.py == "mixed" else NUM_F)
    if a.py == "int" and b.py == "int":
        return NUM_I
    if a.py == "float" or b.py == "float":
        return NUM_F
    return NUM_MIXED


def join(a, b):
    """The type of a variable assigned both a and b, or None if they cannot share a name."""
    if a is None:
        return b
    if b is None:
        return a
    if a.kind == "unknown":
        return b
    if b.kind == "unknown":
        return a
    if a == b:
        return a
    if a.kind == "map" and b.kind == "map":
        return maps_mod.unify(a, b)  # two dicts in one name: one map type (maps.py)
    if a.kind == "tuple" and b.kind == "tuple":
        if len(a.elts) != len(b.elts):
            return None
        elts = [join(x, y) for x, y in zip(a.elts, b.elts)]
        if any(e is None for e in elts):
            return None
        return T("tuple", elts=elts)
    if a.kind == "num" and b.kind == "num":
        # an int on one path and a float on another: Python prints them differently
        return NUM_MIXED
    if a.kind == "num" and b.kind == "bool" or a.kind == "bool" and b.kind == "num":
        # Python prints a Bool as True or False and a number as a number, and Viv's Bool is
        # not a number: one name cannot hold both (RULES.md, Fixed holes: int-bool-join)
        return None
    if a.kind == b.kind and a.kind in ("arr", "carr") and a.rank == b.rank:
        if (a.elts or b.elts) and a.elts != b.elts:
            return None  # tuples, or ints beside floats, whose items keep their own types (funcs.py)
        if a.container != b.container or a.py != b.py:
            # a Python list and a NumPy array (`+` joins one and adds the other, and they print
            # differently), or arrays of two dtypes (an int or Bool array prints and divides as
            # itself, not as floats): one name, parameter or result cannot hold both (RULES.md,
            # Fixed holes: list-array-join, dtype-join)
            return None
        return T(a.kind, a.rank, a.py, a.container)
    if a.kind == "cnum" and b.kind == "num" or a.kind == "num" and b.kind == "cnum":
        return CNUM
    return None


# ---------------------------------------------------------------- refusals


class Refusal(Exception):
    def __init__(self, node, what, hint, gap=None):
        super().__init__(what)
        self.node = node
        self.what = what
        self.hint = hint
        self.gap = gap  # a language gap that would lift this refusal (R3, R5, ...), if any
        self.cascade = False  # a consequence of another refusal (an unknown type), not a construct of its own

    def where(self):
        n = self.node
        line = getattr(n, "lineno", 1)
        col = getattr(n, "col_offset", 0)
        end_line = getattr(n, "end_lineno", line)
        end_col = getattr(n, "end_col_offset", col + 1)
        return line, col, end_line, end_col


def node_pos(node):
    """Where a node is in the Python file: (line, col, end_line, end_col), columns 0-based byte
    offsets as CPython's ast gives them; None for no node or a node with no position."""
    if node is None or getattr(node, "lineno", None) is None:
        return None
    line, col = node.lineno, node.col_offset
    return (line, col, getattr(node, "end_lineno", None) or line, getattr(node, "end_col_offset", None) or col + 1)


def posmap_entry(pos):
    """A Viv line's entry in the manifest's `posmap`: [line, col, end_line, end_col] of the Python
    it came from, 1-based, columns in bytes (the end one past the last); null for a line with no
    Python origin (the header, a helper, a blank line). `viv run file.py` locates every error,
    in every engine, through it."""
    if pos is None:
        return None
    line, col, end_line, end_col = pos
    return [line, col + 1, end_line, end_col + 1]


def format_refusal(filename, lines, r):
    line, col, end_line, end_col = r.where()
    src = lines[line - 1] if 0 < line <= len(lines) else ""
    width = (end_col - col) if end_line == line else max(1, len(src.rstrip()) - col)
    width = max(1, width)
    gap = " (lifted by %s)" % r.gap if r.gap else ""
    return "%s:%d:%d: not numeric Python: %s%s\n    %s\n    %s%s\n  %s" % (
        filename, line, col + 1, r.what, gap, src.rstrip(), " " * col, "^" * width, r.hint)


# ---------------------------------------------------------------- the rules

# The NumPy, math and builtin mappings are data: rules/*.toml, applied by rule_engine.py (the
# format is tools/py2viv/RULES.md). The Viv helper defs a translation may call (the prelude,
# the guards that stop where Python raises, NumPy's and CPython's orders of summation) are in
# rules/helpers.toml, in the order a translation emits them.
RULES = rule_engine.Engine(types={"T": T, "num_of": num_of, "NUM_I": NUM_I, "NUM_F": NUM_F, "NUM_MIXED": NUM_MIXED, "BOOL": BOOL,
                                  "CNUM": CNUM, "UNK": UNK, "NONE": NONE, "STR": STR, "STRS": STRS,
                                  "host_py": list(HOST_PY), "viv_name": viv_name,
                                  "unicode_same": strs_mod.HOST_UNICODE == strs_mod.VIV_UNICODE})
PRELUDE = RULES.helpers

NP_ALIASES = {"numpy"}
MATH_ALIASES = {"math"}

REFUSED_MODULE_HINT = RULES.module_hints


class Options:
    def __init__(self, print_mode="exact"):
        self.print_mode = print_mode


# ---------------------------------------------------------------- scopes


class FuncInfo:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.params = [a.arg for a in node.args.args]
        nd = len(node.args.defaults)
        self.defaults = {}
        for p, d in zip(self.params[len(self.params) - nd:], node.args.defaults):
            self.defaults[p] = d
        self.param_types = [None] * len(self.params)
        self.prev_params = [None] * len(self.params)
        self.ret = None
        self.prev_ret = None
        self.locals = set()
        self.globals_decl = set()
        self.is_test = node.name.startswith("test_") and not self.params
        self.called = False


def appended_in(stmts, name):
    """Whether `name.append(...)` is called anywhere in these statements."""
    return any(isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and m.func.attr == "append"
               and isinstance(m.func.value, ast.Name) and m.func.value.id == name for st in stmts for m in ast.walk(st))


def grow_refusal(body, name, funcs, np_names):
    """The first use of list `name` in `body` that Viv's `lst = append(lst, x)` would not
    translate exactly (node, why), or None. See Translator._grow_refusal."""
    tree = ast.Module(body=list(body), type_ignores=[])
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def appends_to(stmts):
        return appended_in(stmts, name) or py_funcs.mutated_in(stmts, name)

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef)) and \
                any(isinstance(m, ast.Name) and m.id == name for m in ast.walk(n)):
            return n, "and read in a nested function or a lambda"
        if not (isinstance(n, ast.Name) and n.id == name):
            continue
        p = parent.get(n)
        if isinstance(n.ctx, ast.Store):
            if isinstance(p, ast.Assign) and p.targets == [n] and isinstance(p.value, ast.List) \
                    and not any(isinstance(e, ast.Starred) for e in p.value.elts):
                continue
            if py_funcs.fresh_list_binding(n, p, parent):
                continue  # a new list: a comprehension, list(...), sorted(...), [x] * n (funcs.py)
            return n, "and also bound to something else"
        if isinstance(n.ctx, ast.Del):
            return n, "and deleted"
        if isinstance(p, ast.Attribute):
            call = parent.get(p)
            if p.attr == "append" and isinstance(call, ast.Call) and call.func is p and isinstance(parent.get(call), ast.Expr):
                continue
            if py_funcs.list_method_use(p, call, parent):
                continue  # pop, insert, extend, remove, clear, reverse, sort, index, count, copy (funcs.py)
            return p, "and changed or read by `.%s`" % p.attr
        if isinstance(p, ast.comprehension) and p.iter is n:
            continue  # read by a comprehension's loop
        if isinstance(p, (ast.ListComp, ast.GeneratorExp)) and p.elt is n:
            return n, "and put into a new list as an item (Python shares it)"
        if isinstance(p, ast.Subscript) and p.value is n:
            continue
        if isinstance(p, ast.Call) and n in p.args:
            f = p.func
            if isinstance(f, ast.Name) and f.id not in funcs and f.id in ("len", "sum", "min", "max", "sorted", "print", "list"):
                continue
            loop = parent.get(p)
            if isinstance(f, ast.Name) and f.id not in funcs and f.id in ("enumerate", "reversed", "zip") \
                    and isinstance(loop, (ast.For, ast.comprehension)) and loop.iter is p:
                if isinstance(loop, ast.For) and (appends_to(loop.body) or appends_to(loop.orelse)):
                    return loop, "while a loop runs over it"
                continue
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in np_names:
                continue
            if isinstance(f, ast.Attribute) and f.attr == "extend" and p.args == [n] and not p.keywords:
                continue  # other.extend(lst) copies lst's items
            if isinstance(f, ast.Attribute) and f.attr == "join" and p.args == [n] and not p.keywords:
                continue  # sep.join(lst) reads the items into a new string (strs.py)
            if isinstance(f, ast.Name) and f.id not in funcs and f.id in ("str", "repr") and p.args == [n]:
                continue  # the text of the list
            return n, "and passed to `%s`" % (ast.unparse(f) if hasattr(ast, "unparse") else "a function")
        if isinstance(p, ast.Return) or isinstance(p, ast.FormattedValue):
            continue
        if isinstance(p, (ast.If, ast.While)) and p.test is n:
            continue
        if isinstance(p, ast.BoolOp):
            # `while a and b:`: only a truth value, when the `and`/`or` decides a branch
            top = p
            while isinstance(parent.get(top), (ast.BoolOp,)) or \
                    isinstance(parent.get(top), ast.UnaryOp) and isinstance(parent.get(top).op, ast.Not):
                top = parent.get(top)
            q = parent.get(top)
            if isinstance(q, (ast.If, ast.While, ast.IfExp, ast.Assert)) and q.test is top:
                continue
        if isinstance(p, (ast.Compare, ast.BinOp, ast.UnaryOp)):
            continue
        if isinstance(p, ast.For) and p.iter is n:
            if appends_to(p.body) or appends_to(p.orelse):
                return p, "while a loop runs over it"
            continue
        return n, "and aliased (a second name, an argument, a container)"
    return None


def assigned_names(body):
    """Names a function body binds (Python's rule for locals), and its `global` names."""
    names, glob = set(), set()

    def target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                target(e)
        elif isinstance(t, ast.Starred):
            target(t.value)

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            names.add(n.name)

        def visit_Lambda(self, n):
            pass

        def visit_Assign(self, n):
            for t in n.targets:
                target(t)
            self.generic_visit(n)

        def visit_AugAssign(self, n):
            target(n.target)
            self.generic_visit(n)

        def visit_AnnAssign(self, n):
            target(n.target)
            self.generic_visit(n)

        def visit_For(self, n):
            target(n.target)
            self.generic_visit(n)

        def visit_Global(self, n):
            glob.update(n.names)

        def visit_ListComp(self, n):
            pass

        def visit_GeneratorExp(self, n):
            pass

    for s in body:
        V().visit(s)
    return names - glob, glob


# ---------------------------------------------------------------- the translator

classes_mod.install(globals())


class Translator(maps_mod.MapsMixin, classes_mod.ObjectsMixin, funcs.FuncsMixin, funcs.ElementTypes, funcs.ListMethods,
                 infer.InferMixin, strs_mod.StringsMixin, ints_mod.IntsMixin):
    def __init__(self, src, filename="<python>", options=None):
        self.src = src
        self.filename = filename
        self.lines = src.splitlines()
        self.opt = options or Options()
        self.tree = ast.parse(src, filename)
        # nested defs, `f = lambda`, lambdas passed to defs, map and filter: rewritten in Python's
        # own terms first (funcs.py)
        funcs.P = sys.modules[Translator.__module__]
        infer.P = funcs.P
        strs_mod.P = funcs.P
        ints_mod.P = funcs.P
        self.tree = funcs.lift(self.tree)
        self.comments = self._comments()
        self.funcs = {}
        self.viv_imports = []  # import lines of Viv modules, as written (_s_Import)
        self.ext_mods = {}     # local name -> {def name: return type} of an imported Viv module
        self.ext_funcs = {}    # local name -> (def name, return type) imported from a Viv module
        self.np = set()        # names bound to numpy
        self.math = set()      # names bound to math
        self.from_np = {}      # local name -> numpy attribute path
        self.from_math = {}
        self.from_viv = set()  # grad, value_and_grad
        self.pathlib_names = set()  # names bound to pathlib or pathlib.Path (strs.py: files)
        self.np_random_mods = set()  # names bound to numpy.random (`import numpy.random as npr`)
        self.np_seeded = False  # the module calls np.random.seed somewhere
        self.refused_mods = {}  # local name -> module it came from (refused at the import)
        self.global_types = {}
        self.grow_py = {}      # (scope, name) -> the element type a grown list's appends add (R17)
        self.grow_shape = {}   # (scope, name) -> ("rows", py, column types) of a list grown by rows (funcs.py)
        self.prev_types = {}   # (scope, name) -> T, from the previous pass
        self.cur_types = {}

    # ------------------------------------------------ comments (kept in the output)
    def _comments(self):
        out = {}
        try:
            for tok in tokenize.generate_tokens(io.StringIO(self.src).readline):
                if tok.type == tokenize.COMMENT:
                    out[tok.start[0]] = (tok.start[1], tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
        return out

    # ------------------------------------------------ driver
    def translate(self):
        self._scan_module()
        self.grad_defs = self._grad_reach()
        self._ie_scan_decls()  # int and Optional annotations (ints.py)
        self._infer_scan()
        result = None
        for _ in range(12):
            snapshot = self._state()
            result = self._pass()
            if self._state() == snapshot:
                break
        return result

    def _state(self):
        return (tuple((k, tuple(f.param_types), f.ret) for k, f in sorted(self.funcs.items())),
                tuple(sorted(self.cur_types.items(), key=lambda kv: kv[0])),
                tuple(sorted((k, tuple(sorted(v))) for k, v in (getattr(self, "fn_flows", None) or {}).items())),
                self._objects_state(),
                self._infer_state())

    def _scan_module(self):
        self.all_names = {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)}
        # np.random's legacy draws are translated only in a file that seeds it (the draws check
        # at run time that the seed came first): `<numpy>.random.seed(...)`, `<numpy.random>.seed(...)`
        # or a `seed` imported from numpy.random
        seed_names = {a.asname or a.name for n in ast.walk(self.tree) if isinstance(n, ast.ImportFrom)
                      and n.module == "numpy.random" for a in n.names if a.name == "seed"}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr == "seed" and (
                        isinstance(f.value, ast.Attribute) and f.value.attr == "random" or isinstance(f.value, ast.Name)) \
                        or isinstance(f, ast.Name) and f.id in seed_names:
                    self.np_seeded = True
        self.global_reads = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                local, _ = assigned_names(node.body)
                local |= {a.arg for a in node.args.args}
                for m in ast.walk(node):
                    if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load) and m.id not in local:
                        self.global_reads.add(m.id)
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                f = FuncInfo(node)
                f.locals, f.globals_decl = assigned_names(node.body)
                f.locals |= set(f.params)
                f.plain = True
                self.funcs[node.name] = f
            if isinstance(node, ast.If) and self._is_main_guard(node):
                for s in node.body:
                    if isinstance(s, ast.FunctionDef):
                        f = FuncInfo(s)
                        f.locals, f.globals_decl = assigned_names(s.body)
                        f.locals |= set(f.params)
                        f.plain = True
                        self.funcs[s.name] = f
        self._ie_scan_enums()  # enum.Enum classes (ints.py)
        self._objects_scan()  # classes, named tuples, dict literals (classes.py)

    def _grad_reach(self):
        """The defs `grad`/`value_and_grad` differentiate: their targets and every def they call."""
        roots = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and node.func.args \
                    and isinstance(node.func.args[0], ast.Name):
                name = self._call_name(node.func.func)
                if name in ("grad", "value_and_grad", "viv.grad", "viv.value_and_grad"):
                    roots.add(node.func.args[0].id)
        reach, stack = set(), list(roots)
        while stack:
            f = stack.pop()
            if f in reach or f not in self.funcs:
                continue
            reach.add(f)
            for m in ast.walk(self.funcs[f].node):
                if isinstance(m, ast.Call) and isinstance(m.func, ast.Name) and m.func.id in self.funcs:
                    stack.append(m.func.id)
        return reach

    @staticmethod
    def _is_main_guard(node):
        t = node.test
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "__main__" and not node.orelse)

    def _pass(self):
        self.refusals = []
        self.out = []          # (indent, text, pyline, pypos): pypos is where() of the node, or None
        self.tests = []        # emitted after the program
        self.used = set()      # prelude helpers used
        self.sites = []        # print sites for the manifest
        self.tmp = 0
        self.scope = None      # None at module level, else FuncInfo
        self.types = {}        # current scope's variable types
        self.prev_types = self.cur_types
        self.cur_types = {}
        self.versions = {}
        self.indent = 0
        self.last_comment_line = 0
        self.slab_names = {}   # name bound to a row of an array -> node (write-through check)
        self.fancy_nodes = set()  # ids of subscripts translated as a mask or integer indices (copies)
        self._xmemo = {}
        self._views = None     # views.Views of this pass (built when first asked)
        self._order = None     # views.Order of this pass
        self.tuple_site = None  # the call whose several values are being unpacked or dropped
        self.stops = []        # the stop sites: line, message, whether the message is Python's exactly
        self.inexact_ops = {}   # what -> first Python line: operations Viv does not compute bit for bit as CPython/NumPy
        self.wrappers = []
        self.comp_env = []     # comprehension and lambda scopes (funcs.py)
        # what the appends of each grown list add, found anew each pass (a type guessed on an
        # earlier pass must not stick); a `[]` reads the last pass's (funcs.py, _grown_type)
        self.grow_prev = (self.grow_py, self.grow_shape)
        self.grow_py, self.grow_shape = {}, {}
        for f in self.funcs.values():
            f.prev_params = f.param_types
            f.param_types = [None] * len(f.params)
            f.prev_ret = f.ret
            f.ret = None
        self._infer_pass_start()  # the clones of defs called with several argument types (infer.py)
        self._objects_pass_start()
        self._block(self.tree.body, top=True)
        self._objects_finish()
        # functions never called: infer parameters from how the body uses them
        return self._assemble()

    # ------------------------------------------------ emission helpers
    def emit(self, text, node=None):
        line = getattr(node, "lineno", 0) if node is not None else 0
        if node is not None:
            self._flush_comments(line)
        self.out.append((self.indent, text, line, node_pos(node)))

    def _flush_comments(self, line):
        for ln in range(self.last_comment_line + 1, line + 1):
            c = self.comments.get(ln)
            if c is None:
                continue
            col, text = c
            src = self.lines[ln - 1]
            # tokenize counts characters; the positions are bytes, as ast's are
            pos = (ln, len(src[:col].encode()), ln, len(src.encode()))
            if src[:col].strip() == "":
                self.out.append((self.indent, text, ln, pos))
            else:
                self.out.append((self.indent, text, ln, pos))
        if line > self.last_comment_line:
            self.last_comment_line = line

    def refuse(self, node, what, hint, gap=None):
        r = Refusal(node, what, hint, gap)
        if "unknown type" in what:
            r.cascade = True
        raise r

    def inexact(self, node, what):
        """Record an operation whose Viv result can differ from CPython's or NumPy's in the last
        bits (a different libm, LAPACK, BLAS or FFT). The translation keeps it; diff.py compares
        exactly unless the program states a tolerance, and the report lists these."""
        self.inexact_ops.setdefault(what, getattr(node, "lineno", 0))

    def new_tmp(self, stem="t"):
        self.tmp += 1
        return "py_%s%d" % (stem, self.tmp)

    def use(self, helper):
        self.used.add(helper)
        return helper

    # ------------------------------------------------ variable types
    def scope_key(self):
        return (getattr(self.scope, "sid", None) or self.scope.name) if self.scope else "<module>"

    def _scope_of(self, name):
        if self.scope is not None and name in self.scope.locals and name not in self.scope.globals_decl:
            return getattr(self.scope, "sid", None) or self.scope.name  # a clone's own variables (infer.py)
        return "<module>"

    def vname(self, name):
        """The Viv name of a Python variable: renamed when Python reused the name for a value of
        another type (a new version, `y_2`), and when it collides with a Viv word."""
        c = self._comp_get(name)
        if c is not None:
            return c[0]
        return self.versions.get((self._scope_of(name), name), viv_name(name))

    def var_type(self, name):
        c = self._comp_get(name)
        if c is not None:
            return c[1]
        key = (self._scope_of(name), self.vname(name))
        t = self.cur_types.get(key)
        p = self.prev_types.get(key)
        return ints_mod.join_var(t, p) or t or p

    def nest(self):
        return self.indent - (1 if self.scope is not None else 0)

    def set_var(self, name, t, node, may_version=True):
        if self.scope is not None and name in self.scope.globals_decl:
            self.refuse(node, "`global %s` rebinding a module-level name" % name,
                        "Keep the value in a one-element array and write `%s[0] = ...`, as the ports do." % name)
        self._ie_set_check(name, t, node)  # an Int or an optional bound without conversion (ints.py)
        sk = self._scope_of(name)
        key = (sk, self.vname(name))
        old = self.cur_types.get(key)
        j = ints_mod.join_var(old, t)
        if j is not None and j.py == "mixed" and old is not None and old.py != "mixed" and t.py != "mixed" \
                and may_version and self.nest() == 0 and not (sk == "<module>" and name in self.global_reads):
            j = None  # an int that becomes a float in straight-line code: a new name keeps each exact
        if j is None:
            # Python reused the name for another type. In straight-line code (not inside an
            # if or a loop), every later use sees the new value: give it a new Viv name.
            # A module variable a def reads is left alone: the def reads whichever is current.
            if may_version and self.nest() == 0 and not (sk == "<module>" and name in self.global_reads):
                k = 2
                while True:
                    cand = "%s_%d" % (viv_name(name), k)
                    if cand not in self.all_names and cand not in self.versions.values():
                        break
                    k += 1
                self.versions[(sk, name)] = cand
                self.cur_types[(sk, cand)] = t
                return
            self.refuse(node, "`%s` holds %s here and %s elsewhere" % (name, self._kind_words(t), self._kind_words(old)),
                        "A Viv variable keeps one type: use two names (py2viv renames only in straight-line code).")
        self.cur_types[key] = j

    def _kind_words(self, t):
        if t is None:
            return "nothing"
        if t.kind == "unknown":
            return "a value of unknown type"
        if t.kind == "num":
            return "a number"
        if t.kind == "bool":
            return "a bool"
        if t.kind == "arr":
            return "a rank-%d array" % t.rank
        if t.kind == "carr":
            return "a complex array"
        if t.kind == "tuple":
            return "%d values" % len(t.elts)
        if t.kind == "rng":
            return "a Generator (np.random.default_rng)"
        if t.kind == "rec":
            info = self._rec_info(t)
            return info.words() if info is not None else "an object"
        if t.kind == "map":
            info = self._map_info(t)
            return "a " + (info.kind if info is not None else "dict")  # maps.py
        if t.kind == "str":
            return "a string"
        if t.kind == "strs":
            return "a list of strings"
        if t.kind in ints_mod.IE_KINDS:
            return self._ie_words(t)  # ints.py
        return "a " + t.kind

    # ------------------------------------------------ blocks and statements
    def _block(self, body, top=False):
        for i, s in enumerate(body):
            # a function's docstring was already turned into its `#|` intent
            try:
                self._stmt(s, top)
            except Refusal as r:
                self._taint_refused(s)  # what it binds has no type: a cascade where it is used (infer.py)
                self.refusals.append(r)
                self.out.append((self.indent, "# REFUSED: " + r.what, getattr(s, "lineno", 0), node_pos(s)))

    def _stmt(self, s, top=False):
        self.pre = []
        self._xmemo = {}
        mark = getattr(s, funcs.REFUSE, None)
        if mark is not None:
            self.refuse(s, *mark)
        m = getattr(self, "_s_" + type(s).__name__, None)
        if m is None:
            self._refuse_stmt(s)
        self._check_hoist_order(s)
        m(s, top)

    # the expressions whose translation runs before the rest of their statement (hoisted into
    # statements above it): a conditional expression, a comprehension, a reduction over a
    # generator, and math.sinh/cosh's argument
    _HOISTS = (ast.IfExp, ast.ListComp, ast.GeneratorExp)

    def _check_hoist_order(self, s):
        """Refuse a statement where a hoisted expression calls a def of the program and another
        call of a def, which Python evaluates first, sits outside it: the translation would
        run them in the other order (an effect, a print or a write into an array, would show
        it)."""
        if isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            return
        if isinstance(s, (ast.If, ast.While)):
            parts = [s.test]
        elif isinstance(s, ast.For):
            parts = [s.iter]
        elif isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Return, ast.Assert)):
            parts = [c for c in ast.iter_child_nodes(s) if isinstance(c, ast.expr)]
        else:
            return

        effects = set(self._effect_names())

        def user_calls(node):
            return [c for c in ast.walk(node) if isinstance(c, ast.Call) and (isinstance(c.func, ast.Name)
                    and c.func.id in effects or self._draws(c) or self._obj_user_call(c))]

        def hoisted(node):
            return isinstance(node, self._HOISTS) or (
                isinstance(node, ast.Call) and (self._call_name(node.func) in ("math.sinh", "math.cosh", "min", "max", "sorted", "next")
                                                or isinstance(node.func, ast.Lambda)))
        nodes = [n for p in parts for n in ast.walk(p)]
        calls = [c for p in parts for c in user_calls(p)]
        hoists = [x for x in nodes if hoisted(x)]
        for h in nodes:
            if not hoisted(h):
                continue
            inside = user_calls(h)
            if not inside:
                continue
            inner = set(map(id, ast.walk(h)))
            # a call inside an earlier hoisted part runs earlier, as in Python
            earlier = set()
            for h2 in hoists:
                if h2 is not h and (h2.lineno, h2.col_offset) < (h.lineno, h.col_offset) and id(h) not in set(map(id, ast.walk(h2))):
                    earlier |= set(map(id, ast.walk(h2)))
            for c in calls:
                if id(c) in inner or id(h) in set(map(id, ast.walk(c))) or id(c) in earlier:
                    continue  # inside the hoisted part, or a call it is an argument of
                if (c.lineno, c.col_offset) < (h.lineno, h.col_offset):
                    what = c.func.id if isinstance(c.func, ast.Name) else ast.unparse(c.func)
                    self.refuse(h, "an expression py2viv evaluates before the rest of its statement, after a call of `%s`" % what,
                                "Python calls `%s` first, and the translation second; assign the parts to variables in "
                                "Python's order." % what)

    GENERATOR_DRAWS = {"random", "integers", "uniform", "standard_normal", "normal", "choice", "shuffle", "permutation"}

    def _draws(self, call):
        """Whether a call draws from a random generator (np.random.* or a Generator's method):
        its place in the order of evaluation shows in the values."""
        name = self._call_name(call.func)
        if name and name.startswith("np.random.") and name not in ("np.random.default_rng",):
            return True
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr in self.GENERATOR_DRAWS and isinstance(f.value, ast.Name):
            t = self.var_type(f.value.id)
            return t is not None and t.kind == "rng"
        return False

    def flush_pre(self, node):
        for text in self.pre:
            self.emit(text, node)
        self.pre = []

    def _refuse_stmt(self, s):
        name = type(s).__name__
        table = {
            "ClassDef": ("a class", "Viv has no classes: a record and functions taking it (after R1), or plain arrays and functions."),
            "Try": ("`try`/`except`", "Check the condition and stop with `assert`."),
            "TryStar": ("`try`/`except*`", "Check the condition and stop with `assert`."),
            "With": ("`with`", "The numeric subset has no context managers; drop the `with`."),
            "AsyncFunctionDef": ("`async def`", "Write a plain `def`."),
            "AsyncFor": ("`async for`", "Write a plain `for`."),
            "AsyncWith": ("`async with`", "Drop it."),
            "Delete": ("`del`", "Viv frees nothing by hand; drop the `del`."),
            "Nonlocal": ("`nonlocal`", "Keep the value in a one-element array and write into it."),
            "Match": ("`match`", "Write `if`/`elif`/`else`."),
            "TypeAlias": ("a `type` alias", "Drop it; Viv infers types."),
        }
        what, hint = table.get(name, ("a `%s` statement" % name, "Not in the numeric subset (tools/py2viv/SUBSET.md)."))
        self.refuse(s, what, hint)

    # imports
    def _s_Import(self, s, top):
        for a in s.names:
            root = a.name.split(".")[0]
            local = a.asname or root
            if a.name == "numpy" or a.name == "numpy.random" and not a.asname:
                self.np.add(local)
            elif a.name == "numpy.random":
                self.np_random_mods.add(local)
            elif a.name == "math":
                self.math.add(local)
            elif a.name == "viv" or a.name in ("dataclasses", "collections", "typing"):
                pass  # dataclasses.dataclass, collections.namedtuple, typing.NamedTuple (classes.py)
            elif a.name == "pathlib":
                self.pathlib_names.add(local)  # pathlib.Path(p).read_text() (strs.py)
            elif a.name == "enum":
                pass  # enum.Enum classes (ints.py)
            elif self._viv_module(a.name) is not None:
                # a Viv module of the tree: `import hf.scf as scf` is the same Viv line
                defs = self._viv_module(a.name)
                self.viv_imports.append("import %s%s" % (a.name, " as " + a.asname if a.asname else ""))
                if a.asname or "." not in a.name:
                    self.ext_mods[local] = defs
            else:
                self.refused_mods[local] = a.name
                self.refuse(s, "`import %s`" % a.name,
                            REFUSED_MODULE_HINT.get(root, "`%s` is not in the numeric subset (math, numpy, viv)." % a.name))
        self._flush_comments(s.lineno)

    def _s_ImportFrom(self, s, top):
        mod = s.module or ""
        if mod == "__future__":
            return
        for a in s.names:
            local = a.asname or a.name
            if mod == "numpy":
                self.from_np[local] = a.name
            elif mod in ("numpy.linalg", "numpy.fft", "numpy.random"):
                self.from_np[local] = mod.split(".")[1] + "." + a.name
            elif mod == "math":
                self.from_math[local] = a.name
            elif mod == "viv" and a.name in ("grad", "value_and_grad"):
                self.from_viv.add(local)
            elif mod in ("typing", "collections.abc") :
                pass
            elif mod == "dataclasses" and a.name in ("dataclass", "replace") or mod == "collections" and a.name == "namedtuple":
                pass  # classes.py
            elif mod == "collections" and a.name in ("defaultdict", "Counter", "OrderedDict"):
                pass  # maps.py
            elif mod == "pathlib" and a.name == "Path":
                self.pathlib_names.add(local)  # Path(p).read_text() (strs.py)
            elif mod == "enum":
                pass  # Enum classes (ints.py); IntEnum, Flag, auto, ... are refused where they are used
            elif self._viv_module(mod) is not None and a.name in self._viv_module(mod):
                self.viv_imports.append("from %s import %s%s" % (mod, a.name, " as " + a.asname if a.asname else ""))
                self.ext_funcs[local] = (a.name, self._viv_module(mod)[a.name])
            else:
                root = mod.split(".")[0]
                self.refused_mods[local] = mod
                self.refuse(s, "`from %s import %s`" % (mod, a.name),
                            REFUSED_MODULE_HINT.get(root, "`%s` is not in the numeric subset (math, numpy, viv)." % mod))
        self._flush_comments(s.lineno)

    def _viv_module(self, dotted):
        """The defs of the Viv module `dotted` names (`hf.scf`), found as the
        compiler finds an import (lang/viv/src/imports.rs): beside the file,
        under lib/, under the repository root; {def: return type, or None when
        the def does not declare one}. None when there is no such module."""
        cache = self.__dict__.setdefault("_viv_mods", {})
        if dotted in cache:
            return cache[dotted]
        p = dotted.replace(".", "/")
        here = os.path.dirname(os.path.abspath(self.filename)) if self.filename and not self.filename.startswith("<") else os.getcwd()
        root = rule_engine.REPO_ROOT
        cands = [os.path.join(root, p + ".viv")] if p.startswith("lib/") else [os.path.join(here, p + ".viv"), os.path.join(root, "lib", p + ".viv"), os.path.join(root, p + ".viv")]
        found = next((c for c in cands if os.path.isfile(c)), None)
        defs = None
        if found:
            types = {"Num": NUM_F, "Bool": BOOL, "Str": STR, "Int": INT}
            defs = {}
            for m in re.finditer(r"^(?:private\s+)?def\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:->\s*([A-Za-z_]\w*))?", open(found).read(), re.M):
                defs[m.group(1)] = types.get(m.group(2))
        cache[dotted] = defs
        return defs

    def _ext_call(self, n, name, ret):
        """A call of a def of an imported Viv module: its arguments as they translate, its
        value the type its signature declares (py2viv knows nothing else about it)."""
        if ret is None:
            self.refuse(n, "a call of `%s`, whose Viv def declares no return type" % name,
                        "Give the def a return annotation (`-> Num`, `-> Bool`, `-> Str`, `-> Int`).")
        if n.keywords:
            self.refuse(n, "keyword arguments to an imported def", "Pass them positionally, or by the Viv def's names.")
        codes = [self.expr(a)[0] for a in n.args]
        return "%s(%s)" % (name, ", ".join(codes)), ret, 100

    def _s_Raise(self, s, top):
        """`raise E("message")`: the run stops with the traceback's last line (`stop`), located,
        exit status 1, as the uncaught exception ends CPython. try/except is refused, so nothing
        can catch it."""
        if s.exc is None:
            self.refuse(s, "a bare `raise`", "Raise a named exception: raise ValueError(\"...\").")
        if self.scope is not None and self.scope.is_test:
            self.refuse(s, "`raise` inside a test", "pytest records it as an error and moves on; Viv would stop the run. Assert instead.")
        e = s.exc
        args = []
        if isinstance(e, ast.Call):
            if e.keywords:
                self.refuse(e, "an exception with keyword arguments", "Pass the message only.")
            args = e.args
            e = e.func
        if not isinstance(e, ast.Name):
            self.refuse(e, "raising a qualified or computed exception", "Raise a builtin exception by name: ValueError, RuntimeError, ...")
        if len(args) > 1:
            self.refuse(s.exc, "an exception with several arguments", "Pass one message; Python prints several as a tuple.")
        self.flush_pre(s)
        self.emit(self._stop(e.id, args[0] if args else None, s), s)

    def _s_Pass(self, s, top):
        pass

    def _s_Break(self, s, top):
        self.emit("break", s)

    def _s_Continue(self, s, top):
        self.emit("continue", s)

    def _s_Global(self, s, top):
        self._flush_comments(s.lineno)

    def _s_FunctionDef(self, s, top):
        if not top or self.scope is not None:
            self.refuse(s, "a nested function", "Move `%s` to module level and pass what it reads as arguments." % s.name)
        if s.decorator_list:
            self.refuse(s.decorator_list[0], "a decorator", "Drop the decorator.")
        a = s.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            self.refuse(s, "`*args`, `**kwargs` or keyword-only parameters", "List the parameters positionally.")
        f = self.funcs[s.name]
        self._emit_defaults(f, s)  # a default Python evaluates once, when the def runs (classes.py)
        for v in self._variants(f):  # one def, or a clone per argument types (infer.py)
            outer_types = self.types
            self.scope = v
            saved_indent = self.indent
            try:
                self._funcdef(s, v)
            finally:
                self.indent = saved_indent
                self.scope = None
                self.types = outer_types

    def _funcdef(self, s, f):
        body = s.body
        doc = ast.get_docstring(s)
        if doc is not None:
            body = body[1:]
        saved_indent = self.indent
        if f.is_test:
            # the body is a def, so that a failed assert can end it (`return`), as pytest ends a
            # test at its first failed assertion; the test block calls it
            start = len(self.out)
            self._flush_comments(s.lineno)
            if doc:
                for dl in doc.strip().splitlines():
                    self.emit("#| " + dl.strip() if dl.strip() else "#|", s)
            body_def = "py_test_%s" % s.name
            self.emit("def %s() {" % body_def, s)
            self.indent += 1
            self._block(body)
            self.emit("assert_true(true)", s)
            self.emit("return 0", s)
            self.indent -= 1
            self.emit("}", s)
            self.emit('test "%s" {' % s.name, s)
            self.emit("    %s()" % body_def, s)
            self.emit("}", s)
            # tests run after the program, as pytest runs them after the import
            self.tests.append(self.out[start:])
            del self.out[start:]
        else:
            if doc:
                self._flush_comments(s.lineno - 1)
                for dl in doc.strip().splitlines():
                    self.emit("#| " + dl.strip() if dl.strip() else "#|", None)
            params = []
            for i, p in enumerate(f.params):
                pt = self._ie_param_decl(f, i) or self._param_type(f, i)  # an int or Optional annotation (ints.py)
                self.set_var(p, pt, s)
                params.append(self.vname(p) + self._annot(pt))
            rd = self._ie_ret_decl(f)
            ret = " -> " + self._ie_viv_type(rd) if rd is not None and rd.kind in ("int", "opt") else ""
            self.emit("def %s(%s)%s {" % (self._viv_def_name(f), ", ".join(params), ret), s)
            self.indent += 1
            self._block(body)
            self._ie_def_end(f, s, body)  # an optional result returns None at the end (ints.py)
            self._obj_def_tail(f, s)  # a def that changes an object passed to it returns it (classes.py)
            self.indent -= 1
            self.emit("}", s)

    def _annot(self, t):
        if t is None:
            return ""
        if t.kind == "arr":
            return ": Arr"
        if t.kind == "rng":
            return ": [Int]"  # lib/nprandom's PCG64 state: an array of Ints
        if t.kind == "carr":
            return ": CArr"
        if t.kind == "rec":
            return ": " + viv_name(t.name)
        if t.kind == "map":
            return ": Map"  # maps.py
        if t.kind in ints_mod.IE_KINDS:
            return ": " + self._ie_viv_type(t)  # ints.py
        return ""

    def _param_type(self, f, i):
        t = join(f.param_types[i], f.prev_params[i])
        if t is not None and t.kind != "unknown":
            return t
        # no call site translated: guess from how the body uses the parameter
        name = f.params[i]
        if any(isinstance(m, ast.Name) and m.id == f.name and isinstance(m.ctx, ast.Load)
               for m in ast.walk(self.tree) if not (isinstance(m, ast.Name) and self._inside(m, f.node))):
            # the program calls it: the calls type the parameter (on a later pass); a guess here
            # would join them (an int argument and a guessed float make a number of either type,
            # a list argument and a guessed NumPy array a value of neither kind). A dict argument
            # is typed the same way: its map type is the call's (maps.py, _map_args), one map
            # type per parameter, and a def called with a dict at one site and a list at another
            # is cloned per kind (infer.py)
            return UNK_C if self._tainted_param(f, i) or self._calls_all_refused(f) else UNK
        if self._map_param(f, name) or self._map_hint(f, i) is not None:
            # never called, and used as only a dict is used (or annotated as one): no call made
            # a map for it, and no array or number is guessed (maps.py)
            return UNK
        rank = 0
        arrayish = False
        ARR_ATTRS = {"copy", "sum", "mean", "max", "min", "shape", "T", "size", "dot", "tolist", "real", "imag", "std", "ndim", "sort"}
        ARR_FUNCS = {"len", "np.zeros_like", "np.ones_like", "np.empty_like", "np.sum", "np.mean", "np.max", "np.min",
                     "np.copy", "np.dot", "np.linalg.norm", "np.linalg.solve", "np.sort", "np.fft.fft", "np.fft.ifft", "sorted"}

        def is_p(x):
            return isinstance(x, ast.Name) and x.id == name
        for n in ast.walk(f.node):
            if isinstance(n, ast.Subscript) and is_p(n.value):
                sl = n.slice
                rank = max(rank, len(sl.elts) if isinstance(sl, ast.Tuple) else 1)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Subscript) and is_p(n.value.value):
                rank = max(rank, 2)
            if isinstance(n, ast.Attribute) and is_p(n.value) and n.attr in ARR_ATTRS:
                arrayish = True
                if n.attr == "T":
                    rank = max(rank, 2)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult) and (is_p(n.left) or is_p(n.right)):
                arrayish = True
            if isinstance(n, ast.Call) and n.args and is_p(n.args[0]) and self._call_name(n.func) in ARR_FUNCS:
                arrayish = True
            if isinstance(n, ast.For) and is_p(n.iter):
                arrayish = True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("zip", "enumerate", "reversed") \
                    and any(is_p(a) for a in n.args):
                arrayish = True  # iterated over
            if isinstance(n, ast.comprehension) and is_p(n.iter):
                arrayish = True
        g = self._guess_by_hint_or_use(f, i, rank)  # its annotation, or an int by use (infer.py)
        if g is not None:
            return g
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and is_p(n.func.value)
               and n.func.attr in strs_mod.STR_ONLY_METHODS for n in ast.walk(f.node)):
            return STR  # a receiver of str's own methods (s.split(), s.upper(), ...): a string

        def str_const(x):
            return isinstance(x, ast.Constant) and isinstance(x.value, str) or \
                isinstance(x, (ast.Tuple, ast.List, ast.Set)) and x.elts and all(str_const(e) for e in x.elts)
        if any(isinstance(n, ast.Compare) and len(n.ops) == 1 and (is_p(n.left) and str_const(n.comparators[0])
               or is_p(n.comparators[0]) and str_const(n.left)) for n in ast.walk(f.node)):
            return STR  # compared with a string constant (`if method == "newton":`): a string
        if rank or arrayish:
            return arr(max(rank, 1))
        return self._guess_rec_param(f, name) or NUM_F

    def _s_Return(self, s, top):
        if self.scope is None:
            self.refuse(s, "`return` outside a function", "Remove it.")
        if self.scope.is_test:
            self.refuse(s, "`return` inside a test", "A Viv test block cannot return; restructure with `if`.")
        if self._obj_return(s):  # a def that changes an object passed to it (classes.py)
            return
        if self._ie_return(s):  # a result annotated int or Optional (ints.py)
            return
        if s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None):
            f = self.scope
            if (f.ret or f.prev_ret) is not None and (f.ret or f.prev_ret).kind == "tuple":
                self.refuse(s, "`%s` returns %s elsewhere and nothing here" % (f.name, self._kind_words(f.ret or f.prev_ret)),
                            "Return the same number of values from every `return`.")
            self.flush_pre(s)
            self.emit("return 0", s)
            return
        if isinstance(s.value, ast.Tuple):
            self._return_tuple(s)
            return
        if isinstance(s.value, ast.IfExp):
            # return a if c else b  ->  if c { return a } else { return b }
            c = self._cond(s.value.test)
            self.flush_pre(s)
            self.emit("if %s {" % c, s)
            self.indent += 1
            self._s_Return(ast.copy_location(ast.Return(value=s.value.body), s), top)
            self.indent -= 1
            self.emit("}", s)
            self.emit("else {", s)
            self.indent += 1
            self._s_Return(ast.copy_location(ast.Return(value=s.value.orelse), s), top)
            self.indent -= 1
            self.emit("}", s)
            return
        forward = isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id in self.funcs
        if forward:
            self.tuple_site = s.value  # `return g(x)` of a def that returns several values forwards them
        try:
            code, t, _ = self.expr(s.value, keep=True)  # an Int, a member, an optional (ints.py)
        finally:
            self.tuple_site = None
        if t.kind == "opt" and self.scope.node.returns is not None:
            code, t, _ = self._ie_lower(s.value, (code, t, 100))  # an optional from a def annotated otherwise
        f = self.scope
        if t.kind == "int" or any(x is not None and x.kind == "int" for x in (f.ret, f.prev_ret)):
            # an Int beside Python ints kept as doubles: the def returns an Int (ints.py)
            code, t = self._ie_ret_int(f, code, t, s)
            j = ints_mod.join_var(f.ret, t)
        else:
            j = join(f.ret, t)
        if j is None:
            self.refuse(s.value, "`%s` returns %s here and %s elsewhere" % (f.name, self._kind_words(t), self._kind_words(f.ret)),
                        "A Viv def returns one type.")
        f.ret = j
        self._note_return(s.value, t)  # a new list returned? (infer.py)
        if t.kind == "bool":
            code = code  # Viv defs may return a Bool
        self.flush_pre(s)
        self.emit("return " + code, s)

    def _return_tuple(self, s):
        """`return a, b`: Viv's several results (a def that returns a tuple returns each
        element; its caller unpacks them: `a, b = f(x)`)."""
        elts = s.value.elts
        if len(elts) < 2:
            self.refuse(s.value, "returning a tuple of %d" % len(elts), "Return the value itself.")
        codes, types = [], []
        for e in elts:
            if isinstance(e, ast.Starred):
                self.refuse(e, "a starred value in a return", "Return the values one by one.")
            if isinstance(e, ast.Tuple):
                self.refuse(e, "a tuple inside a returned tuple", "Return the values flat: `return a, b, c`.")
            code, t, _ = self.expr(e)
            if t.kind in ("none", "func", "module", "tuple"):
                self.refuse(e, "returning %s in a tuple" % self._kind_words(t), "Return numbers, bools, strings and arrays.")
            codes.append(code)
            types.append(t)
        t = tup(types)
        f = self.scope
        self._note_return(None, t)
        j = join(f.ret, t)
        if j is None:
            self.refuse(s.value, "`%s` returns %s here and %s elsewhere" % (f.name, self._kind_words(t), self._kind_words(f.ret)),
                        "A Viv def returns one type, and a def that returns several values returns as many from every `return`.")
        f.ret = j
        self.flush_pre(s)
        self.emit("return " + ", ".join(codes), s)

    def _s_Expr(self, s, top):
        v = s.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            # a bare string: a comment
            for dl in v.value.strip().splitlines():
                self.emit("# " + dl.strip(), s)
            return
        if isinstance(v, ast.Constant) and v.value is Ellipsis:
            return
        if isinstance(v, ast.Call):
            fname = self._call_name(v.func)
            if fname == "print":
                self._print(v, s)
                return
            if self._obj_thread_stmt(v, s, None):  # a call that changes an object passed to it (classes.py)
                return
            if self._with_write_stmt(v, s):  # f.write(text) in `with open(p, "w") as f` (strs.py)
                return
            code = self._call_stmt(v, s)
            if code is not None:
                self.flush_pre(s)
                self.emit(code, s)
                return
            self.tuple_site = v
            try:
                code, t, _ = self.expr(v)
            finally:
                self.tuple_site = None
            self.flush_pre(s)
            if isinstance(v.func, ast.Name) and v.func.id in self.funcs:
                self.emit(code, s)
            else:
                self.emit(code, s)
            return
        self.refuse(v, "an expression whose value is dropped", "Assign it, print it, or remove it.")

    def _s_Assert(self, s, top):
        """`assert c`: in a test, a failed assertion that ends the test (as pytest ends it);
        elsewhere, a stop with Python's AssertionError, which ends the program as the uncaught
        exception does."""
        c = self._cond(s.test)
        self.flush_pre(s)
        self.emit("if not %s {" % self._paren(c, 50 if " " in c else 100, 81), s)
        self.indent += 1
        if self.scope is not None and self.scope.is_test:
            self.emit("assert_true(false)", s)
            self.emit("return 0", s)
        else:
            self.emit(self._stop("AssertionError", s.msg, s), s)
        self.indent -= 1
        self.emit("}", s)

    def _stop(self, exc, msg_node, node):
        """`stop("Type: message")` for an exception the program raises. A literal message is
        kept exactly (it is the traceback's last line); a computed one keeps the exception's
        type and the literal parts, and the manifest marks the site so diff.py compares the type
        only."""
        exact = True
        if msg_node is None:
            text = exc
        elif isinstance(msg_node, ast.Constant) and isinstance(msg_node.value, str):
            text = "%s: %s" % (exc, msg_node.value) if msg_node.value else exc
        elif self._stop_text(msg_node) is not None:
            # a message computed at run time (an f-string, a string variable, a number): its text
            # as Python's str writes it, after the type (strs.py)
            code = self._stop_text(msg_node)
            self.flush_pre(node)
            self.stops.append({"line": node.lineno, "message": "%s: ..." % exc, "exact": True})
            return "stop(%s(%s, %s))" % (self.use("py_excline"), viv_str(exc), code)
        elif isinstance(msg_node, ast.JoinedStr):
            pieces = []
            for v in msg_node.values:
                if isinstance(v, ast.Constant):
                    pieces.append(str(v.value))
                else:
                    pieces.append("{%s}" % (ast.get_source_segment(self.src, v.value) or "..."))
            text = "%s: %s" % (exc, "".join(pieces))
            exact = False
        else:
            text = "%s: {%s}" % (exc, ast.get_source_segment(self.src, msg_node) or "...")
            exact = False
        text = "".join(ch if ch >= " " else " " for ch in text)
        self.stops.append({"line": node.lineno, "message": text, "exact": exact})
        return "stop(%s)" % viv_str(text)

    def _s_AnnAssign(self, s, top):
        if s.value is None:
            return
        self._assign_to(s.target, s.value, s)

    def _s_Assign(self, s, top):
        if self._objects_assign(s.targets[0], s.value, s):  # P = namedtuple(...) (classes.py)
            return
        if len(s.targets) > 1:
            # a = b = v: evaluate once
            code, t, _ = self.expr(s.value)
            if t.kind in ("arr", "carr"):
                self.refuse(s, "chained assignment of an array", "Assign it to one name, then `b = a` (arrays alias, as in Python).")
            for tgt in s.targets:
                self._store(tgt, code, t, s)
            return
        tgt = s.targets[0]
        if isinstance(tgt, (ast.Tuple, ast.List)):
            self._tuple_assign(tgt, s.value, s)
            return
        self._assign_to(tgt, s.value, s)

    def _assign_to(self, tgt, value, s):
        # special forms whose Viv is a statement
        if self._obj_thread_stmt(value, s, [tgt]):  # x = obj.m(...) of a method that changes obj (classes.py)
            return
        if isinstance(value, ast.List) and not value.elts:
            # `lst = []` of a list only ever grown by `lst.append(x)` (R17): `lst = []`
            bad = self._grow_refusal(tgt) if isinstance(tgt, ast.Name) else (value, "into an element or a field")
            if bad is not None:
                self.refuse(bad[0], "an empty list grown %s" % bad[1],
                            "py2viv grows a list bound only to new lists and read only by `len`, indexing, loops, conditions, "
                            "`return` and its own methods; Viv's `lst = append(lst, x)` gives a new array, so an alias would "
                            "not see the change.")
            self.set_var(tgt.id, self._grown_type(tgt.id), s)
            self.flush_pre(s)
            self.emit("%s = []" % self.vname(tgt.id), s)
            return
        if isinstance(value, ast.List) and isinstance(tgt, ast.Name) and appended_in(self._grow_body(), tgt.id):
            # `lst = [a, b]` of a list then grown by `lst.append(x)` (R17): the literal, whose
            # numbers fix the element type the appends must match
            bad = self._grow_refusal(tgt)
            if bad is not None:
                self.refuse(bad[0], "a list grown %s" % bad[1],
                            "py2viv grows a list bound only to new lists and read only by `len`, indexing, loops, conditions, "
                            "`return` and its own methods; Viv's `lst = append(lst, x)` gives a new array, so an alias would "
                            "not see the change.")
            items = [self.expr(e) for e in value.elts]
            pys = {t.py if t.kind == "num" else t.kind for _, t, _ in items}
            if pys == {"arr"}:
                self._grow_rows_literal(tgt, value, items, s)  # rows (funcs.py)
                return
            if pys == {"str"}:
                # a list of strings then grown by appends (strs.py)
                key = (self._scope_of(tgt.id), tgt.id)
                if self.grow_py.setdefault(key, "str") != "str":
                    self.refuse(value, "a list of strings grown by appends of numbers", "Python keeps both; keep them in two lists.")
                self.set_var(tgt.id, STRS, s)
                self.flush_pre(s)
                self.emit("%s = [%s]" % (self.vname(tgt.id), ", ".join(c for c, _, _ in items)), s)
                return
            if len(pys) != 1 or not pys <= {"int", "float"}:
                self.refuse(value, "a list grown by `.append` that starts with %s" % " and ".join(sorted(pys)),
                            "Start it with ints only or floats only (Python prints each as it is), or with `[]`.", gap="R17")
            py = pys.pop()
            key = (self._scope_of(tgt.id), tgt.id)
            if self.grow_py.setdefault(key, py) != py:
                self.refuse(value, "a list of ints and floats", "Python prints each as it is; make them all floats.")
            self.set_var(tgt.id, T("arr", 1, py, "list"), s)
            self.flush_pre(s)
            self.emit("%s = [%s]" % (self.vname(tgt.id), ", ".join(c for c, _, _ in items)), s)
            return
        code, t, _ = self.expr(value, target_hint=tgt, keep=isinstance(tgt, ast.Name))  # an Int, a member, an optional (ints.py)
        if t.kind == "none" and not (isinstance(tgt, ast.Name) and (self._ie_decl(tgt.id) or self.var_type(tgt.id) or UNK).kind == "opt"):
            self.refuse(value, "`None` as a value", "Use a number (for example -1 or nan) to mean 'nothing'.")
        if t.kind in ("func", "module", "tuple"):
            self.refuse(value, "a %s as a value" % ("function" if t.kind == "func" else t.kind),
                        "Call it directly." if t.kind == "func" else "Unpack it on the same line.", gap="R5" if t.kind == "tuple" else None)
        # remember names bound to a row of an array: writing through them is a NumPy view write
        if isinstance(tgt, ast.Name):
            if isinstance(value, ast.Subscript) and t.is_arr() and id(value) in self.fancy_nodes:
                # a mask or integer indices: NumPy gives a copy
                self.slab_names.pop(tgt.id, None)
            elif isinstance(value, ast.Subscript) and t.is_arr():
                if not (self._has_range(value) and t.container == "list" and t.rank == 1):
                    sliced = self._has_range(value) or any(self._is_newaxis(i) for _, items in self._levels(value) for i in items)
                    self._row_alias_check(value, value, "a slice" if sliced else "a row", names=[tgt.id])
                    self.slab_names[tgt.id] = value
                else:
                    self.slab_names.pop(tgt.id, None)
            elif t.is_arr() and self._view_base(value) is not None:
                self._row_alias_check(self._view_base(value), value, "a view", names=[tgt.id])
                self.slab_names[tgt.id] = value
            elif t.is_arr() and any(v for _, v in self.views.roots(value)):
                # np.transpose of a slice, a def that returns a view of its argument, ...
                self._row_alias_check(value, value, "a view", names=[tgt.id])
                self.slab_names[tgt.id] = value
            else:
                self.slab_names.pop(tgt.id, None)
        self._store(tgt, code, t, s)

    def _store(self, tgt, code, t, s):
        if isinstance(tgt, ast.Name) and self._ie_store(tgt, code, t, s):
            return  # an Int, a member, an optional, or a variable declared one (ints.py)
        if isinstance(tgt, ast.Name):
            self.set_var(tgt.id, t, s)
            self.flush_pre(s)
            self.emit("%s = %s" % (self.vname(tgt.id), code), s)
            return
        if isinstance(tgt, ast.Subscript):
            if self._obj_subscript_store(tgt, code, t, s):  # d[1] = v of a dict with int keys (classes.py)
                return
            root = tgt
            while isinstance(root, ast.Subscript):
                root = root.value
            rt = self.var_type(root.id) if isinstance(root, ast.Name) else None
            if rt is not None and rt.kind == "str":
                self.refuse(tgt, "a store into a string", "Python raises TypeError ('str' object does not support item "
                            "assignment); build a new string: s = s[:i] + c + s[i + 1:].")
            if rt is not None and rt.kind == "strs":
                self.refuse(tgt, "a store into a list of strings",
                            "A Viv list of strings is never changed in place; build a "
                            "new one: lst = lst[:i] + [w] + lst[i + 1:].")
            if t.kind == "rec":
                self.refuse(tgt, "an object stored into an array", "Viv arrays hold numbers; keep the object in a variable.")
            vnode = getattr(s, "value", None) or s
            r = self._fancy(tgt, store=(vnode, (code, t, 100)))
            if r is not None:
                self.flush_pre(s)
                self.emit(r[0], s)
                return
            base, idx, bt = self._subscript_target(tgt)
            if self.slice_store_rank:
                code = self._slice_value(tgt, code, t, bt, base, idx)
                self.flush_pre(s)
                self.emit("%s[%s] = %s" % (base, idx, code), s)
                return
            if t.is_arr():
                self.refuse(tgt, "an array stored into one element", "NumPy raises ValueError (setting an array element with a sequence); store a number.")
            if bt is not None and bt.is_arr() and bt.elts:
                self._elts_store(tgt, bt, t)  # a tuple (TypeError), or a list of ints and floats (funcs.py)
            self._list_item_store(tgt, bt, t)
            if bt is not None and bt.kind == "arr" and bt.container == "nd" and bt.py == "int" and t.kind == "num" and t.py in ("float", "mixed"):
                self.refuse(tgt, "a float stored into a NumPy integer array (NumPy truncates it)",
                            "Make the array float: np.zeros(n) or np.array([...], dtype=float).")
            if t.kind == "bool":
                code = {"true": "1", "false": "0"}.get(code) or "%s(%s)" % (self.use("py_b01"), code)
            self.flush_pre(s)
            self.emit("%s[%s] = %s" % (base, idx, code), s)
            return
        if isinstance(tgt, ast.Attribute):
            if self._obj_store(tgt, code, t, s):  # a field rebound (classes.py)
                return
            root = tgt
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in self.refused_mods:
                self._e_Name(root)
            self.refuse(tgt, "assigning an attribute", "Viv has no objects; use a variable.")
        self.refuse(tgt, "this assignment target", "Assign to a name or an indexed array element.")

    def _list_item_store(self, tgt, bt, t):
        """`lst[i] = v` of a Python list: the item takes v's own type (an int stays an int, and
        prints so, in a list of floats), where Viv's array holds one kind of number. The stored
        value must be of the list's type (RULES.md, Fixed holes: list-item-type)."""
        if bt is None or not bt.is_arr() or bt.container != "list" or bt.kind != "arr" or bt.elts:
            return
        if t.kind not in ("num", "bool"):
            return
        tp = "bool" if t.kind == "bool" else t.py
        if tp != bt.py:
            words = {"int": "ints", "float": "floats", "bool": "Bools", "mixed": "numbers"}
            what = {"int": "an int", "float": "a float", "bool": "a Bool", "mixed": "a number that is an int on some paths and a float on others"}
            self.refuse(tgt, "storing %s into a list of %s" % (what.get(tp, tp), words.get(bt.py, bt.py)),
                        "Python keeps the item's own type in the list (and prints it so); store a value of the list's type, "
                        "or make the list a NumPy array.")

    def _slice_value(self, tgt, code, t, bt, base, idx):
        """The value stored into a slice or a row (`a[1:3] = v`, `A[i] = v`): a number every
        element takes, or an array of the slice's rank, which NumPy broadcasts and Viv takes
        only at the slice's exact shape (py_fit stops where the shapes differ)."""
        rank = self.slice_store_rank
        if t.kind == "bool":
            code = {"true": "1", "false": "0"}.get(code) or "%s(%s)" % (self.use("py_b01"), code)
            t = NUM_I
        if t.kind in ("cnum", "carr") and bt.kind != "carr":
            self.refuse(tgt, "a complex value stored into a real array", "NumPy drops the imaginary part (with a warning); take the real part yourself.")
        if bt.py == "int" and (t.py in ("float", "mixed") or t.kind in ("cnum", "carr")):
            self.refuse(tgt, "a float stored into a NumPy integer array (NumPy truncates it)",
                        "Make the array float: np.zeros(n) or np.array([...], dtype=float).")
        if bt.py == "bool":
            self.refuse(tgt, "a store into a slice of a Bool array", "Store numbers into a number array.")
        if t.kind in ("num", "cnum", "unknown"):
            return code
        if not t.is_arr():
            self.refuse(tgt, "storing %s into a slice" % self._kind_words(t), "Store a number or an array.")
        if t.container == "list" and t.py == "mixed":
            self.refuse(tgt, "a list mixing ints and floats stored into a slice", "Make them all floats.")
        if t.rank > rank:
            self.refuse(tgt, "a rank-%d value stored into a rank-%d slice" % (t.rank, rank),
                        "NumPy drops the value's leading axes of length 1, or raises; reshape it to the slice's shape first.")
        if (t.kind == "carr") != (bt.kind == "carr"):
            self.refuse(tgt, "a %s array stored into a slice of a %s array" % ("complex" if t.kind == "carr" else "real",
                                                                             "complex" if bt.kind == "carr" else "real"),
                        "Convert the value first.")
        # NumPy broadcasts the value into the slice's shape (py_bset stops where it cannot)
        return "%s(%s, %s[%s])" % (self.use("py_cbset" if bt.kind == "carr" else "py_bset"), code, base, idx)

    def _subscript_target(self, tgt):
        """a[i][j] or a[i, j] as a store target: (base code, index code, base type)."""
        idxs = []
        node = tgt
        while isinstance(node, ast.Subscript):
            sl = node.slice
            items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            idxs = items + idxs
            node = node.value
        fld = self._obj_store_base(node)  # p.v[i] = x: an array field (classes.py)
        if fld is None and not isinstance(node, ast.Name):
            self.refuse(tgt, "a store into a computed array", "Store into a named array.")
        bt = fld[1] if fld else self.var_type(node.id)
        if fld is None and node.id in self.slab_names and bt is not None and bt.is_arr():
            if not isinstance(self.slab_names[node.id], ast.Subscript):
                self.refuse(tgt, "a write through `%s`, which is a view of another array (a transpose, or a real or imaginary part)" % node.id,
                            "NumPy writes through the view into the original array; Viv's is a copy. Write into the original array.")
            what = "a slice" if self._has_range(self.slab_names[node.id]) else "a row"
            self.refuse(tgt, "a write through `%s`, which is %s of another array" % (node.id, what),
                        "Rows and slices are copies in Viv; write into the original array: `A[i, j] = ...`.")
        base = fld[0] if fld else self.vname(node.id)
        if any(isinstance(i, ast.Slice) for i in idxs) or (bt is not None and bt.is_arr() and len(idxs) < bt.rank):
            # a store into a slice or a row: `a[1:3] = v`, `A[:, j] = v`, `A[i] = v`
            for lv, items in self._levels(tgt)[:-1]:
                if any(isinstance(i, ast.Slice) for i in items):
                    self.refuse(lv, "a store into a subscript of a slice (`a[1:3][0] = v`)",
                                "NumPy writes through the slice into `a`; write `a[1:3, 0] = v` or `a[1] = v`.")
            if bt is None or not bt.is_arr():
                self.refuse(tgt, "a store into a slice of %s" % self._kind_words(bt), "Store into an array.")
            if bt.container == "list":
                self.refuse(tgt, "a store into a slice or a row of a Python list",
                            "Python replaces the list's items (the length may change); store element by element.")
            if len(idxs) > bt.rank:
                self.refuse(tgt, "%d indices into a rank-%d array" % (len(idxs), bt.rank), "Use one index per dimension.")
            codes = self._subscript_items(tgt, node, idxs, base, bt, 100)
            if not any(isinstance(i, ast.Slice) for i in idxs):
                codes.append(":")  # a row: Viv writes a range, the rest of the axes whole
            self.slice_store_rank = bt.rank - sum(1 for i in idxs if not isinstance(i, ast.Slice))
            return base, ", ".join(codes), bt
        self.slice_store_rank = 0
        codes = [self._index(i, node, k, bt) for k, i in enumerate(idxs)]
        return base, ", ".join(codes), bt

    def _index(self, i, base_node, axis, bt):
        if isinstance(i, ast.Slice):
            self.refuse(i, "a slice", "Viv has no slices yet: write the loop over the index range.", gap="R3")
        if self._is_newaxis(i) or isinstance(i, ast.Constant) and i.value is Ellipsis:
            self.refuse(i, "`None`, np.newaxis or `...` in a store target", "NumPy writes through the view it makes; store into the array's own axes.")
        code, t, _ = self.expr(i)
        if t.kind == "bool":
            self.refuse(i, "a Bool index", "Index with a number.")
        if t.is_arr():
            self.refuse(i, "fancy (array) indexing", "Write the loop, or take(A, idx) for a flat gather.")
        if t.kind not in ("num", "unknown"):
            self.refuse(i, "an index that is %s" % self._kind_words(t), "Index with an int.")
        # a negative literal index counts from the end, in Viv as in NumPy
        if isinstance(i, ast.UnaryOp) and isinstance(i.op, ast.USub) and isinstance(i.operand, ast.Constant):
            return "-%s" % i.operand.value
        return code

    def _tuple_assign(self, tgt, value, s):
        names = tgt.elts
        # w, V = np.linalg.eigh(H) (refused), q, r = divmod(a, b): unpacking rules (rules/numpy.toml)
        if isinstance(value, ast.Call) and RULES.unpack(self, tgt, value, s, self._call_name(value.func) or ""):
            return
        if self._obj_attr_targets(tgt, value, s) or self._obj_thread_stmt(value, s, names) \
                or self._obj_unpack(tgt, value, s):  # classes.py
            return
        # l, g = value_and_grad(f)(x, ...)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Call) and self._call_name(value.func.func) in ("value_and_grad", "viv.value_and_grad"):
            code, t = self._grad_call(value, "value_and_grad")
            if len(names) != 2 or not all(isinstance(n, ast.Name) for n in names):
                self.refuse(tgt, "this unpacking of value_and_grad", "Write `l, g = value_and_grad(f)(x, ...)`.")
            self.set_var(names[0].id, NUM_F, s)
            self.set_var(names[1].id, t, s)
            self.flush_pre(s)
            self.emit("%s, %s = %s" % (self.vname(names[0].id), self.vname(names[1].id), code), s)
            return
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(names) \
                and any(isinstance(n, (ast.Tuple, ast.List)) for n in names):
            self._nested_unpack(tgt, value, s)  # (a, b), c = u, 2.0 (funcs.py)
            return
        for n in names:
            if isinstance(n, ast.Starred):
                self.refuse(n, "a starred unpacking target", "Unpack into as many names as there are values.")
            if isinstance(n, (ast.Tuple, ast.List)):
                self.refuse(n, "a nested unpacking target", "Unpack into names, then unpack again on the next line.")
            if not isinstance(n, (ast.Name, ast.Subscript)):
                self.refuse(n, "this unpacking target", "Unpack into names or array elements.")
        # n, m = A.shape: Viv unpacks the array shape(A), checking its length
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            base, bt, _ = self.expr(value.value)
            if bt is not None and bt.is_arr() and bt.rank != len(names):
                self.refuse(value, "unpacking the shape of a rank-%d array into %d names" % (bt.rank, len(names)),
                            "Give one name per dimension.")
            self._unpack_into(names, "shape(%s)" % base, [NUM_I] * len(names), s)
            return
        # a, b = x, y  (parallel assignment: every right side first)
        if isinstance(value, (ast.Tuple, ast.List)):
            if len(value.elts) != len(names):
                self.refuse(value, "unpacking of a different length", "Match the counts.")
            vals = []
            all_names = all(isinstance(n, ast.Name) for n in names)
            for v in value.elts:
                if isinstance(v, ast.Starred):
                    self.refuse(v, "a starred value", "Write the values one by one.")
                vals.append(self.expr(v, keep=all_names))  # an Int, a member, an optional (ints.py)
            # sequential stores are safe when no later right side reads an earlier target
            reads = [{m.id for m in ast.walk(v) if isinstance(m, ast.Name)} for v in value.elts]
            safe = all(isinstance(n, ast.Name) for n in names)
            if safe:
                for k, n in enumerate(names):
                    if any(n.id in reads[j] for j in range(k + 1, len(names))):
                        safe = False
                        break
            if all_names and self._ie_parallel(names, vals, safe, s):
                return  # ints.py
            vals = [self._ie_lower(v, r) for v, r in zip(value.elts, vals)]
            if safe:
                for (code, t, _), n in zip(vals, names):
                    self._store(n, code, t, s)
                return
            # a swap: Viv's `a, b = b, a` computes every right side before any store
            codes = []
            for (code, t, _), n in zip(vals, names):
                if t.kind == "bool" and isinstance(n, ast.Subscript):
                    code = {"true": "1", "false": "0"}.get(code) or "%s(%s)" % (self.use("py_b01"), code)
                codes.append(code)
            self._unpack_into(names, ", ".join(codes), [t for _, t, _ in vals], s)
            return
        # a, b = f(x): a def that returns several values
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in self.funcs:
            self.tuple_site = value
            try:
                code, t, _ = self.expr(value)
            finally:
                self.tuple_site = None
            if t.kind == "tuple":
                if len(t.elts) != len(names):
                    self.refuse(value, "unpacking %d values into %d names" % (len(t.elts), len(names)), "Give one name per value.")
                self._unpack_into(names, code, t.elts, s)
                return
            if t.kind == "unknown":
                # its return type is found on a later pass
                self._unpack_into(names, code, [UNK] * len(names), s)
                return
            if not (t.kind == "arr" and t.rank == 1):
                self.refuse(value, "unpacking %s" % self._kind_words(t), "Only several returned values, a vector and a shape unpack.", gap="R5")
            self._unpack_into(names, code, [num_of(t) if t.py != "bool" else BOOL] * len(names), s)
            return
        # a, b = v: the elements of a vector (Viv checks its length)
        code, t, _ = self.expr(value)
        if t.kind == "arr" and t.rank == 1:
            et = BOOL if t.py == "bool" else (NUM_I if t.py == "int" else NUM_F)
            if t.elts:
                if len(t.elts) != len(names):
                    self.refuse(value, "unpacking %d values into %d names" % (len(t.elts), len(names)), "Give one name per value.")
                if any(e.kind == "bool" for e in t.elts):
                    self.refuse(value, "unpacking a Bool from a tuple or a list of several types",
                                "The Bool is held as 1 or 0; read it by its index (`t[0]`), which gives it back as a Bool.")
                self._unpack_into(names, code, list(t.elts), s)
                return
            self._unpack_into(names, code, [et] * len(names), s)
            return
        if t.kind == "tuple":
            self._unpack_into(names, code, t.elts, s)
            return
        self.refuse(value, "unpacking %s" % self._kind_words(t),
                    "Only several returned values, a vector (its rows for a matrix are not unpacked) and a shape unpack.", gap="R5")

    def _unpack_into(self, names, code, types, s):
        """`a, b[i] = <code>`: Viv's unpacking, each target stored with its value's type."""
        if len(names) != len(types):
            self.refuse(s, "unpacking %d values into %d names" % (len(types), len(names)), "Give one name per value.")
        targets = []
        for n, t in zip(names, types):
            if isinstance(n, ast.Name):
                if t.kind in ("none", "func", "module", "tuple"):
                    self.refuse(n, "unpacking %s into `%s`" % (self._kind_words(t), n.id), "Unpack numbers, bools, strings and arrays.")
                self.set_var(n.id, t, s)
                self.slab_names.pop(n.id, None)
                targets.append(self.vname(n.id))
            else:
                base, idx, bt = self._subscript_target(n)
                if t.kind == "bool":
                    self.refuse(n, "unpacking a bool into an array element", "Unpack into a name, then store it.")
                targets.append("%s[%s]" % (base, idx))
        self.flush_pre(s)
        self.emit("%s = %s" % (", ".join(targets), code), s)

    def _s_AugAssign(self, s, top):
        op = s.op
        tgt = s.target
        if self._obj_augassign(s):  # obj.f += v (classes.py)
            return
        saved_pre = list(self.pre)
        code, t, prec = self.expr(s.value, keep=True)
        if isinstance(tgt, ast.Name) and not isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)) and (
                t.kind in ints_mod.IE_KINDS or (self._ie_decl(tgt.id) or self.var_type(tgt.id) or UNK).kind in ints_mod.IE_KINDS):
            # x op= v with an Int, a member or an optional: x = x op v (ints.py)
            self.pre = saved_pre
            load = ast.copy_location(ast.Name(id=tgt.id, ctx=ast.Load()), tgt)
            full = ast.copy_location(ast.BinOp(left=load, op=op, right=s.value), s)
            self.parents[load] = full
            self.parents[full] = s
            fc, ft, _ = self.expr(full, keep=True)
            self._store(tgt, fc, ft, s)
            return
        code, t, prec = self._ie_lower(s.value, (code, t, prec))
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            cur, ct, _ = self.expr(tgt)
            self._store(tgt, self._bitop(op, cur, ct, code, t, ast.copy_location(ast.BinOp(left=tgt, op=op, right=s.value), s)), NUM_I, s)
            return
        if isinstance(tgt, ast.Name) and (self._is_text_type(self.var_type(tgt.id)) or self._is_text_type(t)):
            # s += t, s *= n: a new string (Python's str is immutable too); lst += [...] of a list
            # of strings changes the list in place in Python, which Viv's cannot
            ct = self.var_type(tgt.id) or UNK
            if ct.kind == "strs":
                self.refuse(s, "`%s=` of a list of strings" % self._opsym(op),
                            "Python changes the list in place (every name holding it sees it); write lst = lst + [...], "
                            "a new list.")
            full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=ast.Name(id=tgt.id, ctx=ast.Load()), op=op, right=s.value), s))
            self.set_var(tgt.id, ft, s, may_version=False)
            self.flush_pre(s)
            self.emit("%s = %s" % (self.vname(tgt.id), full), s)
            return
        if isinstance(tgt, ast.Name):
            ct = self.var_type(tgt.id) or UNK
            if ct.is_arr() and ct.container == "nd":
                # NumPy computes `x += v` into x: the result must fit x's dtype
                if ct.kind == "arr" and t.kind in ("cnum", "carr") \
                        or ct.py == "bool" \
                        or ct.py == "int" and isinstance(op, ast.Div) \
                        or ct.py == "int" and t.kind != "unknown" and (t.py not in ("int", "bool") or t.kind in ("cnum", "carr")):
                    self.refuse(s, "`%s=` of an array whose dtype cannot hold the result" % self._opsym(op),
                                "NumPy raises (it cannot cast the result back in place); make the array float.")
            aliased = ct.is_arr() and ct.container == "nd" and self.views.aliased(self.scope, tgt.id)
            # an array value of x's own shape by its form (`u += dt * np.roll(u, 1)`) needs no check
            broadcast = t.is_arr() and not (self._shape_key(tgt) is not None and self._shape_key(tgt) == self._shape_key(s.value))
            if ct.is_arr() and ct.container == "nd" and (aliased or broadcast):
                # `x += v` of an array another name, a caller or a view may hold: NumPy changes
                # the array in place, where Viv's `x += v` makes a new one and rebinds x. With an
                # array v, NumPy broadcasts it, and raises where the result would not fit x.
                full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=ast.Name(id=tgt.id, ctx=ast.Load()), op=op, right=s.value), s))
                if ft.is_arr() and ft.rank != ct.rank:
                    self.refuse(s, "`%s=` of a rank-%d array with a rank-%d result" % (self._opsym(op), ct.rank, ft.rank),
                                "NumPy raises (the broadcast result does not fit the array).")
                v = self.vname(tgt.id)
                self.flush_pre(s)
                if broadcast:
                    fit = self.use("py_cinpl" if ct.kind == "carr" else "py_inpl")
                else:
                    fit = self.use("py_cfit" if ct.kind == "carr" else "py_fit")
                self.emit("%s%s = %s(%s, %s)" % (v, "[:]" if aliased else "", fit, full, v), s)
                return
            rt = self._binop_type(op, ct, t, s)
            self.set_var(tgt.id, rt, s, may_version=False)
            self.flush_pre(s)
            plain = isinstance(op, (ast.Add, ast.Sub, ast.Mult)) and not (ct.kind == "num" and ct.py in ("int", "mixed") and t.kind in ("num", "bool") and t.py in ("int", "mixed", "bool"))
            plain = plain or (isinstance(op, ast.Div) and (ct.is_arr() or self._nonzero_literal(s.value)))
            if isinstance(op, ast.Mult) and ct.kind == "arr" and ct.container == "list":
                plain = False
            if plain and ct.kind == "arr" and ct.py == "int" and isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
                plain = False
            if plain:
                self.emit("%s %s= %s" % (self.vname(tgt.id), self._opsym(op), code), s)
            else:
                full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=ast.Name(id=tgt.id, ctx=ast.Load()), op=op, right=s.value), s))
                self.emit("%s = %s" % (self.vname(tgt.id), full), s)
            return
        if isinstance(tgt, ast.Subscript):
            # a[m] += v, a[idx] += v: NumPy reads a[idx] (a copy), adds and stores it back, so a
            # repeated index is added to once
            load = ast.copy_location(ast.Subscript(value=tgt.value, slice=tgt.slice, ctx=ast.Load()), tgt)
            fr = self._fancy(load)
            if fr is not None:
                for it in (tgt.slice.elts if isinstance(tgt.slice, ast.Tuple) else [tgt.slice]):
                    if isinstance(it, ast.Slice):
                        continue
                    ic = self.expr(it)[0]
                    for f in self._effect_names():
                        if re.search(r"(?<![\w.])%s\(" % re.escape(viv_name(f)), ic):
                            self.refuse(it, "`%s=` through indices that call `%s`" % (self._opsym(op), f),
                                        "The translation reads and writes through the indices, evaluating them twice, and "
                                        "Python once; assign them to a variable first.")
                self._xmemo[id(load)] = fr
                full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=load, op=op, right=s.value), s))
                self._store(tgt, full, ft, s)
                return
            base, idx, bt = self._subscript_target(tgt)
            if self.slice_store_rank:
                # a[1:3] += v: NumPy's in-place operation on the slice's elements
                rank = self.slice_store_rank
                if bt.py in ("int", "bool") and not isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
                    self.refuse(s, "`%s=` into a slice of an integer array" % self._opsym(op),
                                "NumPy cannot cast the result back into the integer array in place; make the array float.")
                full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=tgt, op=op, right=s.value), s))
                if ft.is_arr() and ft.rank != rank or ft.kind in ("cnum", "carr") and bt.kind != "carr":
                    self.refuse(s, "`%s=` of a slice with %s" % (self._opsym(op), self._kind_words(t)),
                                "NumPy broadcasts the value or refuses the cast; write the loop.")
                if bt.py == "int" and ft.py != "int":
                    self.refuse(s, "`%s=` into a slice of an integer array with %s" % (self._opsym(op), self._kind_words(t)),
                                "NumPy cannot cast the float result back into the integer array in place; make the array float.")
                self.flush_pre(s)
                if t.is_arr():
                    fit = self.use("py_cinpl" if bt.kind == "carr" else "py_inpl")
                else:
                    fit = self.use("py_cfit" if bt.kind == "carr" else "py_fit")
                self.emit("%s[%s] = %s" % (base, idx, "%s(%s, %s[%s])" % (fit, full, base, idx) if ft.is_arr() else full), s)
                return
            if bt is not None and bt.is_arr() and bt.container == "list":
                # the item becomes the result of the operation, of its own type
                rt = self._binop_type(op, NUM_I if bt.py == "int" else (BOOL if bt.py == "bool" else NUM_F), t, s)
                self._list_item_store(tgt, bt, rt)
            elem_int = bt is not None and bt.is_arr() and bt.py == "int"
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult)) and not elem_int or isinstance(op, ast.Div) and bt is not None and bt.container == "nd":
                # a NumPy element is a NumPy scalar: `/` by zero is inf there, not an exception
                self.flush_pre(s)
                self.emit("%s[%s] %s= %s" % (base, idx, self._opsym(op), code), s)
                return
            full, ft, _ = self.expr(ast.copy_location(ast.BinOp(left=tgt, op=op, right=s.value), s))
            self.flush_pre(s)
            self.emit("%s[%s] = %s" % (base, idx, full), s)
            return
        self.refuse(tgt, "this augmented assignment target", "Use a name or an array element.")

    @staticmethod
    def _opsym(op):
        return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
                ast.Pow: "^", ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
                ast.LShift: "<<", ast.RShift: ">>"}[type(op)]

    def _s_If(self, s, top):
        if top and self.scope is None and self._is_main_guard(s):
            self._flush_comments(s.lineno)
            self._block(s.body, top=True)
            return
        if self._ie_narrow(s):  # if x is None / is not None of an optional: a match (ints.py)
            return
        c = self._cond(s.test)
        self.flush_pre(s)
        self.emit("if %s {" % c, s)
        self._body(s.body)
        orelse = s.orelse
        while orelse:
            if len(orelse) == 1 and isinstance(orelse[0], ast.If) and self._ie_narrow_test(orelse[0].test) is not None:
                # elif x is None: a match inside the else (ints.py)
                self.emit("}", s)
                self.emit("else {", orelse[0])
                self.indent += 1
                self._ie_narrow(orelse[0])
                self.indent -= 1
                break
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                e = orelse[0]
                self.pre = []
                c = self._cond(e.test)
                if self.pre:
                    # the condition needs statements first: nest it in the else
                    self.emit("}", s)
                    self.emit("else {", e)
                    self.indent += 1
                    self.flush_pre(e)
                    self.emit("if %s {" % c, e)
                    self._body(e.body)
                    if e.orelse:
                        self.emit("}", e)
                        self.emit("else {", e)
                        self._body(e.orelse)
                    self.emit("}", e)
                    self.indent -= 1
                    self.emit("}", s)
                    return
                self.emit("}", e)
                self.emit("else if %s {" % c, e)
                self._body(e.body)
                orelse = e.orelse
                continue
            self.emit("}", s)
            self.emit("else {", orelse[0])
            self._body(orelse)
            break
        self.emit("}", s)

    def _body(self, body):
        self.indent += 1
        self._block(body)
        self.indent -= 1

    def _s_While(self, s, top):
        if s.orelse:
            self.refuse(s, "`while ... else`", "Use a flag variable set before `break`.")
        c = self._cond(s.test)
        if self.pre:
            self.refuse(s.test, "a while condition that needs statements before it (a call adapter or a conditional expression)",
                        "Compute the condition into a variable before the loop and again at the end of the body.")
        self.emit("while %s {" % c, s)
        self._body(s.body)
        self.emit("}", s)

    def _assigns_name(self, body, name):
        """Does the body rebind `name` (not merely write into an array it indexes)?"""
        def binds(t):
            if isinstance(t, ast.Name):
                return t.id == name
            if isinstance(t, (ast.Tuple, ast.List)):
                return any(binds(e) for e in t.elts)
            return False
        for st in body:
            for n in ast.walk(st):
                if isinstance(n, ast.Assign) and any(binds(t) for t in n.targets):
                    return True
                if isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For)) and binds(n.target):
                    return True
        return False

    def _s_For(self, s, top):
        if self._obj_for(s):  # for k, v in d.items(): one copy of the body per key (classes.py)
            return
        if self._funcs_for(s):
            return  # over a generator, enumerate(a, start), reversed(a), rows unpacked (funcs.py)
        if s.orelse:
            self.refuse(s, "`for ... else`", "Use a flag variable set before `break`.")
        it = s.iter
        tgt = s.target
        # for i in range(...) / reversed(range(...))
        rev = False
        if isinstance(it, ast.Call) and self._call_name(it.func) == "reversed" and len(it.args) == 1:
            rev = True
            it = it.args[0]
        if isinstance(it, ast.Call) and self._call_name(it.func) == "range":
            if not isinstance(tgt, ast.Name):
                self.refuse(tgt, "unpacking a range", "Loop over one index.")
            args = [self.expr(a) for a in it.args]
            for (c, t, _), a in zip(args, it.args):
                if t.kind not in ("num", "unknown"):
                    self.refuse(a, "a range bound that is not a number", "Pass a number.")
            name = tgt.id
            self.set_var(name, NUM_I, s)
            hidden = self._assigns_name(s.body, name)
            v = self.vname(name)
            loopvar = self.new_tmp("k") if hidden else v
            if len(args) == 1:
                head = "for %s to %s" % (loopvar, args[0][0])
            elif len(args) == 2:
                head = "for %s from %s to %s" % (loopvar, args[0][0], args[1][0])
            else:
                step = it.args[2]
                if isinstance(step, ast.Constant) and step.value == 0:
                    self.refuse(step, "a range step of 0", "Python raises here; use a nonzero step.")
                head = "for %s from %s to %s by %s" % (loopvar, args[0][0], args[1][0], args[2][0])
            if rev:
                head += " reversed"
            self.flush_pre(s)
            self.emit(head + " {", s)
            self.indent += 1
            if hidden:
                self.emit("%s = %s" % (v, loopvar), s)
            self._block(s.body)
            self.indent -= 1
            self.emit("}", s)
            return
        if rev:
            self.refuse(s.iter, "`reversed` of something other than a range", "Loop over the indices backward: `for i in range(n - 1, -1, -1)`.")
        # for i, x in enumerate(a)
        if isinstance(it, ast.Call) and self._call_name(it.func) == "enumerate":
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts)):
                self.refuse(tgt, "this enumerate target", "Write `for i, x in enumerate(a)`.")
            if len(it.args) != 1 or it.keywords:
                self.refuse(it, "enumerate with a start", "Add the offset to the index yourself.")
            self._for_over([tgt.elts[1]], [it.args[0]], s, index_name=tgt.elts[0].id)
            return
        # for a, b in zip(x, y)
        if isinstance(it, ast.Call) and self._call_name(it.func) == "zip":
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == len(it.args)):
                self.refuse(tgt, "this zip target", "Unpack one name per zipped sequence.")
            self._for_over(tgt.elts, it.args, s)
            return
        # for x in (1.0, 2.0) / for h, n in ((0.05, 1), (1.0, 2))
        if isinstance(it, (ast.Tuple, ast.List)):
            self._for_literal(tgt, it, s)
            return
        self._for_over([tgt], [it], s)

    def _for_over(self, targets, seqs, s, index_name=None):
        """Loop over the leading axis of arrays (or ranges) in lockstep."""
        codes = []
        for q in seqs:
            if isinstance(q, ast.Call) and self._call_name(q.func) == "range":
                codes.append(("range", q))
                continue
            fl = self._lines_of_file(q)  # open(p), or a `with` file: its lines (strs.py)
            if fl is not None:
                code, t = fl, STRS
            else:
                code, t, _ = self.expr(q)
            if not t.is_arr() and t.kind not in ("str", "strs"):
                self.refuse(q, "iterating over %s" % self._kind_words(t), "Loop over `range(len(a))` and index.")
            if not isinstance(q, ast.Name):
                tmp = self.new_tmp("s")
                self.pre.append("%s = %s" % (tmp, code))
                code = tmp
            codes.append((code, t))
        k = self.vname(index_name) if index_name else self.new_tmp("i")
        if index_name:
            self.set_var(index_name, NUM_I, s)
        # the loop count: the shortest sequence (zip), a range's own count
        counts = []
        range_exprs = []
        for c in codes:
            if c[0] == "range":
                q = c[1]
                a = [self.expr(x)[0] for x in q.args]
                if len(a) == 1:
                    counts.append(a[0])
                    range_exprs.append(("0", "1"))
                elif len(a) == 2:
                    counts.append("max(%s - %s, 0)" % (a[1], a[0]))
                    range_exprs.append((a[0], "1"))
                else:
                    counts.append("max(ceil((%s - %s) / %s), 0)" % (a[1], a[0], a[2]))
                    range_exprs.append((a[0], a[2]))
            else:
                counts.append("len(%s)" % c[0])
                range_exprs.append(None)
        count = counts[0]
        for c in counts[1:]:
            count = "min(%s, %s)" % (count, c)
        self.flush_pre(s)
        self.emit("for %s to %s {" % (k, count), s)
        self.indent += 1
        for tg, c, rx, seq in zip(targets, codes, range_exprs, seqs):
            if not isinstance(tg, ast.Name):
                self.refuse(tg, "this loop target", "Loop over one name per sequence.")
            if c[0] == "range":
                start, step = rx
                val = k if (start == "0" and step == "1") else "%s + %s * %s" % (start, k, step)
                self.set_var(tg.id, NUM_I, s)
                self.emit("%s = %s" % (self.vname(tg.id), val), s)
            else:
                code, t = c
                if t.kind in ("str", "strs"):
                    # a string's characters, a list's strings (strs.py)
                    self.set_var(tg.id, STR, s)
                    self.emit("%s = %s[%s]" % (self.vname(tg.id), code, k), s)
                    continue
                if t.rank == 1:
                    et = BOOL if t.py == "bool" and t.container == "list" else (NUM_I if t.py == "int" else (CNUM if t.kind == "carr" else NUM_F))
                    if et.kind == "bool":
                        et = NUM_I
                    self.set_var(tg.id, et, s)
                    self.emit("%s = %s[%s]" % (self.vname(tg.id), code, k), s)
                else:
                    self._row_alias_check(seq, s, names=[tg.id])
                    self.set_var(tg.id, row_type(t), s)
                    self.emit("%s = slab(%s, %s)" % (self.vname(tg.id), code, k), s)
                    self.slab_names[tg.id] = s
        self._block(s.body)
        self.indent -= 1
        self.emit("}", s)

    def _for_literal(self, tgt, it, s):
        """for x in (a, b, c): the values become an array literal, indexed per pass."""
        rows = it.elts
        if not rows:
            return
        if isinstance(tgt, ast.Name) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in rows):
            # for w in ("a", "b"): a loop over a list of strings (strs.py)
            tmp = self.new_tmp("v")
            k = self.new_tmp("i")
            self.flush_pre(s)
            self.emit("%s = [%s]" % (tmp, ", ".join(self._str_lit(e.value, e) for e in rows)), s)
            self.set_var(tgt.id, STR, s)
            self.emit("for %s to %d {" % (k, len(rows)), s)
            self.indent += 1
            self.emit("%s = %s[%s]" % (self.vname(tgt.id), tmp, k), s)
            self._block(s.body)
            self.indent -= 1
            self.emit("}", s)
            return
        if not self._all_numeric_rows(tgt, rows):
            self._unroll(tgt, rows, s)
            return
        if isinstance(tgt, ast.Name):
            vals = [self.expr(e) for e in rows]
            if not all(t.kind == "num" for _, t, _ in vals):
                self.refuse(it, "a loop over values that are not all numbers", "Loop over an index and select with if/elif.")
            tmp = self.new_tmp("v")
            k = self.new_tmp("i")
            self.flush_pre(s)
            self.emit("%s = [%s]" % (tmp, ", ".join(self._as_float_lit(c) for c, _, _ in vals)), s)
            ty = NUM_I if all(t.py == "int" for _, t, _ in vals) else NUM_F
            self.set_var(tgt.id, ty, s)
            self.emit("for %s to %d {" % (k, len(rows)), s)
            self.indent += 1
            self.emit("%s = %s[%s]" % (self.vname(tgt.id), tmp, k), s)
            self._block(s.body)
            self.indent -= 1
            self.emit("}", s)
            return
        if isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts):
            w = len(tgt.elts)
            cols = [[] for _ in range(w)]
            for r in rows:
                if not (isinstance(r, (ast.Tuple, ast.List)) and len(r.elts) == w):
                    self.refuse(r, "a row of a different length", "Give every row the same number of values.")
                for j, e in enumerate(r.elts):
                    cols[j].append((e, self.expr(e)))
            for j, col in enumerate(cols):
                if not all(t.kind == "num" for _, (c, t, _) in col):
                    self.refuse(col[0][0], "a loop over values that are not all numbers (a string, an array)",
                                "Loop over an index and select the non-numeric item with if/elif.")
            k = self.new_tmp("i")
            self.flush_pre(s)
            tmps = []
            for j, col in enumerate(cols):
                tmp = self.new_tmp("v")
                self.emit("%s = [%s]" % (tmp, ", ".join(self._as_float_lit(c) for _, (c, _, _) in col)), s)
                ty = NUM_I if all(t.py == "int" for _, (c, t, _) in col) else NUM_F
                tmps.append((tmp, ty))
            self.emit("for %s to %d {" % (k, len(rows)), s)
            self.indent += 1
            for (tmp, ty), n in zip(tmps, tgt.elts):
                self.set_var(n.id, ty, s)
                self.emit("%s = %s[%s]" % (self.vname(n.id), tmp, k), s)
            self._block(s.body)
            self.indent -= 1
            self.emit("}", s)
            return
        self.refuse(tgt, "this loop target", "Loop over one name, or a tuple of names.")

    @staticmethod
    def _as_float_lit(c):
        return c

    def _all_numeric_rows(self, tgt, rows):
        def numeric(e):
            if isinstance(e, ast.Constant):
                return isinstance(e.value, (int, float)) and not isinstance(e.value, bool)
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
                return numeric(e.operand)
            if isinstance(e, (ast.List, ast.Tuple, ast.Call, ast.Dict, ast.JoinedStr)):
                return False
            return True  # a name or an arithmetic expression: typed when translated
        if isinstance(tgt, ast.Name):
            return all(numeric(r) for r in rows)
        return all(isinstance(r, (ast.Tuple, ast.List)) and all(numeric(e) for e in r.elts) for r in rows)

    def _unroll(self, tgt, rows, s):
        """for name, A in (("label", np.array(...)), ...): a few literal rows holding strings or
        arrays. The body is written out once per row: a string is substituted where it is used
        (it can only be printed), anything else is assigned to the loop name first."""
        if len(rows) > 16:
            self.refuse(s.iter, "a loop over more than 16 literal rows that are not all numbers", "Loop over an index and select with if/elif.")
        for st in s.body:
            for m in ast.walk(st):
                if isinstance(m, (ast.Break, ast.Continue)):
                    self.refuse(m, "`%s` in a loop over literal rows (the loop is unrolled)" % ("break" if isinstance(m, ast.Break) else "continue"),
                                "Use a flag and an if.")
        names = [tgt] if isinstance(tgt, ast.Name) else list(tgt.elts) if isinstance(tgt, (ast.Tuple, ast.List)) else None
        if names is None or not all(isinstance(n, ast.Name) for n in names):
            self.refuse(tgt, "this loop target", "Loop over one name, or a tuple of names.")
        import copy
        for r in rows:
            elts = [r] if len(names) == 1 else (r.elts if isinstance(r, (ast.Tuple, ast.List)) else None)
            if elts is None or len(elts) != len(names):
                self.refuse(r, "a row of a different length", "Give every row the same number of values.")
            subst = {}
            for n, e in zip(names, elts):
                if isinstance(e, ast.Constant) and isinstance(e.value, str) and self._str_only_printed(s.body, n.id):
                    subst[n.id] = e
                else:
                    self.pre = []
                    self._assign_to(n, e, s)
            body = [copy.deepcopy(b) for b in s.body]
            if subst:
                class Sub(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Load) and node.id in subst:
                            c = ast.copy_location(ast.Constant(value=subst[node.id].value), node)
                            c._py2viv_subst = True  # Python reads a variable here (diff.py's read-back)
                            return c
                        return node
                body = [Sub().visit(b) for b in body]
            self.emit("# %s = %s" % (", ".join(n.id for n in names), (ast.get_source_segment(self.src, r) or "")[:100].replace("\n", " ")), s)
            self._block(body)

    # ------------------------------------------------ print
    def _print(self, call, s):
        parts = []
        sep = " "
        end = "\n"
        for k in call.keywords:
            if k.arg == "sep" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
                sep = k.value.value
            elif k.arg == "end" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
                end = k.value.value
            elif k.arg == "file":
                self.refuse(k.value, "print to a file", "Print to standard output.")
            elif k.arg == "flush":
                pass
            else:
                self.refuse(k.value, "print's `%s=`" % k.arg, "Use a string literal for sep and end.")
        if end != "\n" and self.opt.print_mode == "exact":
            self.refuse(call, "print's end=%r" % end, "Every Viv print ends its line: build the line in one print.", gap="R4")
        for text in [sep] + [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]:
            self._str_lit(text, call)  # a literal Viv can write (strs.py: the Unicode check)
        values = []  # (code, type, node)
        for i, a in enumerate(call.args):
            if i:
                parts.append({"kind": "str", "text": sep})
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                parts.append({"kind": "str", "text": a.value})
                if getattr(a, "_py2viv_subst", False):
                    parts[-1]["subst"] = True  # Python printed a variable's value here (diff.py)
                continue
            if isinstance(a, ast.JoinedStr):
                self._fstring(a, parts, values)
                continue
            if isinstance(a, ast.Starred):
                # print(*words): the strings, sep between them (a list of strings only; strs.py)
                if len(call.args) != 1:
                    self.refuse(a, "a starred argument of print beside other arguments",
                                "Python writes a separator only between the items there are (none for an empty list); "
                                "print the list alone, or join it into one string.")
                code, t, _ = self.expr(a.value)
                if t.kind != "strs":
                    self.refuse(a, "a starred argument of print of %s" % self._kind_words(t),
                                "Python prints each item; loop over them, or join their text: \" \".join(str(x) for x in xs).")
                self._print_value(a, "join(%s, %s)" % (viv_str(sep), code), STR, "str", None, parts, values)
                parts[-1]["sub"] = True  # no value of its own in Python's record (diff.py reads it as text)
                continue
            if self._print_text(a, parts, values):
                continue  # "text " + str(x) + f"...", str(x) (funcs.py)
            render = "str"
            node = a
            if isinstance(a, ast.Call) and self._call_name(a.func) == "repr" and len(a.args) == 1:
                render = "repr"
                node = a.args[0]
            elif isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "tolist" and not a.args:
                render = "tolist"
                node = a.func.value
            code, t, _ = self.expr(node, keep=True)  # an Int, a member (ints.py)
            self._print_value(node, code, t, render, None, parts, values)
        parts.append({"kind": "str", "text": end})
        site = len(self.sites)
        self.sites.append({"site": site, "line": call.lineno, "col": call.col_offset, "parts": parts})
        self.flush_pre(s)
        if self.opt.print_mode == "plain":
            if not values:
                text = "".join(p["text"] for p in parts if p["kind"] == "str").strip()
                self.emit('narrate "%s"' % text.replace('"', "'"), s)
            for code, t, node, k in values:
                if t.kind in ("num", "bool", "cnum") and isinstance(node, ast.Name):
                    self.emit("show %s" % code, s)
                elif t.kind == "num":
                    self.emit("print(%s)" % code, s)
                else:
                    self.emit("show %s" % code, s)
            return
        # one Viv print per Python print: its text, and each value in the form fmt writes it
        # which is Python's own text for a float (repr), an int, a list
        # and a format spec of the table; diff.py reads the rest (bools, complex numbers, NumPy
        # arrays) back from the line and prints them the way Python does.
        val_of = {k: (code, t, node) for code, t, node, k in values}
        body = parts[:-1]  # the end ("\n") is the line's own newline
        kinds = [val_of[p["k"]][1].kind for p in body if p["kind"] == "val"]
        n_bool = sum(1 for p in body if p["kind"] == "val" and val_of[p["k"]][1].kind == "bool" and p.get("spec") is None)
        bools_as_text = n_bool > 3  # (more than three: each written as True or False by py_bstr, not one print per case)
        if n_bool and not bools_as_text or "cnum" in kinds or "carr" in kinds:
            # every value computed first, in order, as Python computes the arguments
            for p in body:
                if p["kind"] == "val":
                    code, t, node = val_of[p["k"]]
                    if not re.fullmatch(r"[A-Za-z_]\w*|-?[0-9.]+(e[-+]?[0-9]+)?|true|false", code):
                        tmp = self.new_tmp("v")
                        self.emit("%s = %s" % (tmp, code), s)
                        val_of[p["k"]] = (tmp, t, node)
        items = []  # ("text", str) | ("val", forms) | ("bool", code)
        for p in body:
            if p["kind"] == "str":
                items.append(("text", p["text"]))
                continue
            code, t, node = val_of[p["k"]]
            if t.kind == "bool" and p.get("spec") is None and bools_as_text:
                items.append(("val", ["%s(%s)" % (self.use("py_bstr"), code)]))
            elif t.kind == "bool" and p.get("spec") is None:
                items.append(("bool", code))
            else:
                items.append(("val", self._print_forms(node, code, t, p)))

        def line(truth):
            args, prev_val = [], False
            for kind, x in items:
                if kind == "bool":
                    kind, x = "text", ("True" if truth[x] else "False")
                if kind == "text":
                    if not x:
                        continue
                    if args and args[-1].startswith('"') and not prev_val:
                        args[-1] = viv_str(unquote_viv(args[-1]) + x)
                    else:
                        args.append(viv_str(x))
                    prev_val = False
                    continue
                if prev_val:
                    args.append('""')  # two values in a row, with nothing between (print_call)
                args.extend(x)
                prev_val = not x[-1].startswith('"')
            return print_call(args)

        bools = [x for kind, x in items if kind == "bool"]
        # a bool prints True or False, as Python does: one print per combination

        def emit_for(k, truth):
            if k == len(bools):
                self.emit(line(truth), s)
                return
            b = bools[k]
            if b in truth:  # the same variable printed twice
                emit_for(k + 1, truth)
                return
            if b in ("true", "false"):  # a Bool known here (`"k" in d`)
                emit_for(k + 1, dict(truth, **{b: b == "true"}))
                return
            self.emit("if %s {" % b, s)
            self.indent += 1
            emit_for(k + 1, dict(truth, **{b: True}))
            self.indent -= 1
            self.emit("}", s)
            self.emit("else {", s)
            self.indent += 1
            emit_for(k + 1, dict(truth, **{b: False}))
            self.indent -= 1
            self.emit("}", s)
        emit_for(0, {})

    def _print_forms(self, node, code, t, part):
        """The Viv print arguments that write one Python print value (see _print)."""
        spec = part.get("spec")
        kind = t.kind
        if kind == "map":
            return self._map_print_forms(node, code, t)  # maps.py
        if kind in ("str", "strs"):
            return self._str_print_forms(node, code, t, part)  # strs.py
        if kind == "int":
            kind = "num"  # an Int prints as a Python int: fmt(n, "d"), exactly (ints.py)
        if spec is not None:
            if kind not in ("num",):
                self.refuse(node, "a format spec on %s" % self._kind_words(t), "Format numbers only.", gap="R4")
            if t.py == "mixed" and not re.search(r"[efg]$", spec):
                self.refuse(node, "the format spec `%s` on a number that is an int on some paths and a float on others" % spec,
                            "Python writes an int and a float differently under this spec; convert it with float(...) first.")
            d = strs_mod.parse_spec(spec)
            if d is not None and (d["align"] or d["fill"]):
                # a fill and an alignment: the number's text, then padded (strs.py)
                return [self._num_spec(node, code, t, d, spec)]
            vspec = viv_spec(spec, t.py == "int")
            if vspec is None:
                self.refuse(node, "the format spec `%s`" % spec,
                            "Viv's fmt takes [width][.precision] with type g, e, f or d: "
                            "no fill, alignment, sign, grouping, `%`, `n`, `#` or upper case.", gap="R4")
            return ["fmt(%s, %s)" % (self._intfix(code, t), viv_str(vspec))]
        if kind == "num":
            if t.py == "int":
                return ["fmt(%s, \"d\")" % self._intfix(code, t)]
            if t.py == "mixed":
                self.refuse(node, "printing a number that is an int on some paths and a float on others",
                            "Python prints the two differently (3 or 3.0); convert it with float(...) or int(...) first.")
            return ["fmt(%s)" % code]
        if kind == "cnum":
            return ['"("', "fmt(re(%s))" % code, '","', "fmt(im(%s))" % code, '")"']
        if kind == "arr":
            if t.name == "ndrows":
                self.refuse(node, "printing a list of NumPy arrays", "Python writes each row as `array([...])`; print "
                            "the rows one by one, or make them lists (.tolist()).")
            if t.elts or t.py == "mixed":
                self._elts_print(node, t)  # tuples, and lists of ints beside floats (funcs.py)
            spec_d = t.py in ("int", "bool")
            return ["fmt(%s%s)" % (code, ', "d"' if spec_d else "")]
        if kind == "carr":
            if t.rank != 1:
                self.refuse(node, "printing a complex array of rank > 1", "Print its real and imaginary parts row by row.")
            return ['"("', "fmt(re(%s))" % code, '","', "fmt(im(%s))" % code, '")"']
        self.refuse(node, "printing %s" % self._kind_words(t), "Print numbers, bools and arrays of numbers.")

    @staticmethod
    def _intfix(code, t):
        """An int for `fmt(x, "d")`: `+ 0.0` so that a zero computed as -0.0 prints `0`, as
        Python's int does. A Viv Int (ints.py) has no -0 and is written as itself."""
        if t.kind == "int":
            return code
        if t.py == "int":
            return "%s + 0.0" % code if code.replace("_", "").isalnum() else "(%s) + 0.0" % code
        return code

    def _print_value(self, node, code, t, render, spec, parts, values):
        if t.kind == "rec":
            return self._obj_print(node, code, t, render, spec, parts, values)  # classes.py
        if t.kind == "map":
            return self._map_print(node, code, t, render, spec, parts, values)  # maps.py
        if t.kind == "unknown":
            r = Refusal(node, "printing a value of unknown type (it comes from a construct refused elsewhere)",
                        "Fix the refusal that produced it.")
            r.cascade = True
            raise r
        if t.kind == "none":
            self.refuse(node, "`None` as a value", "Use a number (for example -1 or nan) to mean 'nothing'.")
        if t.kind == "enum":
            return self._ie_print_enum(node, code, t, render, spec, parts, values)  # `C.A`, as Python prints it (ints.py)
        if t.kind == "opt":
            self.refuse(node, "printing %s" % self._ie_words(t),
                        "Python prints None or the value; print it inside `if x is not None:` (py2viv translates that "
                        "test into a match) and print None in the else.")
        if t.kind in ("func", "module", "tuple"):
            self.refuse(node, "printing %s" % ("a tuple" if t.kind == "tuple" else "a " + t.kind),
                        "Print numbers, bools and arrays of numbers.", gap="R5" if t.kind == "tuple" else None)
        k = len(values)
        parts.append({"kind": "val", "k": k, "type": self._type_json(t), "render": render, "spec": spec,
                      "expr": ast.get_source_segment(self.src, node) or ""})
        values.append((code, t, node, k))

    def _fstring(self, js, parts, values):
        for v in js.values:
            if isinstance(v, ast.Constant):
                parts.append({"kind": "str", "text": v.value})
                continue
            spec = None
            if v.format_spec is not None:
                if not all(isinstance(x, ast.Constant) for x in v.format_spec.values):
                    self.refuse(v.format_spec, "a computed format spec", "Write the spec literally.")
                spec = "".join(x.value for x in v.format_spec.values)
            if v.conversion == 97:
                self.refuse(v, "the conversion `!a`", "Python escapes every character beyond ASCII; use !r for ASCII text.")
            render = {-1: "str", 114: "repr", 115: "str"}.get(v.conversion, "str")
            node = v.value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tolist" and not node.args:
                render = "tolist"
                node = node.func.value
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # a string constant: its text as Python formats it, conversion and spec applied
                conv = {-1: None, 114: "r", 115: "s"}[v.conversion]
                parts.append({"kind": "str", "text": self._const_field(node.value, conv, spec or "", node)})
                if getattr(node, "_py2viv_subst", False):
                    parts[-1]["subst"] = True  # Python formatted a variable's value here (diff.py)
                continue
            code, t, _ = self.expr(node, keep=True)  # an Int, a member (ints.py)
            self._print_value(node, code, t, render, spec, parts, values)

    @staticmethod
    def _type_json(t):
        if t.kind == "int":
            return {"kind": "num", "py": "int"}  # a Viv Int prints as a Python int (ints.py)
        if t.kind == "num":
            return {"kind": "num", "py": t.py}
        if t.kind == "bool":
            return {"kind": "bool"}
        if t.kind == "cnum":
            return {"kind": "cnum"}
        return {"kind": t.kind, "rank": t.rank, "dtype": t.py, "container": t.container}

    # ------------------------------------------------ calls as statements
    def _call_stmt(self, call, s):
        """In-place method calls and other calls whose Viv is a statement; None if not one."""
        f = call.func
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and (f.value.id in self.np or f.value.id in self.math):
                return None
            if self._rec_info(self._static_type(f.value)) is not None or self._map_info(self._static_type(f.value)) is not None:
                return None  # a method of an object (classes.py) or a dict (maps.py), whatever its name
            # a.sort(), a.fill(v), lst.append(x) of a list py2viv grows (R17), the other list
            # methods (refused): rules/numpy.toml
            return RULES.statement(self, call)

    def _grow_append(self, lst, x, t, code):
        """The rule `list.append-statement` (rules/numpy.toml), through the hook grow_append:
        x is a number, translated to `code` of type t."""
        key = (self._scope_of(lst.id), lst.id)
        self._seed_param_list(lst.id, key)
        if t.kind == "str" or self.grow_py.get(key) == "str":
            # a list of strings (strs.py): Viv's append of a string gives a new list, as for numbers
            if t.kind != "str":
                self.refuse(x, "%s appended to a list of strings" % self._kind_words(t),
                            "Python keeps both; a Viv list holds strings only, or numbers only.")
            if self.grow_shape.get(key, ("num",))[0] != "num" or self.grow_py.setdefault(key, "str") != "str":
                self.refuse(x, "a string appended to a list of numbers", "Python keeps both; keep them in two lists.")
            self.set_var(lst.id, STRS, x)
            n = self.vname(lst.id)
            return "%s = append(%s, %s)" % (n, n, code)
        if t.kind != "num":
            return self._grow_row(lst, x, t, code)  # a row, or a tuple (funcs.py)
        if self.grow_shape.get(key, ("num",))[0] != "num":
            self.refuse(x, "a number appended to a list of rows", "Python keeps both; keep numbers and rows in two lists.")
        if self.grow_py.setdefault(key, t.py) != t.py:
            self.refuse(x, "a list of ints and floats", "Python prints each as it is; make them all floats.")
        self.set_var(lst.id, T("arr", 1, t.py, "list"), x)
        n = self.vname(lst.id)
        return "%s = append(%s, %s)" % (n, n, code)

    def _seed_param_list(self, name, key):
        """A list parameter a def changes (infer.py) holds the items its calls pass: the items
        added to it must be of their type (a list of ints stays one)."""
        if self.scope is None or name not in self.scope.params or key in self.grow_py or key in self.grow_shape:
            return
        t = self.var_type(name)
        if t is not None and t.kind == "arr" and t.container == "list":
            if t.rank == 1 and not t.elts:
                self.grow_py[key] = t.py
            elif t.rank == 2:
                self.grow_shape[key] = ("rows", t.py, tuple(t.elts) if t.elts else None, t.name)

    def _grow_body(self):
        return self.scope.node.body if self.scope is not None else [st for st in self.tree.body if not isinstance(st, ast.FunctionDef)]

    def _grow_refusal(self, name):
        """None when the list `name` names is one Viv's `append` grows exactly: bound only to
        `[]` or a list literal, and read only where no alias of it can be taken (an alias would
        see Python's append and not Viv's). Else (node, why) for the first use that is not."""
        body = self._grow_body()
        if self.scope is None and name.id in self.global_reads:
            return name, "at module level, where a def reads it"
        if self.scope is not None and name.id in self.scope.params:
            if not self._param_fresh(self.scope, self.scope.params.index(name.id)):
                return name, "and a parameter (Python changes the caller's list; Viv's append would rebind the parameter alone)"
            # every call passes a new list no other name holds (infer.py): the caller cannot see
            # the change, so the parameter is the list's only name
            return grow_refusal(body, name.id, set(self.funcs), self.np)
        if not any(isinstance(m, ast.Name) and m.id == name.id and isinstance(m.ctx, ast.Store)
                   for st in body for m in ast.walk(st)):
            return name, "and bound outside this scope (Python changes that list; Viv's append would bind a new one here)"
        return grow_refusal(body, name.id, set(self.funcs), self.np)

    # ------------------------------------------------ conditions
    def _cond(self, node):
        self.cond_ok = getattr(self, "cond_ok", set())
        stack = [node]
        while stack:  # `and`/`or` whose value only decides a branch: truthiness is enough
            x = stack.pop()
            if isinstance(x, ast.BoolOp):
                self.cond_ok.add(id(x))
                stack.extend(x.values)
            elif isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not):
                stack.append(x.operand)
        code, t, prec = self.expr(node, cond=True, keep=True)
        return self._truth(node, code, t, prec)

    def _truth(self, node, code, t, prec):
        if t.kind == "bool":
            return code
        if t.kind in ints_mod.IE_KINDS:
            return self._ie_truth(node, code, t, prec)  # ints.py
        if t.kind == "num":
            return "%s != 0" % self._paren(code, prec, 51)
        if t.is_arr():
            if t.container == "list":
                return "len(%s) > 0" % code
            self.refuse(node, "the truth value of an array", "NumPy refuses it too: write np.all(a) or np.any(a).")
        if t.kind == "unknown":
            return code
        if t.kind == "map":
            return "len(%s) > 0" % code  # a dict is true when it has a key (maps.py)
        if t.kind in ("str", "strs"):
            return "len(%s) > 0" % code  # an empty string or list is false
        self.refuse(node, "the truth value of %s" % self._kind_words(t), "Compare explicitly.")

    # ------------------------------------------------ expressions
    # precedence (higher binds tighter): atom 100, ^ 90, unary 80, * / @ % // 70, + - 60,
    # comparison 50, and/or 40
    @staticmethod
    def _paren(code, prec, need):
        return code if prec >= need else "(%s)" % code

    def _expr_code(self, node):
        return self.expr(node)[0]

    def expr(self, node, cond=False, target_hint=None, keep=False):
        """(code, type, precedence) of an expression. An Int, an enum member or an optional is
        kept as it is only where the caller asks for it (keep) and translates it (ints.py);
        anywhere else an Int becomes a number exactly and a member or an optional is refused."""
        memo = getattr(self, "_xmemo", None)
        if memo:
            r = memo.pop(id(node), None)
            if r is not None:
                # translated once already, to see whether a subscript item is an array
                return r if keep else self._ie_lower(node, r)
        mark = getattr(node, funcs.REFUSE, None)
        if mark is not None:
            self.refuse(node, *mark)
        m = getattr(self, "_e_" + type(node).__name__, None)
        if m is None:
            self._refuse_expr(node)
        r = m(node)
        if r[1] is UNK or r[1] == UNK:
            r = self._taint_expr(node, r)  # unknown because a part of it comes from a refusal (infer.py)
        if r[1] is not None and r[1].kind == "rng":
            self._rng_use(node)
        if r[1] is not None and r[1].kind in ("rec", "func"):
            self._obj_use(node, r[1])  # an object, or a class (classes.py)
        if not keep:
            r = self._ie_lower(node, r)  # ints.py
        return r

    def _rng_use(self, node):
        """A Generator (np.random.default_rng) is the state array of lib/nprandom: it may be
        bound to a name, passed to a def, returned, and drawn from (a method call); anything else
        (arithmetic, a container, a comparison, a conditional expression) is refused."""
        p = self.parents.get(node)
        if isinstance(p, ast.Assign) and p.value is node and all(isinstance(t, ast.Name) for t in p.targets):
            return
        if isinstance(p, ast.Attribute) and p.value is node:
            return  # a method call; any other attribute is refused where it is translated
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "print":
            return  # refused by print, with its own message
        if isinstance(p, ast.Call) and node in p.args and isinstance(p.func, ast.Name) and p.func.id in self.funcs:
            return
        if isinstance(p, ast.Return) and p.value is node:
            return
        if p is None:
            return  # a node the translator made (a default argument), not the program's
        self.refuse(node, "a Generator (np.random.default_rng) used as a value here",
                    "A Generator can be bound to a name, passed to a def, returned, and drawn from (rng.random(...)).")

    def _refuse_expr(self, node):
        name = type(node).__name__
        table = {
            "ListComp": ("this list comprehension", "Write the loop."),
            "GeneratorExp": ("a generator expression used as a value",
                             "A generator is read by sum, any, all, min, max, sorted, next, list, a for loop or a comprehension; "
                             "build a list with a comprehension."),
            "SetComp": ("a set comprehension", "Viv arrays hold numbers; key by integer index."),
            "DictComp": ("a dict comprehension", "Viv arrays hold numbers; key by integer index."),
            "Dict": ("a dict", "Viv arrays hold numbers; key by integer index."),
            "Set": ("a set", "Viv arrays hold numbers; use an array of 0/1 flags indexed by the element."),
            "Lambda": ("a lambda used as a value", "A lambda is translated when it is called, bound once to a name, passed to a "
                       "def, or read by a key, map or filter; Viv holds no function in a list or a variable: write a def."),
            "Yield": ("`yield`", "Viv has no generators; fill an array."),
            "YieldFrom": ("`yield from`", "Viv has no generators; fill an array."),
            "Await": ("`await`", "Not in the numeric subset."),
            "NamedExpr": ("the walrus operator", "Assign on its own line first."),
            "Starred": ("a starred expression", "Pass the values one by one."),
            "Slice": ("a slice", "Viv has no slices yet: write the loop over the index range."),
        }
        what, hint = table.get(name, ("a `%s` expression" % name, "Not in the numeric subset (tools/py2viv/SUBSET.md)."))
        self.refuse(node, what, hint, gap="R3" if name == "Slice" else None)

    def _e_Constant(self, n):
        v = n.value
        if isinstance(v, bool):
            return ("true" if v else "false"), BOOL, 100
        if isinstance(v, int):
            if abs(v) > MAX_EXACT_INT and abs(v) < 2 ** 63 and self._ie_big_literal_ok(n):
                return str(v), INT, 100  # an Int literal where an Int is bound or computed (ints.py)
            if abs(v) > MAX_EXACT_INT:
                self.refuse(n, "an integer literal above 2^53", "Viv numbers are float64; above 2^53 they are inexact (Int is typed_array_elements.md).")
            return str(v), NUM_I, 100
        if isinstance(v, float):
            if v != v:
                return "(0.0 / 0.0)", NUM_F, 100
            if v in (float("inf"), float("-inf")):
                return ("(1.0 / 0.0)" if v > 0 else "(-1.0 / 0.0)"), NUM_F, 100
            r = repr(v)
            return r, NUM_F, 100
        if isinstance(v, complex):
            if v.real:
                return "(%r + %ri)" % (v.real, v.imag), CNUM, 100
            return "%ri" % v.imag, CNUM, 100
        if isinstance(v, str):
            return self._str_lit(v, n), STR, 100
        if v is None:
            return "0", NONE, 100
        self.refuse(n, "this constant", "Use a number.")

    def _e_Name(self, n):
        name = n.id
        c = self._comp_get(name)
        if c is not None:
            # a comprehension's or a lambda's own variable (funcs.py)
            code, t = c
            return code, t, 100 if re.fullmatch(r"[A-Za-z_]\w*|\d+(\.\d+)?|true|false", code) else 50
        if name in self.refused_mods and self.var_type(name) is None:
            self.refuse(n, "a use of `%s` (from `%s`, refused at its import)" % (name, self.refused_mods[name]),
                        REFUSED_MODULE_HINT.get(self.refused_mods[name].split(".")[0], "Not in the numeric subset."))
        if name in self.funcs:
            return viv_name(name), T("func", name=name), 100
        if name in self.np or name in self.math:
            return name, T("module", name=name), 100
        if name in self.from_np:
            path = self.from_np[name]
            if path in ("pi", "e", "inf", "nan"):
                return self._const_named(path, n)
            return name, T("func", name="np." + path), 100
        if name in self.from_math:
            path = self.from_math[name]
            if path in ("pi", "e", "inf", "nan", "tau"):
                return self._const_named(path, n)
            return name, T("func", name="math." + path), 100
        if name in self.from_viv:
            return name, T("func", name=name), 100
        if name in ("True", "False"):
            return ("true" if name == "True" else "false"), BOOL, 100
        self._ie_class_value(n)  # an enum class as a value: refused (ints.py)
        t = self.var_type(name)
        if t is None:
            if name in ("abs", "min", "max", "len", "sum", "range", "float", "int", "round", "print",
                        "enumerate", "zip", "reversed", "sorted", "list", "pow", "repr", "all", "any"):
                return name, T("func", name=name), 100
            if name in ("__name__",):
                self.refuse(n, "`__name__`", "Only `if __name__ == \"__main__\":` at module level is understood.")
            t = UNK_C if self._tainted_name(name) else UNK
        return self.vname(name), t, 100

    def _const_named(self, path, n):
        import math
        v = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": float("inf"), "nan": float("nan")}[path]
        return self._e_Constant(ast.copy_location(ast.Constant(value=v), n))

    def _e_List(self, n):
        return self._literal(n)

    def _e_Tuple(self, n):
        # a tuple in an expression: a shape, or a literal vector
        return self._literal(n, tuple_ok=True)

    def _literal(self, n, tuple_ok=False):
        if not n.elts:
            self.refuse(n, "an empty list", "py2viv grows a list only as `lst = []` then `lst.append(x)`; otherwise allocate the final size with np.zeros(n) and fill it.", gap="R17")
        r = self._funcs_literal(n)  # tuples, and ints beside floats: each item keeps its type (funcs.py)
        if r is not None:
            return r
        items = [self.expr(e) for e in n.elts]
        kinds = {t.kind for _, t, _ in items}
        if ("str" in kinds or "strs" in kinds) and isinstance(n, ast.Tuple):
            self.refuse(n, "a tuple holding a string", "Python prints a tuple with parentheses and keeps it apart from a "
                        "list; use a list of strings, or loop over the tuple where it is written.")
        if kinds == {"str"} or kinds == {"str", "unknown"}:
            return "[%s]" % ", ".join(c for c, _, _ in items), STRS, 100  # a list of strings
        if "str" in kinds or "strs" in kinds:
            self.refuse(n, "a list mixing %s" % " and ".join(sorted(self._kind_words(t) for t in {t for _, t, _ in items})),
                        "Python keeps each item as it is; a Viv list holds strings only, or numbers only.")
        if "rec" in kinds:
            self.refuse(n, "a list of objects", "Viv arrays hold numbers (arrays of records are not in Viv yet); "
                        "keep one array per field.")
        if kinds == {"num", "bool"}:
            # Bools beside numbers: each item keeps its type (Python prints True, not 1)
            codes = [c if t.kind == "num" else ("1" if c == "true" else "0" if c == "false" else "%s(%s)" % (self.use("py_b01"), c))
                     for c, t, _ in items]
            py = "float" if any(t.py in ("float", "mixed") for _, t, _ in items) else "int"
            return "[%s]" % ", ".join(codes), T("arr", 1, py, "list", elts=[t for _, t, _ in items]), 100
        if kinds <= {"num", "bool"}:
            py = "int" if all(t.py in ("int",) for _, t, _ in items) else ("bool" if kinds == {"bool"} else "float")
            codes = []
            for c, t, _ in items:
                codes.append(c if t.kind == "num" else ("1" if c == "true" else "0" if c == "false" else "%s(%s)" % (self.use("py_b01"), c)))
            return "[%s]" % ", ".join(codes), arr(1, py, "list"), 100
        if kinds == {"cnum"} or kinds <= {"cnum", "num"}:
            return "[%s]" % ", ".join(c for c, _, _ in items), T("carr", 1, "complex", "list"), 100
        if kinds == {"arr"} and len({t.rank for _, t, _ in items}) == 1:
            r = items[0][1].rank
            if all(isinstance(e, (ast.List, ast.Tuple)) for e in n.elts):
                if r >= 2:
                    # Viv's array literals are vectors and matrices (the C backend stops on a
                    # deeper one: "expected a number for a matrix element")
                    self.refuse(n, "a rank-%d list literal" % (r + 1),
                                "Viv writes vectors and matrices as literals; allocate np.zeros((...)) and fill it.")
                py = "int" if all(t.py == "int" for _, t, _ in items) else "float"
                cols = self._literal_columns(n, [t for _, t, _ in items])
                if cols is not None:
                    # rows of ints beside floats: each column keeps its items' type (an int
                    # column prints and computes as ints), or is a number of either type
                    return "[%s]" % ", ".join(c for c, _, _ in items), T("arr", r + 1, "float", "list", elts=cols), 100
                return "[%s]" % ", ".join(c for c, _, _ in items), arr(r + 1, py, "list"), 100
            self.refuse(n, "a list of arrays", "Allocate a rank-%d array with np.zeros and fill its rows." % (r + 1))
        if "unknown" in kinds:
            return "[%s]" % ", ".join(c for c, _, _ in items), arr(1, "float", "list"), 100
        self.refuse(n, "a list mixing %s" % ", ".join(sorted(kinds)), "Viv arrays hold numbers of one kind.")

    def _literal_columns(self, n, rows):
        """The column types of a list literal of rows (lists of numbers) that mixes ints and
        floats, or None when every item has one type (a plain int or float matrix). Rows of
        different lengths, or rows that are not lists of numbers, give no columns: the literal
        is refused where a column type is needed."""
        pys = set()
        for e, t in zip(n.elts, rows):
            if t.elts:
                pys |= {"bool" if x.kind == "bool" else x.py for x in t.elts}
            else:
                pys.add(t.py)
        if len(pys & {"int", "bool"}) + (1 if pys & {"float", "mixed"} else 0) < 2:
            return None
        width = {len(e.elts) for e in n.elts}
        if len(width) != 1:
            self.refuse(n, "rows of different lengths mixing ints and floats", "Make the rows one length, or the numbers all floats.")
        w = width.pop()
        cols = [None] * w
        for e, t in zip(n.elts, rows):
            for j in range(w):
                ct = t.elts[j] if t.elts else ({"int": NUM_I, "bool": BOOL}.get(t.py, NUM_F))
                # a column of Bools beside numbers is neither: a value of either type (refused
                # where they differ, as a number that is an int or a float)
                cols[j] = ct if cols[j] is None else (join(cols[j], ct) or NUM_MIXED)
        return cols

    def _e_BinOp(self, n):
        op = n.op
        # [c] * n  ->  n copies
        if isinstance(op, ast.Mult) and isinstance(n.left, ast.List) and len(n.left.elts) == 1:
            v, vt, _ = self.expr(n.left.elts[0])
            cnt, ct, _ = self.expr(n.right)
            if vt.kind == "num" or vt.kind == "bool":
                code = v if vt.kind == "num" else ("1" if v == "true" else "0" if v == "false" else "%s(%s)" % (self.use("py_b01"), v))
                if code in ("0", "0.0"):
                    return "zeros(%s)" % cnt, arr(1, vt.py if vt.kind == "num" else "bool", "list"), 100
                return "%s(%s, %s)" % (self.use("py_full"), cnt, code), arr(1, vt.py if vt.kind == "num" else "bool", "list"), 100
            if vt.kind == "arr" and vt.rank == 1:
                self.refuse(n, "a list of references to one list (`[row] * n`)", "Allocate np.zeros((n, m)).")
        if isinstance(op, ast.Mod) and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str):
            return self._str_percent(n)  # "..." % x: printf-style formatting (strs.py)
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            a, ta, _ = self.expr(n.left)
            b, tb, _ = self.expr(n.right)
            return self._bitop(op, a, ta, b, tb, n), NUM_I, 100
        arith = isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow))
        a, ta, pa = self.expr(n.left, keep=arith)
        b, tb, pb = self.expr(n.right, keep=arith)
        if ta.kind in ints_mod.IE_KINDS or tb.kind in ints_mod.IE_KINDS:
            # an Int operand: Viv's Int operators, or the Ints as numbers (ints.py)
            r = self._ie_binop(n, op, a, ta, pa, b, tb, pb)
            if r is not None:
                return r
            a, ta, pa = self._ie_lower(n.left, (a, ta, pa))
            b, tb, pb = self._ie_lower(n.right, (b, tb, pb))
        r = self._str_binop(n, op, a, ta, pa, b, tb, pb)  # a string or a list of strings (strs.py)
        if r is not None:
            return r
        rt = self._binop_type(op, ta, tb, n)
        if isinstance(op, ast.Div) and (ta.kind in ("cnum", "carr") or tb.kind in ("cnum", "carr")):
            self.inexact(n, "complex division (CPython's and NumPy's formulas are their own)")
        if ta.is_arr() and tb.is_arr() and isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            # two arrays: NumPy broadcasts them (the operator rules, rules/numpy.toml)
            r = RULES.binop(self, n, self._opsym(op), (n.left, (a, ta, pa)), (n.right, (b, tb, pb)), rt)
            if r is not None:
                return r
        guarded = self._scalar_arith(op, n, a, ta, pa, b, tb, pb, rt)
        if guarded is not None:
            return guarded
        if isinstance(op, ast.MatMult):
            self.inexact(n, "@ (BLAS sums in its own order)")
        if isinstance(op, ast.MatMult) or (isinstance(op, ast.Mult) and False):
            if ta.is_arr() and tb.is_arr() and ta.rank == 1 and tb.rank == 1:
                return "sum(%s * %s)" % (self._paren(a, pa, 70), self._paren(b, pb, 71)), NUM_F, 100
            return "%s @ %s" % (self._paren(a, pa, 70), self._paren(b, pb, 71)), rt, 70
        if isinstance(op, ast.Pow):
            if tb.is_arr():
                self.refuse(n, "`**` with an array exponent", "Viv's ^ takes a number exponent; write the loop.")
            if ta.is_arr():
                # through a def whose parameter is an array: Viv types `a ^ y` of an array a def
                # returned as a number
                return "%s(%s, %s)" % (self.use("py_apow"), a, b), rt, 100
            return "%s ^ %s" % (self._paren(a, pa, 91), self._paren(b, pb, 80)), rt, 90
        if isinstance(op, (ast.Mod, ast.FloorDiv)):
            if ta.kind == "arr" and ta.rank == 1 and tb.kind == "num":
                fn = self.use("py_mod" if isinstance(op, ast.Mod) else "py_floordiv")
                isint = "true" if (ta.py == "int" and tb.py == "int") else "false"
                return "%s(%s, %s, %s)" % (fn, a, b, isint), rt, 100
            if ta.is_arr() or tb.is_arr():
                self.refuse(n, "`%s` on a matrix or with an array divisor" % self._opsym(op), "Viv's `%` and `//` take numbers: write the loop.")
        sym = self._opsym(op)
        prec = 70 if isinstance(op, (ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)) else 60
        return "%s %s %s" % (self._paren(a, pa, prec), sym, self._paren(b, pb, prec + 1)), rt, prec

    def _bitop(self, op, a, ta, b, tb, node):
        """A bit operation on Python ints, exactly (py_band, py_shl, py_shr: every operand
        checked to be an integer below 2^53 at run time)."""
        for t, side in ((ta, node.left), (tb, node.right)):
            if t.kind in ("str", "strs"):
                self.refuse(side, "a bit operation on %s" % self._kind_words(t), "Python raises TypeError.")
            if t.kind == "bool":
                self.refuse(side, "a bit operation on a Bool", "Python gives a Bool back for `&`, `|`, `^` of two Bools; use `and`, `or`, `!=`.")
            if t.is_arr():
                self.refuse(side, "a bit operation on an array", "Loop over the entries.")
            if t.kind == "num" and t.py == "float":
                self.refuse(side, "a bit operation on a float", "Python refuses it too (TypeError).")
        if isinstance(op, ast.LShift):
            return "%s(%s, %s)" % (self.use("py_shl"), a, b)
        if isinstance(op, ast.RShift):
            return "%s(%s, %s)" % (self.use("py_shr"), a, b)
        k = {ast.BitAnd: 0, ast.BitOr: 1, ast.BitXor: 2}[type(op)]
        return "%s(%s, %s, %d)" % (self.use("py_band"), a, b, k)

    @staticmethod
    def _small_int(node):
        """A value that is an integer well below 2^52 in size by its form: an int literal below
        2^31, len/numel of an array, an entry of a shape. (A sum or difference of two of them
        cannot reach 2^53.)"""
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return abs(node.value) < 2 ** 31
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return Translator._small_int(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
            return True
        if isinstance(node, ast.Attribute) and node.attr in ("size", "ndim"):
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return True
        return False

    @staticmethod
    def _nonzero_literal(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            node = node.operand
        return isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value != 0 \
            and node.value == node.value

    def _scalar_arith(self, op, n, a, ta, pa, b, tb, pb, rt):
        """Arithmetic whose Python meaning a bare Viv operator would not keep: None when the
        plain operator is exact. Scalars follow Python (int results checked and without -0,
        ZeroDivisionError, floored // and % as CPython computes them, float ** as float_pow);
        arrays follow NumPy (int64 entries checked below 2^53, // and % as npy_divmod)."""
        is_int = lambda t: t.kind == "num" and t.py in ("int", "mixed") or t.kind == "bool"
        scalar = ta.kind in ("num", "bool") and tb.kind in ("num", "bool")
        if ta.kind == "unknown" or tb.kind == "unknown":
            return None
        if scalar and (ta.py == "mixed" or tb.py == "mixed"):
            # a number that is an int on some paths and a float on others: Python computes an int
            # result (exact, no -0.0, int ** int by squaring) or a float one depending on the path.
            # With a float beside it the result is a float either way.
            other = tb if ta.py == "mixed" else ta
            if other.kind == "num" and other.py == "float" and not isinstance(op, ast.Div):
                if isinstance(op, ast.Pow):
                    self.inexact(n, "float ** (Viv's pow is not the platform libm's)")
                    return "%s(%s, %s)" % (self.use("py_fpow"), a, b), NUM_F, 100
                if isinstance(op, (ast.FloorDiv, ast.Mod)):
                    which = "floordiv" if isinstance(op, ast.FloorDiv) else "mod"
                    return "%s(%s, %s)" % (self.use("py_%s_f" % which), a, b), NUM_F, 100
                return None
            if isinstance(op, ast.Div) and (self._nonzero_literal(n.right) or tb.py == "float" or ta.py == "float"):
                if self._nonzero_literal(n.right):
                    return None
                return "%s(%s, %s)" % (self.use("py_div_f"), a, b), NUM_F, 100
            self.refuse(n, "`%s` of a number that is an int on some paths and a float on others" % self._opsym(op),
                        "Python computes it as an int on one path and as a float on the other; convert it with float(...) first.")
        if scalar:
            both_int = is_int(ta) and is_int(tb)
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
                if not both_int:
                    return None
                code = "%s %s %s" % (self._paren(a, pa, 60 if not isinstance(op, ast.Mult) else 70), self._opsym(op),
                                     self._paren(b, pb, 61 if not isinstance(op, ast.Mult) else 71))
                if not isinstance(op, ast.Mult) and self._small_int(n.left) and self._small_int(n.right):
                    return None
                return "%s(%s)" % (self.use("py_int"), code), rt, 100
            if isinstance(op, ast.Div):
                if self._nonzero_literal(n.right):
                    return None
                fn = "py_div_i" if both_int else "py_div_f"
                return "%s(%s, %s)" % (self.use(fn), a, b), rt, 100
            if isinstance(op, (ast.FloorDiv, ast.Mod)):
                which = "floordiv" if isinstance(op, ast.FloorDiv) else "mod"
                fn = "py_%s_%s" % (which, "i" if both_int else "f")
                return "%s(%s, %s)" % (self.use(fn), a, b), rt, 100
            if isinstance(op, ast.Pow):
                if both_int:
                    if isinstance(n.right, ast.UnaryOp) and isinstance(n.right.op, ast.USub):
                        self.inexact(n, "float ** (Viv's pow is not the platform libm's)")
                        return "%s(%s, %s)" % (self.use("py_fpow"), a, b), NUM_F, 100
                    return "%s(%s, %s)" % (self.use("py_ipow"), a, b), NUM_I, 100
                self.inexact(n, "float ** (Viv's pow is not the platform libm's)")
                return "%s(%s, %s)" % (self.use("py_fpow"), a, b), rt, 100
            return None
        # arrays (NumPy)
        if isinstance(op, ast.Pow):
            self.inexact(n, "** on an array (Viv's pow is not the platform libm's)")
        if rt.kind == "arr" and rt.py == "int" and isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
            sym = self._opsym(op)
            if sym == "^":
                if tb.is_arr():
                    self.refuse(n, "`**` with an array exponent", "Viv's ^ takes a number exponent; write the loop.")
                code = "%s(%s, %s)" % (self.use("py_apow"), a, b)
            else:
                code = "%s %s %s" % (self._paren(a, pa, 70), sym, self._paren(b, pb, 71))
            return "%s(%s)" % (self.use("py_int_arr"), code), rt, 100
        return None

    def _binop_type(self, op, ta, tb, node):
        if ta.kind == "rec" or tb.kind == "rec":
            self.refuse(node, "an operator on %s" % self._kind_words(ta if ta.kind == "rec" else tb),
                        "Operators on objects (__add__, ...) are not translated; write a def that takes the objects.")
        if ta.kind == "str" or tb.kind == "str":
            self.refuse(node, "string arithmetic", "Viv strings are not values.")
        if ta.kind == "unknown" or tb.kind == "unknown":
            if ta.is_arr():
                return ta
            if tb.is_arr():
                return tb
            return UNK
        if (ta.is_arr() and ta.container == "list") and (tb.is_arr() and tb.container == "list") and isinstance(op, ast.Add):
            self.refuse(node, "`+` on two lists (concatenation)", "Allocate the joined length with np.zeros and copy both; growable arrays are R17.", gap="R17")
        if ta.is_arr() and ta.container == "list" and tb.kind == "num" and isinstance(op, ast.Mult):
            self.refuse(node, "`*` of a list by a number (repetition)", "Use np.zeros(n) (or [v] * n for one value).")
        if ta.kind == "bool" or tb.kind == "bool":
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                self.refuse(node, "arithmetic on a Bool", "Viv Bools are not numbers: write `1 if c else 0`, or keep a 0/1 number.")
        if (ta.is_arr() and ta.container == "shape") or (tb.is_arr() and tb.container == "shape"):
            self.refuse(node, "arithmetic on a shape (a tuple)", "Python's tuple arithmetic repeats or joins; take the dimensions: n, m = A.shape.")
        if not isinstance(op, ast.MatMult) and (ta.is_arr() or tb.is_arr()) and \
                all(not t.is_arr() or t.container == "list" for t in (ta, tb)) and \
                not ((ta.container == "list" and tb.container == "list" and isinstance(op, ast.Add)) or
                     (ta.is_arr() and tb.kind == "num" and isinstance(op, ast.Mult))):
            self.refuse(node, "arithmetic on a Python list", "Python lists do not compute elementwise (a TypeError, or repetition); make it a NumPy array: np.array(...).")
        if (ta.kind == "arr" and tb.kind == "cnum") or (ta.kind == "cnum" and tb.kind == "arr"):
            self.refuse(node, "arithmetic between a real array and a complex number", "Viv does not mix them; build a complex array (np.zeros(n, dtype=complex)) first.")
        if isinstance(op, (ast.Add, ast.Sub)) and ((ta.kind == "carr" and tb.kind in ("num", "arr")) or
                                                   (tb.kind == "carr" and ta.kind in ("num", "arr"))):
            self.refuse(node, "adding a real value to a complex array", "Viv adds complex arrays to complex values only; copy the real one into np.zeros(n, dtype=complex) first.")
        if isinstance(op, ast.MatMult):
            if not (ta.is_arr() and tb.is_arr()):
                self.refuse(node, "`@` on a number", "Use `*`.")
            if ta.rank == 1 and tb.rank == 1:
                return NUM_F
            # NumPy's matmul, which Viv's @ follows: a vector operand's axis is
            # dropped, and stacks of matrices broadcast their leading axes
            if ta.rank == 1:
                return arr(tb.rank - 1)
            if tb.rank == 1:
                return arr(ta.rank - 1)
            return arr(max(ta.rank, tb.rank))
        if ta.is_arr() or tb.is_arr():
            # two arrays broadcast (the operator rules, rules/numpy.toml): the larger rank
            base = ta if ta.is_arr() and (not tb.is_arr() or ta.rank >= tb.rank) else tb
            kind = "carr" if (ta.kind in ("carr", "cnum") or tb.kind in ("carr", "cnum")) else "arr"
            py = "float"
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)):
                pa = ta.py if ta.is_arr() else ta.py
                pb = tb.py if tb.is_arr() else tb.py
                if pa == "bool" and pb == "bool":
                    # NumPy: + and * of Bool arrays are logical or and and, - raises TypeError
                    self.refuse(node, "arithmetic on two Bool arrays",
                                "NumPy's + and * of Bools are `or` and `and`, and - raises; convert one with np.where(m, 1, 0).")
                # a Bool array is 0/1 in NumPy's arithmetic, and an int with an int
                py = "int" if pa in ("int", "bool") and pb in ("int", "bool") else "float"
            return T(kind, base.rank, py if kind == "arr" else "complex", "nd")
        if ta.kind == "cnum" or tb.kind == "cnum":
            return CNUM
        if isinstance(op, ast.Div):
            return NUM_F
        if isinstance(op, ast.Pow):
            return NUM_F if not (ta.py == "int" and tb.py == "int") else NUM_I
        return num_of(ta, tb)

    def _e_UnaryOp(self, n):
        a, t, p = self.expr(n.operand, keep=isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)))
        if t.kind in ints_mod.IE_KINDS:
            r = self._ie_unary(n, a, t, p)  # -n of an Int (ints.py)
            if r is not None:
                return r
        if t.kind == "rec" and not isinstance(n.op, ast.Not):
            self.refuse(n, "an operator on %s" % self._kind_words(t), "Operators on objects are not translated.")
        if isinstance(n.op, (ast.USub, ast.UAdd, ast.Invert)) and t.kind in ("str", "strs"):
            self.refuse(n, "a unary operator on %s" % self._kind_words(t), "Python raises TypeError.")
        if isinstance(n.op, ast.USub):
            if t.kind == "bool":
                self.refuse(n, "negating a Bool", "Viv Bools are not numbers.")
            if t.is_arr() and t.container in ("list", "shape"):
                self.refuse(n, "negating a %s" % ("list" if t.container == "list" else "shape (a tuple)"),
                            "Python raises TypeError; make it a NumPy array first.")
            if t.kind == "arr" and t.py == "bool":
                self.refuse(n, "negating a Bool array", "NumPy raises TypeError; use ~m or np.where(m, 1, 0).")
            if t.kind == "carr":
                self.refuse(n, "negating a complex array",
                            "Viv's compiled engines lose the sign of a zero imaginary part; write 0 - X if the sign of zero does not matter.")
            if t.kind == "num" and t.py == "mixed":
                self.refuse(n, "negating a number that is an int on some paths and a float on others",
                            "Python's -0.0 is a float's; convert it with float(...) first.")
            if t.kind == "num" and t.py in ("int", "mixed") and not isinstance(n.operand, ast.Constant):
                # -0 is 0 for an int
                return "0 - %s" % self._paren(a, p, 61), t, 60
            if t.kind == "arr" and t.py == "int":
                return "0 - %s" % self._paren(a, p, 61), t, 60
            if t.kind == "arr":
                # -1.0 * a is -a exactly (signs of zero included); Viv's `-a` of an array that
                # a def returned is typed as a number
                return "-1.0 * %s" % self._paren(a, p, 71), t, 70
            return "-%s" % self._paren(a, p, 80), t, 80
        if isinstance(n.op, ast.UAdd):
            return a, t, p
        if isinstance(n.op, ast.Not):
            c = self._truth(n.operand, a, t, p)
            cp = 100 if c == a and p >= 80 else 50
            return "not %s" % self._paren(c, cp, 80), BOOL, 80
        if isinstance(n.op, ast.Invert):
            if t.kind == "bool" or t.is_arr() or (t.kind == "num" and t.py == "float"):
                self.refuse(n, "`~` of %s" % self._kind_words(t), "Python's `~` inverts an int: -x - 1.")
            # ~x = -x - 1 on Python's ints: exact on a double below 2^53
            return "%s(0 - %s(%s) - 1)" % (self.use("py_int"), self.use("py_bits"), a), NUM_I, 100
        self.refuse(n, "this unary operator", "Not in the subset.")

    def _e_BoolOp(self, n):
        as_value = id(n) not in getattr(self, "cond_ok", set())
        codes = []
        parts = []  # (statements it needs first, its truth as a Viv condition)
        for i, v in enumerate(n.values):
            saved = self.pre
            self.pre = []
            c, t, p = self.expr(v)
            got = self.pre
            self.pre = saved
            if as_value and t.kind not in ("bool", "unknown"):
                self.refuse(n, "`and`/`or` of %s used as a value" % self._kind_words(t),
                            "Python gives back one of the operands, not True or False; write `a if a else b`, or compare first.")
            if t.kind not in ("bool", "unknown"):
                c = self._truth(v, c, t, p)
                p = 50
            if isinstance(v, ast.BoolOp):
                p = 0
            parts.append((got, c))
            codes.append(self._paren(c, p, 50))
        word = " and " if isinstance(n.op, ast.And) else " or "
        if not any(got for got, _ in parts[1:]):
            self.pre.extend(parts[0][0])
            return word.join(codes), BOOL, 40
        # a later operand needs statements first: they run only when Python evaluates it,
        # under an `if` on what came before
        tmp = self.new_tmp("b")
        self.pre.extend(parts[0][0])
        self.pre.append("%s = %s" % (tmp, parts[0][1]))
        ind = ""
        for got, c in parts[1:]:
            self.pre.append(ind + ("if %s {" % tmp if isinstance(n.op, ast.And) else "if not %s {" % tmp))
            ind += "    "
            self.pre.extend(ind + x for x in got)
            self.pre.append(ind + "%s = %s" % (tmp, c))
        while ind:
            ind = ind[:-4]
            self.pre.append(ind + "}")
        return tmp, BOOL, 100

    def _e_Compare(self, n):
        ops = n.ops
        parts = []
        plain = not any(isinstance(op, (ast.In, ast.NotIn)) for op in ops)
        lc, lt, lp = self.expr(n.left, keep=plain)  # an Int, a member, an optional (ints.py)
        types = [lt]
        for op, right in zip(ops, n.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                oc = self._obj_in(n, op, right, lc, lt)  # "k" in d (classes.py)
                if oc is not None:
                    if len(ops) > 1:
                        self.refuse(n, "`in` inside a chained comparison", "Split it with `and`.")
                    parts.append(oc)
                    continue
                if isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                    eqs = []
                    for e in right.elts:
                        c, t, p = self.expr(e)
                        if t.kind == "rec" or lt.kind == "rec":
                            self.refuse(n, "`in` with %s" % self._kind_words(t if t.kind == "rec" else lt),
                                        "Python compares objects with ==; compare the fields.")
                        if (t.kind in ("str", "strs") or lt.kind in ("str", "strs")) and t.kind != lt.kind \
                                and "unknown" not in (t.kind, lt.kind):
                            self.refuse(n, "`in` of %s among %s" % (self._kind_words(lt), self._kind_words(t)),
                                        "Python compares with == (False between a string and a number); compare values of one kind.")
                        eqs.append("%s == %s" % (self._paren(lc, lp, 51), self._paren(c, p, 51)))
                    s = " or ".join(eqs) if eqs else "false"
                    if isinstance(op, ast.NotIn):
                        s = "not (%s)" % s
                    elif len(eqs) > 1:
                        s = "(%s)" % s
                    parts.append(s)
                    if len(ops) > 1:
                        self.refuse(n, "`in` inside a chained comparison", "Split it with `and`.")
                    continue
                if len(ops) > 1:
                    self.refuse(n, "`in` inside a chained comparison", "Split it with `and`.")
                sc = self._str_in(n, op, lc, lt, lp, right)  # a substring, a string in a list of strings (strs.py)
                if sc is not None:
                    return sc, BOOL, 40
                # x in lst, x in a (funcs.py)
                return self._funcs_in(n, op, lc, lt, right), BOOL, 40
            if isinstance(op, (ast.Is, ast.IsNot)) and lt.kind not in ints_mod.IE_KINDS:
                self.refuse(n, "`is` / `is not`", "Compare values with `==`; `None` is not a Viv value.")
            rc, rt, rp = self.expr(right, keep=plain)
            if lt.kind in ints_mod.IE_KINDS or rt.kind in ints_mod.IE_KINDS:
                # ints.py: Int comparisons, == of members; or the Ints as numbers
                lc, lt, lp, rc, rt, rp, part = self._ie_compare(n, op, lc, lt, lp, rc, rt, rp)
                if part is not None:
                    types[-1] = lt
                    types.append(rt)
                    parts.append(part)
                    lc, lt, lp = rc, rt, rp
                    continue
                if isinstance(op, (ast.Is, ast.IsNot)):
                    self.refuse(n, "`is` / `is not`", "Compare values with `==`; `None` is not a Viv value.")
            types.append(rt)
            if lt.kind == "rec" or rt.kind == "rec":
                self.refuse(n, "comparing %s" % self._kind_words(lt if lt.kind == "rec" else rt),
                            "A dataclass compares its fields, a plain class the objects' identity; compare the fields you mean.")
            sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[type(op)]
            sc = self._str_compare(n, sym, lc, lt, lp, rc, rt, rp)  # strings, lists of strings (strs.py)
            if sc is not None:
                parts.append(sc)
                lc, lt, lp = rc, rt, rp
                continue
            if lt.is_arr() and rt.is_arr() and isinstance(op, (ast.Eq, ast.NotEq)) and lt.container == "list" and rt.container == "list":
                if len(ops) > 1:
                    self.refuse(n, "`==` of lists in a chained comparison", "Split it with `and`.")
                return self._funcs_listeq(n, sym, lc, lt, rc, rt), BOOL, 40  # one Bool, as Python (funcs.py)
            if (lt.is_arr() and lt.container in ("list", "shape") or rt.is_arr() and rt.container in ("list", "shape")) \
                    and not (lt.is_arr() and lt.container == "nd" or rt.is_arr() and rt.container == "nd"):
                self.refuse(n, "`%s` of a Python list or a tuple" % sym, "Python compares lists and tuples as a whole (in order, or "
                            "raises TypeError against a number); make it a NumPy array to compare entry by entry.")
            if lt.kind == "bool" and rt.kind == "num" or lt.kind == "num" and rt.kind == "bool":
                self.refuse(n, "comparing a Bool with a number", "Viv Bools are not numbers; compare Bools with Bools.")
            if lt.kind == "carr" or rt.kind == "carr" or (lt.kind == "cnum" and rt.is_arr()) or (rt.kind == "cnum" and lt.is_arr()):
                self.refuse(n, "comparing a complex array", "NumPy compares complex numbers entry by entry (`<` by the real part, then the "
                            "imaginary part), and Viv does not; compare np.real(z) and np.imag(z).")
            if lt.is_arr() and rt.is_arr() and len(ops) == 1:
                # two arrays: NumPy broadcasts them (the operator rules, rules/numpy.toml)
                r = RULES.binop(self, n, sym, (n.left, (lc, lt, lp)), (right, (rc, rt, rp)),
                                arr(max(lt.rank, rt.rank), "bool"))
                if r is not None:
                    return r
            parts.append("%s %s %s" % (self._paren(lc, lp, 51), sym, self._paren(rc, rp, 51)))
            lc, lt, lp = rc, rt, rp
        if any(t == UNK_C for t in types):
            # an operand of unknown type because of a refusal: the comparison may be an array's
            return " and ".join(parts), UNK_C, 40
        # a comparison with an array on either side is an array of 0/1 in NumPy and Viv
        arrs = [t for t in types if t.is_arr()]
        if arrs:
            if len(parts) > 1:
                self.refuse(n, "a chained comparison of arrays", "Combine two masks with `*`.")
            return parts[0], arr(max(t.rank for t in arrs), "bool"), 50
        if len(parts) == 1:
            return parts[0], BOOL, 50
        return " and ".join("(%s)" % p if " or " in p or " and " in p else p for p in parts), BOOL, 40

    def _e_IfExp(self, n):
        c = self._cond(n.test)
        saved = self.pre
        self.pre = []
        a, ta, _ = self.expr(n.body)
        pre_a = self.pre
        self.pre = []
        b, tb, _ = self.expr(n.orelse)
        pre_b = self.pre
        self.pre = saved
        t = join(ta, tb)
        if t is None:
            self.refuse(n, "a conditional expression whose branches have different types", "Give both branches one type.")
        if not pre_a and not pre_b:
            # Viv's own `a if c else b`: only the branch chosen is computed, as in
            # Python; it binds loosest of all, so it goes in parentheses
            return "(%s if %s else %s)" % (a, c, b), t, 100
        tmp = self.new_tmp("c")
        ind = "    "
        self.pre.append("if %s {" % c)
        self.pre.extend(ind + x for x in pre_a)
        self.pre.append("%s%s = %s" % (ind, tmp, a))
        self.pre.append("}")
        self.pre.append("else {")
        self.pre.extend(ind + x for x in pre_b)
        self.pre.append("%s%s = %s" % (ind, tmp, b))
        self.pre.append("}")
        return tmp, t, 100

    def _e_Attribute(self, n):
        r = self._ie_attr(n)  # C.A of an enum class (ints.py)
        if r is not None:
            return r
        # module constants (np.pi, ...) and np.random: rules/numpy.toml
        full = self._call_name(n)
        if full:
            r = RULES.constant(self, n, full)
            if r is not None:
                return r
        if full and (full.startswith("np.") or full.startswith("math.")):
            return full, T("func", name=full), 100
        base, bt, bp = self.expr(n.value, keep=True)
        if bt.kind == "enum":
            return self._ie_member_attr(n, base, bt, bp)  # m.name (ints.py)
        base, bt, bp = self._ie_lower(n.value, (base, bt, bp))
        a = n.attr
        r = self._obj_attr(n, base, bt, bp)  # p.x, p.area, d["k"], C.K (classes.py)
        if r is not None:
            return r
        if bt.kind == "rng":
            self.refuse(n, "the attribute `.%s` of a Generator" % a, "Only a Generator's draws are translated (random, "
                        "integers, uniform, standard_normal, normal, choice, shuffle, permutation).")
        if bt.is_arr() and bt.container in ("list", "shape"):
            self.refuse(n, "the attribute `.%s` of a %s" % (a, "Python list" if bt.container == "list" else "shape (a tuple)"),
                        "Python raises AttributeError; make it a NumPy array: np.array(...).")
        # a.T, a.shape, a.size, a.ndim, a.real, a.imag: rules/numpy.toml
        r = RULES.attribute(self, n, (base, bt, bp))
        if r is not None:
            return r
        if bt.kind == "func":
            return "%s.%s" % (base, a), T("func", name="%s.%s" % (bt.name, a)), 100
        if bt == UNK_C:
            self.refuse(n, "the attribute `.%s` of a value of unknown type (it comes from a construct refused elsewhere)" % a,
                        "Fix the refusal that produced it.")
        self.refuse(n, "the attribute `.%s`" % a, "Viv has no objects; use the NumPy function form, e.g. np.sum(a).")

    @staticmethod
    def _levels(n):
        """The subscripts of a chain `A[i][1:3]`, innermost first: [(node, items)]."""
        out = []
        node = n
        while isinstance(node, ast.Subscript):
            sl = node.slice
            out.insert(0, (node, sl.elts if isinstance(sl, ast.Tuple) else [sl]))
            node = node.value
        return out

    def _has_range(self, n):
        return any(isinstance(i, ast.Slice) for _, items in self._levels(n) for i in items)

    def _is_newaxis(self, i):
        """`None` or np.newaxis as a subscript item."""
        return isinstance(i, ast.Constant) and i.value is None or \
            isinstance(i, ast.Attribute) and self._call_name(i) == "np.newaxis"

    def _plainly_array(self, i):
        """A subscript item that is an array by its form: a name of an array, a comparison, a
        list literal, or ~, &, | of them (a mask)."""
        if isinstance(i, ast.Name):
            t = self.var_type(i.id)
            return t is not None and t.is_arr()
        if isinstance(i, (ast.Compare, ast.List)):
            return True
        if isinstance(i, ast.UnaryOp) and isinstance(i.op, ast.Invert):
            return self._plainly_array(i.operand)
        if isinstance(i, ast.BinOp) and isinstance(i.op, (ast.BitAnd, ast.BitOr)):
            return self._plainly_array(i.left) or self._plainly_array(i.right)
        return False

    def _e_Subscript(self, n):
        r = self._obj_subscript(n)  # p[0] of a named tuple, d[1] of a dict (classes.py)
        if r is not None:
            return r
        r = self._str_subscript(n)  # s[i], s[a:b], lst[i] of a string or a list of strings (strs.py)
        if r is not None:
            return r
        # np.nonzero(a)[k] and np.where(m)[k]: an item of a call's tuple (rules/numpy.toml)
        if isinstance(n.value, ast.Call) and not isinstance(n.slice, (ast.Slice, ast.Tuple)):
            name = self._call_name(n.value.func)
            if name and name.startswith("np."):
                r = RULES.item(self, n, name)
                if r is not None:
                    return r
        levels = self._levels(n)
        if any(self._is_newaxis(i) for _, items in levels for i in items):
            return self._newaxis(n)
        r = self._fancy(n)
        if r is not None:
            return r
        if self._has_range(n):
            return self._slice(n)
        # A.shape[k]
        if isinstance(n.value, ast.Attribute) and n.value.attr == "shape":
            base, bt, _ = self.expr(n.value.value)
            k, kt, _ = self.expr(n.slice)
            if k == "0" or k == "-1" and bt.rank == 1:
                return "len(%s)" % base, NUM_I, 100
            if isinstance(n.slice, ast.UnaryOp) and isinstance(n.slice.operand, ast.Constant):
                k = str(bt.rank - n.slice.operand.value)
            return "shape(%s)[%s]" % (base, k), NUM_I, 100
        idxs = []
        node = n
        while isinstance(node, ast.Subscript):
            sl = node.slice
            items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            idxs = items + idxs
            node = node.value
        base, bt, bp = self.expr(node)
        if bt.kind == "unknown":
            codes = [self._index(i, node, k, bt) for k, i in enumerate(idxs)]
            return "%s[%s]" % (self._paren(base, bp, 100), ", ".join(codes)), bt, 100
        if bt.kind in ("str", "strs"):
            return self._str_chain(n, (base, bt, bp))  # a string or a list of strings computed in place (strs.py)
        if not bt.is_arr() and bt.container != "shape":
            self.refuse(n, "indexing %s" % self._kind_words(bt), "Index an array.")
        codes = [self._index(i, node, k, bt) for k, i in enumerate(idxs)]
        if len(codes) > bt.rank:
            self.refuse(n, "%d indices into a rank-%d array" % (len(codes), bt.rank), "Use one index per dimension.")
        if not isinstance(node, ast.Name):
            b = self._paren(base, bp, 100)
        else:
            b = base
        if len(codes) == bt.rank:
            et = CNUM if bt.kind == "carr" else (NUM_I if bt.py in ("int", "bool") and bt.container != "nd" else (NUM_I if bt.py == "int" else NUM_F))
            if bt.elts:
                # a tuple's entries, or a list of ints and floats: each keeps its own type (funcs.py)
                et = self._elts_item(bt, idxs[-1])
                if et.kind == "bool":
                    # a Bool item is stored as 1 or 0: read it back as a Bool
                    return "%s[%s] != 0" % (b, ", ".join(codes)), BOOL, 50
            if bt.py == "bool":
                et = BOOL if bt.container == "list" else NUM_I
                if et.kind == "bool":
                    # a list of True/False is stored as 1/0: read it back as a Bool
                    return "%s[%s] != 0" % (b, ", ".join(codes)), BOOL, 50
            return "%s[%s]" % (b, ", ".join(codes)), et, 100
        if len(codes) == 1:
            if re.fullmatch(r"-\d+", codes[0]):
                # slab counts from 0; a slice counts a negative index from the end
                return "%s[%s, :]" % (b, codes[0]), row_type(bt), 100
            return "slab(%s, %s)" % (b, codes[0]), row_type(bt), 100
        self.refuse(n, "a partial index (%d of %d)" % (len(codes), bt.rank), "Index every dimension, or loop.", gap="R3")

    # NumPy functions whose result has the shape of their first argument
    SAME_SHAPE_FUNCS = {"np.sqrt", "np.exp", "np.log", "np.sin", "np.cos", "np.tan", "np.tanh", "np.sinh", "np.cosh",
                        "np.arcsin", "np.arccos", "np.arctan", "np.asin", "np.acos", "np.atan", "np.floor", "np.ceil",
                        "np.fabs", "np.abs", "np.absolute", "np.sign", "np.square", "np.roll", "np.copy", "np.zeros_like",
                        "np.ones_like", "np.empty_like", "np.full_like", "np.real", "np.imag", "np.conj", "np.conjugate"}

    def _static_scalar(self, n):
        """A number by its form: a literal, a name typed as a number, np.pi, arithmetic of them."""
        if isinstance(n, ast.Constant):
            return type(n.value) in (int, float)
        if isinstance(n, ast.Name):
            t = self.var_type(n.id)
            return t is not None and t.kind == "num"
        if isinstance(n, ast.Attribute):
            return self._call_name(n) in ("np.pi", "math.pi", "np.e", "math.e", "math.tau")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return self._static_scalar(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return self._static_scalar(n.left) and self._static_scalar(n.right)
        return False

    def _shape_key(self, n):
        """Which array's shape n's value has, by its form, or None: a name of a NumPy array; a
        range of one with literal bounds (`u[1:-1]` and `u[2:]` both have len(u) - 2 entries,
        so ranges that trim as much agree); an operation of such values of one key, or of one
        with a number; a function that keeps its argument's shape (np.roll, np.sin, ...). An
        array cannot change shape in place, and a name cannot be rebound inside an expression,
        so two operands with one key have one shape when the operation runs."""
        if isinstance(n, ast.Name):
            t = self.var_type(n.id)
            return ("name", n.id) if t is not None and t.is_arr() and t.container == "nd" else None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return self._shape_key(n.operand)
        if isinstance(n, (ast.BinOp, ast.Compare)):
            if isinstance(n, ast.Compare):
                if len(n.ops) != 1:
                    return None
                left, right = n.left, n.comparators[0]
            else:
                if not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    return None
                left, right = n.left, n.right
            kl, kr = self._shape_key(left), self._shape_key(right)
            if kl is not None and kr is not None:
                return kl if kl == kr else None
            if kl is not None and self._static_scalar(right):
                return kl
            if kr is not None and self._static_scalar(left):
                return kr
            return None
        if isinstance(n, ast.Call) and n.args and not any(isinstance(a, ast.Starred) for a in n.args):
            if self._call_name(n.func) in self.SAME_SHAPE_FUNCS:
                return self._shape_key(n.args[0])
            if isinstance(n.func, ast.Attribute) and n.func.attr == "copy" and not n.args:
                return self._shape_key(n.func.value)
            return None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "copy" and not n.args \
                and not self._is_module_path(n.func):
            return self._shape_key(n.func.value)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            t = self.var_type(n.value.id)
            if t is None or not t.is_arr() or t.container != "nd":
                return None
            items = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            if len(items) > t.rank:
                return None
            axes = []

            def lit(x):
                if isinstance(x, ast.Constant) and type(x.value) is int:
                    return x.value
                if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub) and isinstance(x.operand, ast.Constant) \
                        and type(x.operand.value) is int:
                    return -x.operand.value
                return None
            for it in items:
                if isinstance(it, ast.Slice):
                    if it.step is not None and lit(it.step) != 1:
                        return None
                    lo = 0 if it.lower is None else lit(it.lower)
                    hi = 0 if it.upper is None else lit(it.upper)
                    if lo is None or hi is None:
                        return None
                    if lo >= 0 and hi <= 0:
                        axes.append(("trim", lo - hi))  # max(0, n - lo + hi) entries
                    elif lo >= 0 and hi > 0:
                        axes.append(("fix", lo, hi))
                    else:
                        return None
                elif isinstance(it, ast.Constant) and it.value is Ellipsis or self._is_newaxis(it):
                    return None
                elif self._plainly_array(it):
                    return None
                elif isinstance(it, (ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp)) and self._static_scalar(it):
                    axes.append(("index",))
                else:
                    return None
            axes += [("trim", 0)] * (t.rank - len(items))
            return ("sub", n.value.id, tuple(axes))
        return None

    def _axis_len(self, base, axis):
        return "len(%s)" % base if axis == 0 else "shape(%s)[%d]" % (base, axis)

    def _newaxis(self, n):
        """`c[:, None]`, `x[None, :]`, `v[np.newaxis]`: the subscript without its `None` items,
        then an axis of length 1 at each place a `None` stood (the newaxis rule). A view in
        NumPy: used in place it is checked as a slice is, bound to a name by the assignment."""
        levels = self._levels(n)
        if len(levels) > 1:
            self.refuse(n, "`None` (np.newaxis) in a chain of subscripts", "Write one subscript: `A[i, :, None]`.")
        items = levels[0][1]
        if any(isinstance(i, ast.Constant) and i.value is Ellipsis for i in items):
            self.refuse(n, "`...` in a subscript", "Index each axis with a number or a range.")
        rest = [i for i in items if not self._is_newaxis(i)]
        axes = []
        pos = 0
        for i in items:
            if self._is_newaxis(i):
                axes.append(pos)
                pos += 1
            elif isinstance(i, ast.Slice):
                pos += 1
        if all(isinstance(i, ast.Slice) and i.lower is None and i.upper is None and i.step is None for i in rest):
            inner = n.value  # `c[:, None]`: whole ranges take every entry of c
        else:
            sl = rest[0] if len(rest) == 1 else ast.copy_location(ast.Tuple(elts=rest, ctx=ast.Load()), n.slice)
            inner = ast.copy_location(ast.Subscript(value=n.value, slice=sl, ctx=ast.Load()), n)
        code, t, p = self.expr(inner)
        if t.kind == "unknown":
            self.refuse(n, "`None` (np.newaxis) in a subscript of a value of unknown type", "Give the array a known type first.")
        r = RULES.newaxis(self, n, (inner, (code, t, p)), axes)
        if r is None:
            self.refuse(n, "`None` (np.newaxis) in a subscript", "Not in the numeric subset.")
        if t.is_arr():
            self._slice_in_place(n)
        return r

    def _fancy(self, n, store=None):
        """A subscript with a Bool array (a mask) or an integer array among its items: `a[m]`,
        `a[idx]`, `A[:, idx]`, `A[I, J]`, `A[i, idx]` (the rules index-array and index-arrays;
        with `store`, the value stored through it, (node, (code, T, prec)): index-array-store
        and index-arrays-store). NumPy gives a copy, so no view check applies to a read; a
        store writes into the array. None when no item is an array: the items are translated
        once (the receiver first, as Python evaluates it) and kept for the path that takes the
        subscript then."""
        levels = self._levels(n)
        items = levels[-1][1]
        recv = n.value
        if isinstance(recv, ast.Attribute) and recv.attr == "shape":
            return None
        if len(levels) > 1 and not any(self._plainly_array(i) for i in items):
            return None
        if store is not None and not isinstance(recv, ast.Name) and not any(self._plainly_array(i) for i in items):
            return None
        rv = self.expr(recv) if store is None else None
        if store is not None:
            if not isinstance(recv, ast.Name):
                self.refuse(n, "a store through a mask or integer indices into a computed array", "Store into a named array.")
            rv = (self.vname(recv.id), self.var_type(recv.id) or UNK, 100)
        vals = []
        for it in items:
            if isinstance(it, ast.Slice) or self._is_newaxis(it) or isinstance(it, ast.Constant) and it.value is Ellipsis:
                vals.append(None)
                continue
            vals.append(self.expr(it))
        arrays = [k for k, v in enumerate(vals) if v is not None and v[1].is_arr()]
        if not arrays:
            if len(levels) == 1:
                if store is None:
                    self._xmemo[id(recv)] = rv
                for it, v in zip(items, vals):
                    if v is not None:
                        self._xmemo[id(it)] = v
            return None
        self.fancy_nodes.add(id(n))
        bt = rv[1]
        if store is not None and recv.id in self.slab_names and bt.is_arr():
            self.refuse(n, "a store through `%s`, which is a view of another array" % recv.id,
                        "NumPy writes through the view into the original array; Viv's is a copy. Write into the original array.")
        if not bt.is_arr():
            if bt.kind == "unknown":
                self.refuse(n, "a mask or integer indices into a value of unknown type", "Give the array a known type first.")
            self.refuse(n, "indexing %s with an array" % self._kind_words(bt), "Index an array.")
        for it, v in zip(items, vals):
            if self._is_newaxis(it) or isinstance(it, ast.Constant) and it.value is Ellipsis:
                self.refuse(it, "`None` or `...` beside a mask or integer indices", "Index with the array alone, then add the axis.")
            if isinstance(it, ast.Slice) and (it.lower is not None or it.upper is not None or it.step is not None):
                self.refuse(it, "a range beside a mask or integer indices", "Take the range first (`b = a[1:4]`), then index b with the array.")
            if v is not None and v[1].kind == "bool":
                self.refuse(it, "a Bool index", "Index with a number or an array.")
            if v is not None and not v[1].is_arr() and v[1].kind not in ("num", "unknown"):
                self.refuse(it, "an index that is %s" % self._kind_words(v[1]), "Index with an int or an integer array.")
        if len(items) > bt.rank:
            self.refuse(n, "%d indices into a rank-%d array" % (len(items), bt.rank), "Use one index per axis.")
        slices = [k for k, it in enumerate(items) if isinstance(it, ast.Slice)]
        scalars = [k for k, v in enumerate(vals) if v is not None and not v[1].is_arr()]
        if slices or len(arrays) == 1 and not scalars:
            if len(arrays) > 1 or scalars:
                self.refuse(n, "an array index beside another index and a range", "Index one axis with the array and the others with `:`, or every axis with an int or an integer array.")
            key = [":" if isinstance(it, ast.Slice) else "#" for it in items]
            while key and key[-1] == ":":
                key.pop()
            named = [(items[k], vals[k]) for k in arrays]
        else:
            key = ["#"] * len(items)
            named = [(it, v) for it, v in zip(items, vals)]
        key = "[%s]" % ",".join(key)
        if len(levels) > 1:
            recv_pair = (recv, rv)
        else:
            recv_pair = (recv, rv)
        r = RULES.index(self, n, key, recv_pair, named, store=store)
        if r is None:
            self.refuse(n, "this subscript with an array among its items", "Index one axis with a mask or an integer array and the others with `:`, "
                        "or every axis with an int or an integer array.")
        return r

    def _subscript_items(self, n, node, idxs, base, bt, bp):
        """The Viv index list of a subscript with a range: each range through the slice rule
        (rules/numpy.toml), each plain index as NumPy reads it (a negative one from the end)."""
        codes = []
        for k, it in enumerate(idxs):
            ncode = self._axis_len(self._paren(base, bp, 100), k)
            if isinstance(it, ast.Slice):
                c = RULES.slice_item(self, it, node, (base, bt, bp), k, ncode)
                if c is None:
                    self.refuse(it, "this slice", "Not in the numeric subset.")
                codes.append(c)
                continue
            if isinstance(it, ast.Constant) and it.value is Ellipsis or isinstance(it, ast.Constant) and it.value is None:
                self.refuse(it, "`...` or `None` (np.newaxis) in a subscript", "Index each axis with a number or a range.")
            if isinstance(it, ast.Attribute) and self._call_name(it) == "np.newaxis":
                self.refuse(it, "np.newaxis in a subscript", "Viv arrays do not grow axes by indexing; reshape instead.")
            c, t, p = self.expr(it)
            if t.kind == "bool":
                self.refuse(it, "a Bool index", "Index with a number.")
            if t.is_arr():
                self.refuse(it, "fancy (array) indexing", "Write the loop, or take(A, idx) for a flat gather.")
            if t.kind == "num" and t.py != "int":
                self.refuse(it, "an index that is %s" % ("a float" if t.py == "float" else "an int or a float"),
                            "NumPy raises IndexError for a float index; convert with int(...).")
            if t.kind not in ("num", "unknown"):
                self.refuse(it, "an index that is %s" % self._kind_words(t), "Index with an int.")
            if isinstance(it, ast.Constant):
                codes.append(c)
            elif isinstance(it, ast.UnaryOp) and isinstance(it.op, ast.USub) and isinstance(it.operand, ast.Constant):
                codes.append("-%s" % it.operand.value)  # from the end, and past the start an error, in Viv as in NumPy
            else:
                codes.append("%s(%s, %s)" % (self.use("py_ix"), c, ncode))
        return codes

    def _slice(self, n):
        """A subscript with a range, `a[lo:hi]`, `A[:, j]`, `A[i][1:3]`: a Viv slice (a copy).
        Whether the copy can differ from NumPy's view is views.py's question, asked here for
        a slice used in place (an argument, an operand) and by the assignment for a slice
        bound to a name."""
        r = self._funcs_rslice(n)  # a[::-1], a[hi:lo:-k] of a vector: a reversed copy (funcs.py)
        if r is not None:
            return r
        levels = self._levels(n)
        for lv, items in levels[:-1]:
            if any(isinstance(i, ast.Slice) for i in items):
                self.refuse(lv, "a subscript of a slice (`a[1:3][0]`)",
                            "Write it as one subscript, `a[1:3, 0]` or `a[1]`, with one index or range per axis.")
        node = levels[0][0].value
        idxs = [i for _, items in levels for i in items]
        if isinstance(node, ast.Attribute) and node.attr == "shape":
            self.refuse(n, "a slice of a shape (a tuple)", "Take the dimensions: `n, m = A.shape`.")
        base, bt, bp = self.expr(node)
        if bt.kind in ("str", "strs"):
            return self._str_chain(n, (base, bt, bp))  # a slice of text computed in place (strs.py)
        if bt.container == "shape":
            self.refuse(n, "a slice of a shape (a tuple)", "Take the dimensions: `n, m = A.shape`.")
        if not bt.is_arr() and bt.kind != "unknown":
            self.refuse(n, "a slice of %s" % self._kind_words(bt), "Slice an array or a list.")
        if bt.is_arr() and len(idxs) > bt.rank:
            self.refuse(n, "%d indices into a rank-%d array" % (len(idxs), bt.rank), "Use one index per dimension.")
        if not isinstance(node, ast.Name):
            # the base's code is written once per axis length: it may not call a def, or draw
            if rule_engine.DRAWS.search(base):
                self.refuse(node, "a slice of an expression that draws random numbers",
                            "The translation evaluates the array once per bound and Python once; assign it to a name first.")
            for f in self._effect_names():
                if re.search(r"(?<![\w.])%s\(" % re.escape(viv_name(f)), base):
                    self.refuse(node, "a slice of an expression that calls `%s`" % f,
                                "The translation evaluates the array once per bound and Python once; assign it to a name first.")
        codes = self._subscript_items(n, node, idxs, base, bt, bp)
        if bt.is_arr():
            rank = bt.rank - sum(1 for i in idxs if not isinstance(i, ast.Slice))
            t = T(bt.kind, rank, bt.py, bt.container)
        else:
            t = UNK
        if bt.is_arr() and not (bt.container == "list" and bt.rank == 1):
            self._slice_in_place(n)
        return "%s[%s]" % (self._paren(base, bp, 100), ", ".join(codes)), t, 100

    def _slice_in_place(self, n):
        """A slice used where it stands (not bound to a name: that is the assignment's check):
        refuse it when a write could reach its memory before NumPy has read it, i.e. when its
        statement also calls a def that writes into an array, when it is the receiver of an
        in-place method, or when a loop runs over it while the scope writes into its array."""
        par = self.parents
        st = n
        child = None
        path = []
        while st is not None and not isinstance(st, ast.stmt):
            path.append(st)
            child = st
            st = par.get(st)
        if st is None:
            return
        # a slice bound to a name by this statement: _assign_to checks it
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and self._flows_to_target(n, st.value):
            return
        p = par.get(n)
        if isinstance(p, ast.Attribute) and p.value is n and p.attr in views_mod.INPLACE_METHODS:
            self.refuse(n, "`.%s()` of a slice" % p.attr,
                        "NumPy changes the slice's array in place; Viv's slice is a copy. Change the array itself.")
        w = self.views.writer_call(st if not isinstance(st, (ast.If, ast.While, ast.For)) else
                                   (st.test if not isinstance(st, ast.For) else st.iter))
        if w is not None:
            self.refuse(n, "a slice in a statement that calls `%s`, which writes into an array" % w.func.id,
                        "NumPy's slice is a view: the write can show in it before it is read, and Viv's is a copy. "
                        "Assign the call's result first.")
        if isinstance(st, ast.For) and any(x is st.iter or x in getattr(st.iter, "args", []) for x in path):
            tg = [x.id for x in ast.walk(st.target) if isinstance(x, ast.Name)]
            self._row_alias_check(n, n, "a slice", names=tg)

    def _flows_to_target(self, n, value):
        """Whether the value of n is (or is a view of, or an element of a tuple of) value."""
        if n is value:
            return True
        if isinstance(value, (ast.Tuple, ast.List)):
            return any(self._flows_to_target(n, e) for e in value.elts)
        if isinstance(value, ast.IfExp):
            return self._flows_to_target(n, value.body) or self._flows_to_target(n, value.orelse)
        st = self.views.view_step(value)
        if st is not None:
            return self._flows_to_target(n, st[0])
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in self.funcs:
            params = self.funcs[value.func.id].params
            for k in self.views.returns.get(value.func.id, {}):
                a = self.views.arg_at(value, k, params) if not isinstance(k, tuple) else None
                if a is not None and self._flows_to_target(n, a):
                    return True
        return False

    @property
    def parents(self):
        p = getattr(self, "_parents", None)
        if p is None:
            p = {}
            for tree in [self.tree] + list(getattr(self, "_obj_trees", [])):
                for node in ast.walk(tree):
                    for c in ast.iter_child_nodes(node):
                        p[c] = node
            self._parents = p
        return p

    def _e_Call(self, n):
        root = n.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in self.refused_mods and self.var_type(root.id) is None:
            self._e_Name(root)
        # value_and_grad(f)(x, ...) / grad(f)(x, ...)
        if isinstance(n.func, ast.Call) and self._call_name(n.func.func) in ("grad", "viv.grad"):
            code, t = self._grad_call(n, "grad")
            return code, t, 100
        if isinstance(n.func, ast.Call) and self._call_name(n.func.func) in ("value_and_grad", "viv.value_and_grad"):
            self.refuse(n, "value_and_grad outside `l, g = ...`", "Write `l, g = value_and_grad(f)(x, ...)` on its own line.")
        if isinstance(n.func, ast.Lambda):
            return self._call_lambda(n)  # a key function, map, filter (funcs.py)
        if isinstance(n.func, ast.Name) and self._fn_param_index(n.func.id) is not None:
            return self._call_fn_param(n)  # a parameter the callers pass a def to (funcs.py)
        self._ie_class_value(n.func)  # C(1) of an enum class: refused (ints.py)
        name = self._call_name(n.func)
        if name is None and isinstance(n.func, ast.Attribute):
            return self._method_call(n)
        if name is None:
            self.refuse(n.func, "a call of a computed function", "Call a def by name.")
        if isinstance(n.func, ast.Name) and n.func.id in self.funcs:
            return self._user_call(n)
        if isinstance(n.func, ast.Name) and n.func.id in self.ext_funcs:
            return self._ext_call(n, n.func.id, self.ext_funcs[n.func.id][1])
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in self.ext_mods:
            defs = self.ext_mods[n.func.value.id]
            if n.func.attr not in defs:
                self.refuse(n.func, "`%s.%s`: the module has no such def" % (n.func.value.id, n.func.attr), "Check the name.")
            return self._ext_call(n, "%s.%s" % (n.func.value.id, n.func.attr), defs[n.func.attr])
        # a method call on an array: a.sum(), a.copy(), ...
        if isinstance(n.func, ast.Attribute) and not self._is_module_path(n.func):
            return self._method_call(n)
        return self._builtin_call(n, name)

    def _is_module_path(self, f):
        root = f
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and (root.id in self.np or root.id in self.math or root.id == "viv"
                                               or root.id in self.np_random_mods)

    def _call_name(self, f):
        """np.linalg.solve -> 'np.linalg.solve' (whatever numpy is imported as); abs -> 'abs'."""
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            return None
        root = f.id
        if root in self.np_random_mods:
            root = "np.random"
        elif root in self.np:
            root = "np"
        elif root in self.math:
            root = "math"
        elif root in self.from_np and not parts:
            return "np." + self.from_np[root]
        elif root in self.from_math and not parts:
            return "math." + self.from_math[root]
        parts.append(root)
        return ".".join(reversed(parts))

    def _args(self, call, names):
        """Positional and keyword arguments by parameter name."""
        out = {}
        for i, a in enumerate(call.args):
            if isinstance(a, ast.Starred):
                self.refuse(a, "a starred argument", "Pass the values one by one.")
            if i >= len(names):
                self.refuse(a, "too many arguments", "Check the call.")
            out[names[i]] = a
        kw = {}
        for k in call.keywords:
            if k.arg is None:
                self.refuse(k.value, "`**kwargs`", "Pass the arguments by name.")
            if k.arg in names:
                out[k.arg] = k.value
            else:
                kw[k.arg] = k.value
        return out, kw

    def _user_call(self, n):
        f = self.funcs[n.func.id]
        if f.is_test:
            self.refuse(n, "calling a test function", "Tests run on their own; call a helper instead.")
        if self._threaded(f) and n is not getattr(self, "_thread_site", None):
            self._threaded_refusal(n, f)  # it changes an object passed to it (classes.py)
        self._kw_order_check(f, n)
        args, kw = self._args(n, f.params)
        if kw:
            self.refuse(n, "an unknown keyword `%s`" % next(iter(kw)), "Use the parameter names of `%s`." % f.name)
        vals = []
        for i, p in enumerate(f.params):
            if p in args:
                c, t, _ = self.expr(args[p], keep=True)  # an Int, a member, an optional (ints.py)
            elif p in f.defaults:
                # a default Python evaluates once, where the def runs: a number or a module constant
                # is written at the call, anything else is kept in a module variable (classes.py)
                c, t, _ = self._default_arg(f, p, n)
            else:
                self.refuse(n, "a call missing `%s`" % p, "Pass every parameter.")
            c, t = self._ie_arg(f, i, p, c, t, args.get(p, n))  # to the parameter's declared type (ints.py)
            if t.kind in ("none", "tuple", "module"):
                self.refuse(args.get(p, n), "passing %s" % self._kind_words(t), "Pass numbers, bools, strings and arrays.")
            if t.kind == "func":
                self._fn_arg(f, i, p, t, args.get(p, n))  # a def passed as a value (funcs.py)
            vals.append((c, t))
        # a dict argument: one map type per parameter, whatever the site (maps.py), checked
        # against a `dict[...]` annotation
        vals = self._map_args(f, vals, args, n)
        self._check_hints(f, vals, args, n)  # annotations are evidence, never a source (infer.py)
        # the def, or its clone for these argument types (infer.py)
        v = self._variant_of(f, [t for _, t in vals], n)
        self._note_tainted_args(v, [t for _, t in vals])
        for i, (p, (c, t)) in enumerate(zip(f.params, vals)):
            if t.kind != "unknown" and t.kind != "func":
                old = v.param_types[i]
                j = join(old, t)
                if j is None:
                    self.refuse(args.get(p, n), "`%s` called with %s for `%s`, and elsewhere with %s" % (f.name, self._kind_words(t), p, self._kind_words(old)),
                                "Write two functions, one per kind of argument.")
                v.param_types[i] = j
        self._call_variant[id(n)] = v
        for i, p in enumerate(f.params):
            self._note_call(v, n, i, args.get(p), vals[i][1])  # a new list passed? (infer.py)
        codes = [c for c, _ in vals]
        ret = self._ie_ret_decl(v) or v.ret or v.prev_ret or self._unknown_ret(v)
        if ret.kind == "tuple" and n is not getattr(self, "tuple_site", None):
            self.refuse(n, "a call of `%s`, which returns %d values, used as one value" % (f.name, len(ret.elts)),
                        "Unpack it on its own line: `a, b = %s(...)`." % f.name, gap="R5")
        return "%s(%s)" % (self._viv_def_name(v), ", ".join(codes)), ret, 100

    def _grad_call(self, n, which):
        self.inexact(n, "%s (Viv's reverse mode; the CPython reference differentiates by the complex step)" % which)
        inner = n.func
        if len(inner.args) != 1 or not isinstance(inner.args[0], ast.Name) or inner.args[0].id not in self.funcs:
            self.refuse(inner, "%s of something other than a def" % which, "Pass the name of a def.")
        fname = inner.args[0].id
        argnum = 0
        for k in inner.keywords:
            if k.arg == "argnums" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, int):
                argnum = k.value.value
            else:
                self.refuse(k.value, "%s's `%s=`" % (which, k.arg), "Use argnums=<one integer>; several arguments need packing (lib/params).")
        if len(inner.args) > 1:
            self.refuse(inner, "argnums as a positional argument", "Write argnums=k.")
        f = self.funcs[fname]
        if n.keywords:
            self.refuse(n, "keyword arguments to a differentiated call", "Pass the arguments positionally.")
        call = ast.copy_location(ast.Call(func=ast.Name(id=fname, ctx=ast.Load()), args=n.args, keywords=[]), n)
        self._user_call(call)
        arg_codes, types = [], []
        for a in n.args:
            c, t, _ = self.expr(a)
            if not isinstance(a, (ast.Name, ast.Constant)) and (self.pre or t.is_arr()):
                pass
            arg_codes.append(c)
            types.append(t)
        if argnum >= len(arg_codes):
            self.refuse(inner, "argnums past the last argument", "Check argnums.")
        target = fname
        if argnum != 0:
            # Viv differentiates the first parameter: a wrapper puts parameter k first.
            target = "%s__arg%d" % (viv_name(fname), argnum)
            if target not in [w[0] for w in self.wrappers]:
                ps = [viv_name(p) for p in f.params]
                order = [ps[argnum]] + ps[:argnum] + ps[argnum + 1:]
                self.wrappers.append((target, "def %s(%s) {\n    return %s(%s)\n}" % (target, ", ".join(order), viv_name(fname), ", ".join(ps))))
            arg_codes = [arg_codes[argnum]] + arg_codes[:argnum] + arg_codes[argnum + 1:]
            gt = types[argnum]
        else:
            target = viv_name(fname)
            gt = types[0]
        return "%s(%s, %s)" % (which, target, ", ".join(arg_codes)), gt

    def _method_call(self, n):
        f = n.func
        r = self._file_method(n) or self._with_file_method(n)  # open(p).read(), Path(p).read_text(), f.read() (strs.py)
        if r is not None:
            return r
        if f.attr == "format" and isinstance(f.value, ast.Constant) and isinstance(f.value.value, str):
            return self._str_method_format(n, f.value, n.args, n.keywords)  # "...".format(...) (strs.py)
        base, bt, bp = self.expr(f.value)
        m = f.attr
        if bt.kind in ("str", "strs"):
            # the methods of str and of a list of strings: rules/strings.toml
            r = RULES.method(self, n, (base, bt, bp))
            if r is not None:
                return r
            if bt.kind == "str" and m == "format":
                self.refuse(n, "str.format of a template that is not a literal", "Write the template in the call, or an f-string.")
            self.refuse(n, "the method `.%s` of %s" % (m, self._kind_words(bt)),
                        "py2viv maps the string methods Viv has (split, strip, find, replace, join, upper, "
                        "lower, startswith, the is-predicates, ...); this one is not among them." if bt.kind == "str" else
                        "A list of strings has index, count and, where py2viv owns the list, append, pop, insert, "
                        "extend, remove, clear, reverse and sort.")
        r = self._obj_method(n, base, bt, bp)  # obj.m(...), C.m(...), d.get(...) (classes.py)
        if r is not None:
            return r
        if bt.kind == "rng":
            # a Generator's methods: rules/numpy.toml, family np-random-generator
            r = RULES.method(self, n, (base, bt, bp))
            if r is not None:
                return r
            self.refuse(n, "Generator.%s" % m, "The Generator methods py2viv translates are random, integers, uniform, "
                        "standard_normal, normal, choice, shuffle and permutation (tools/py2viv/SUBSET.md, np.random).")
        if not bt.is_arr() and bt.kind != "unknown":
            self.refuse(n, "the method `.%s` on %s" % (m, self._kind_words(bt)), "Viv has no methods; use the function form.")
        if bt.is_arr() and bt.container == "list" and m not in ("append", "extend", "insert", "pop", "remove", "index",
                                                                 "count", "clear", "sort", "copy", "reverse"):
            self.refuse(n, "the method `.%s` of a Python list" % m, "Python raises AttributeError (a list has no .%s); make it a NumPy array: np.array(...)." % m)
        if bt.is_arr() and bt.container == "shape":
            self.refuse(n, "the method `.%s` of a shape (a tuple)" % m, "Take the dimensions: n, m = A.shape.")
        if bt == UNK_C:
            self.refuse(n, "the method `.%s` of a value of unknown type (it comes from a construct refused elsewhere)" % m,
                        "Fix the refusal that produced it.")
        if n.keywords and not RULES.method_takes_keywords(m):
            self.refuse(n, "`.%s(...)` with keyword arguments (axis=, ...)" % m, "Reductions along an axis are R9; write the loop.", gap="R9")
        # the methods: rules/numpy.toml
        r = RULES.method(self, n, (base, bt, bp))
        if r is not None:
            return r
        self.refuse(n, "the method `.%s`" % m, "Viv has no methods; use the NumPy function form (np.%s(a))." % m)

    # the NumPy and builtin mapping: rules/*.toml, applied by rule_engine.py (RULES.md)
    def _builtin_call(self, n, name):
        r = self._ie_builtin(n, name)  # str, abs, min, max, float, int, round of Ints; str of a member (ints.py)
        if r is not None:
            return r
        r = self._obj_builtin(n, name)  # len(d), dict(a=1), dataclasses.replace (classes.py)
        if r is not None:
            return r
        r = self._funcs_builtin(n, name)  # min/max/sorted with key=, of a generator; next (funcs.py)
        if r is not None:
            return r
        r = RULES.call(self, n, name)
        if r is not None:
            return r
        if name.startswith("np.") or name.startswith("math."):
            self.refuse(n, "`%s`" % name, "Not in the NumPy mapping yet (tools/py2viv/SUBSET.md lists what is).")
        self.refuse(n, "a call of `%s`" % name, "Not a def in this file and not in the numeric subset.")

    def _view_base(self, value):
        """The array a NumPy view is taken of, when the Viv translation copies it: a
        transpose of a matrix (np.transpose(A), A.T, A.transpose()) and the real or imaginary
        part of a complex array (np.real(X), X.real, ...). None for anything else."""
        def arr_rank(node):
            if isinstance(node, ast.Name):
                t = self.var_type(node.id)
                if t is not None and t.is_arr():
                    return t
            return None
        if isinstance(value, ast.Attribute) and value.attr in ("T", "real", "imag"):
            t = arr_rank(value.value)
            if t is not None and ((value.attr == "T" and t.rank == 2) or (value.attr != "T" and t.kind == "carr")):
                return value.value
            return None
        if isinstance(value, ast.Call) and value.args:
            name = self._call_name(value.func)
            t = arr_rank(value.args[0])
            if t is None:
                return None
            if name == "np.transpose" and t.rank == 2:
                return value.args[0]
            if name in ("np.real", "np.imag") and t.kind == "carr":
                return value.args[0]
        if isinstance(value, ast.Call) and not value.args and isinstance(value.func, ast.Attribute) \
                and value.func.attr == "transpose" and not self._is_module_path(value.func):
            t = arr_rank(value.func.value)
            return value.func.value if t is not None and t.rank == 2 else None
        return None

    @property
    def views(self):
        """The whole-module view analysis (views.py) of this pass, with the types it has."""
        if self._views is None:
            self._views = views_mod.Views(self.tree, self.funcs, self._call_name, self._rank_of,
                                          methods=getattr(self, "method_keys", {}),
                                          fn_flows=getattr(self, "fn_flows", None))
        return self._views

    @property
    def order(self):
        """Which values may be held in a transpose's memory order (views.Order)."""
        if self._order is None:
            self._order = views_mod.Order(self.views, self.tree)
        return self._order

    def _transpose_of(self, n):
        """X when n is `X.T`, np.transpose(X) or X.transpose() of a matrix X that is itself in
        row order; else None."""
        x = None
        if isinstance(n, ast.Attribute) and n.attr == "T":
            x = n.value
        elif isinstance(n, ast.Call) and not n.keywords and self._call_name(n.func) == "np.transpose" and len(n.args) == 1:
            x = n.args[0]
        elif isinstance(n, ast.Call) and not n.args and not n.keywords and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "transpose" and not self._is_module_path(n.func):
            x = n.func.value
        if x is None:
            return None
        if isinstance(x, ast.Name):
            t = self.var_type(x.id)
            if t is None or not t.is_arr() or t.rank != 2:
                return None
        if self.order.f_order(x, self.scope):
            return None
        return x

    def _rank_of(self, scope_name, name):
        """The rank this translation gives a variable (0 for a number), or None if it has none
        (or if the clones of its def give it different ranks: infer.py)."""
        f = self.funcs.get(scope_name)
        if f is not None and name in f.locals and name not in f.globals_decl:
            sks = [getattr(v, "sid", None) or v.name for v in self._variants(f)]
        else:
            sks = ["<module>"]
        ranks = set()
        for sk in sks:
            vn = self.versions.get((sk, name), viv_name(name))
            t = self.cur_types.get((sk, vn)) or self.prev_types.get((sk, vn))
            if t is None or t.kind in ("unknown", "func", "module", "tuple", "none"):
                return None
            ranks.add(t.rank if t.is_arr() else 0)
        return ranks.pop() if len(ranks) == 1 else None

    def _row_alias_check(self, src, node, what="a row", names=()):
        """A row of a matrix, a slice, a transpose, or the real or imaginary part of a complex
        array is a view in NumPy (a row of a list of lists is a shared list) and a copy in Viv.
        That is the same only while nothing writes into the array, or into the view, while the
        other is read; refuse the view when the scope writes into either anywhere (views.py:
        stores, `+=` of a whole array, in-place methods, a def that writes into its argument or a
        module array). `names` are the names the view is bound to (a store through one of them
        after the binding, in straight-line code, is refused at the store instead)."""
        rs = self.views.roots(src)
        bases = [r for r, v in rs if v] or [r for r, v in rs]
        if not bases:
            return
        name = bases[0]
        found = self.views.observed(self.scope, bases + list(names), skip=self._later_stores(node, names),
                                    at=self._top_stmt(node))
        if found is None:
            return
        site, how, root = found
        line = getattr(site, "lineno", 0)
        if root in names:
            self.refuse(node, "%s of `%s` bound to `%s`, which this code writes through (%s, line %d)" % (what, name, root, how, line),
                        "NumPy writes through the view into `%s`; Viv's is a copy. Take a copy in Python (`.copy()`), "
                        "or write into `%s` itself." % (name, name))
        if how.startswith("a call of"):
            self.refuse(node, "%s of `%s`, which %s (line %d)" % (what, name, how, line),
                        "The view shares `%s`'s memory in NumPy and is a copy in Viv; take a copy in Python (`.copy()`)." % name)
        if how.startswith("`."):
            self.refuse(node, "%s of `%s`, which this code also changes in place" % (what, name), "Index the matrix, A[i, j].")
        if what == "a row":
            self.refuse(node, "a row of `%s`, which this code also writes into" % name,
                        "A row is a view in NumPy (a shared list in Python) and a copy in Viv; index the matrix, A[i, j].")
        if what == "a slice":
            self.refuse(node, "a slice of `%s`, which this code also writes into (line %d)" % (name, line),
                        "A slice is a view in NumPy and a copy in Viv; take a copy in Python (`.copy()`), or index `%s` itself." % name)
        if isinstance(src, ast.Call) and isinstance(src.func, ast.Name) and src.func.id in self.funcs:
            self.refuse(node, "a view of `%s` (`%s` may return one), which this code also writes into" % (name, src.func.id),
                        "A slice, a row or a transpose is a view in NumPy and a copy in Viv; return a copy (`.copy()`) "
                        "or index the original array.")
        self.refuse(node, "%s of `%s`, which this code also writes into" % (what, name),
                    "A transpose, and the real or imaginary part of a complex array, is a view in NumPy "
                    "and a copy in Viv; index the original array.")

    def _top_stmt(self, node):
        """The statement of the current scope's own body that holds node, or None."""
        _, body = self.views.scope_body(self.scope)
        tops = set(map(id, body))
        n = node
        while n is not None and id(n) not in tops:
            n = self.parents.get(n)
        return n

    def _later_stores(self, node, names):
        """Stores `b[i] = v` through a name bound here, after this statement in straight-line
        code: the store is refused where it is (_subscript_target), with its own message."""
        if not names or self.nest() != 0:
            return None
        line = getattr(node, "end_lineno", getattr(node, "lineno", 0))
        return lambda root, site, how: root in names and how == "a store into an element or a slice" \
            and getattr(site, "lineno", 0) > line

    def _in_grad(self):
        return self.scope is not None and self.scope.name in self.grad_defs

    def _shape_args(self, node):
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.expr(e)[0] for e in node.elts]
        c, t, _ = self.expr(node)
        if t.kind == "num":
            return [c]
        if t.container == "shape":
            self.refuse(node, "a shape held in a variable", "Pass the dimensions: np.zeros((n, m)).")
        self.refuse(node, "this shape", "Pass a number or a tuple of numbers.")

    # ------------------------------------------------ output
    def _assemble(self):
        header = [
            "###",
            "Translated from %s by py2viv %s (numeric Python -> Viv)." % (os.path.basename(self.filename), VERSION),
            "Python source sha256: %s" % hashlib.sha256(self.src.encode()).hexdigest(),
            "Each print(...) is one print(...) line; fmt(x) writes a float as Python's repr does.",
            "A `raise`, a failed assert and every guard `stop` with Python's exception line.",
            "###",
            "",
        ]
        lines = []
        pymap = []
        posmap = []  # the Python span of each Viv line, as posmap_entry() writes it
        for h in header + (self.viv_imports + [""] if self.viv_imports else []):
            lines.append(h)
            pymap.append(0)
            posmap.append(None)
        # prelude: the helpers used, with their dependencies, in a fixed order
        need = set(self.used)
        changed = True
        while changed:
            changed = False
            for h in list(need):
                for d in PRELUDE[h][0].split():
                    if d and d not in need:
                        need.add(d)
                        changed = True
        stops = list(self.stops)
        for name in PRELUDE:
            if name in need:
                for m in RULES.helper_stops.get(name, []):
                    stops.append({"line": 0, "message": m, "exact": False, "prefix": True})
                for l in PRELUDE[name][1].strip("\n").splitlines():
                    lines.append(re.sub(r"\{E:(\w+)\}", lambda m: HOST_ERRORS[m.group(1)].replace('"', "'"), l))
                    pymap.append(0)
                    posmap.append(None)
                lines.append("")
                pymap.append(0)
                posmap.append(None)
        for name, text in self.wrappers:
            for l in text.splitlines():
                lines.append(l)
                pymap.append(0)
                posmap.append(None)
            lines.append("")
            pymap.append(0)
            posmap.append(None)
        body = list(self.out)
        for t in self.tests:
            body.append((0, "", 0, None))
            body.extend(t)
        # flush comments after the last statement
        for indent, text, line, pos in body:
            for sub in text.split("\n"):
                lines.append(("    " * indent + sub) if sub else "")
                pymap.append(line)
                posmap.append(posmap_entry(pos))
        return {
            "viv": "\n".join(lines) + "\n",
            "refusals": self.refusals,
            "linemap": pymap,
            "posmap": posmap,
            "sites": self.sites,
            "stops": stops,
            "inexact": [{"what": w, "line": l} for w, l in sorted(self.inexact_ops.items(), key=lambda kv: kv[1])],
        }


# ---------------------------------------------------------------- API


def translate(src, filename="<python>", print_mode="exact"):
    """Translate Python source. Returns a dict: viv (text), refusals (list of Refusal),
    linemap (Python line of each Viv line, 0 for generated), sites (print sites), and
    messages (the refusals formatted for a terminal)."""
    try:
        tr = Translator(src, filename, Options(print_mode=print_mode))
    except SyntaxError as e:
        r = Refusal(None, "a syntax error: %s" % e.msg, "This CPython cannot parse the file.")
        # the location is copied out of e now: Python unbinds `e` when the except clause ends,
        # and manifest() reads the location after that
        loc = (e.lineno or 1, (e.offset or 1) - 1, e.lineno or 1, e.offset or 1)
        r.where = lambda: loc
        return {"viv": "", "refusals": [r], "linemap": [], "posmap": [], "sites": [], "stops": [], "inexact": [],
                "messages": [format_refusal(filename, src.splitlines(), r)]}
    res = tr.translate()
    lines = src.splitlines()
    # one message per distinct location and text
    seen = set()
    msgs = []
    uniq = []
    for r in sorted(res["refusals"], key=lambda r: r.where()):
        key = (r.where(), r.what)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        msgs.append(format_refusal(filename, lines, r))
    res["refusals"] = uniq
    res["messages"] = msgs
    return res


def manifest(res, filename, src, print_mode):
    return {
        "translator": "py2viv",
        "version": VERSION,
        "source": os.path.basename(filename),
        "sha256": hashlib.sha256(src.encode()).hexdigest(),
        "print_mode": print_mode,
        "python": "%d.%d" % HOST_PY,
        "sites": res["sites"],
        "stops": res.get("stops", []),
        "inexact": res.get("inexact", []),
        "linemap": res["linemap"],
        "posmap": res.get("posmap", []),
        "refusals": [
            {"line": r.where()[0], "col": r.where()[1] + 1, "what": r.what, "hint": r.hint, "gap": r.gap,
             "cascade": getattr(r, "cascade", False), "end_line": r.where()[2], "end_col": r.where()[3] + 1}
            for r in res["refusals"]
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Translate numeric Python to Viv.")
    ap.add_argument("file", help="the Python file, or - for standard input")
    ap.add_argument("-o", "--output", help="write the Viv here (default: standard output)")
    ap.add_argument("--manifest", help="write the print sites, line map and refusals as JSON")
    ap.add_argument("--print", dest="print_mode", choices=["exact", "plain"], default="exact")
    ap.add_argument("--json", action="store_true", help="refusals as one JSON object on stderr")
    args = ap.parse_args(argv)
    src = sys.stdin.read() if args.file == "-" else open(args.file).read()
    res = translate(src, "<stdin>" if args.file == "-" else args.file, print_mode=args.print_mode)
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest(res, args.file, src, args.print_mode), f, indent=1)
    if res["refusals"]:
        if args.json:
            json.dump(manifest(res, args.file, src, args.print_mode)["refusals"], sys.stderr)
            sys.stderr.write("\n")
        else:
            for m in res["messages"]:
                sys.stderr.write(m + "\n")
            sys.stderr.write("%d construct%s outside the numeric subset (tools/py2viv/SUBSET.md)\n"
                             % (len(res["refusals"]), "" if len(res["refusals"]) == 1 else "s"))
        return 2
    if args.output:
        with open(args.output, "w") as f:
            f.write(res["viv"])
    else:
        sys.stdout.write(res["viv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

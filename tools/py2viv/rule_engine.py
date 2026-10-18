# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The rule engine: py2viv's NumPy, math and builtin mappings, read from rules/*.toml.

The format is specified in tools/py2viv/RULES.md. This module loads the rule files, matches a
call, method, attribute or unpacking against them, binds the arguments, and runs the rule's
steps against the Translator (py2viv.Translator), which still owns expression translation,
types, refusals and emission. Standard library only (tomllib: Python 3.11+).

    python3 tools/py2viv/rule_engine.py            check the rule files and print a summary
    python3 tools/py2viv/rule_engine.py --table    a Markdown table of every rule (for SUBSET.md)
"""

import ast
import fnmatch
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(HERE, "rules")
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if not os.path.isdir(os.path.join(REPO_ROOT, "lib")) and os.environ.get("VIV_ROOT"):
    REPO_ROOT = os.environ["VIV_ROOT"]       # a copy outside the repository (vipyr's wheel): Viv's library root
# Rule files, in lookup order. helpers.toml and modules.toml hold no rules.
RULE_FILES = ("strings.toml", "builtins.toml", "numpy.toml")
EXACTNESS = ("exact", "tolerance", "refuse")
MODULE_ROOTS = ("np", "math", "random")


class RuleError(Exception):
    """A malformed rule file (raised at load time, never while translating)."""


def lib_module_text(rel):
    """A Viv library module (a helper's `file`, relative to the repository root) as a
    translation carries it: without its `###` header, its `#!` lines (a helper's `deps` stand
    for its imports) and its `test` blocks (they run when the module is the program,
    as an import leaves them behind too)."""
    path = os.path.join(REPO_ROOT, rel)
    try:
        lines = open(path).read().splitlines()
    except OSError as e:
        raise RuleError("helper file %s: %s" % (rel, e))
    out = []
    i = 0
    if lines and lines[0].strip() == "###":
        i = 1
        while i < len(lines) and lines[i].strip() != "###":
            i += 1
        i += 1
    depth = 0
    in_test = False
    in_import = False
    for line in lines[i:]:
        code = line.split("#", 1)[0] if '"' not in line else line
        if not in_test and line.lstrip().startswith("test ") and code.rstrip().endswith("{"):
            in_test = True
            depth = 0
        if in_test:
            quoted = False
            for ch in line:
                if ch == '"':
                    quoted = not quoted
                elif not quoted and ch == "#":
                    break
                elif not quoted and ch == "{":
                    depth += 1
                elif not quoted and ch == "}":
                    depth -= 1
            if depth <= 0:
                in_test = False
            continue
        if line.startswith("#!"):
            continue
        # an import statement (the helper's `deps` stand for it), its list's lines too
        if in_import:
            in_import = ")" not in line
            continue
        if re.match(r"(import\s+[A-Za-z_]|from\s+[A-Za-z_][\w.]*\s+import\b)", line):
            in_import = "(" in line and ")" not in line
            continue
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- the expression language
#
# Conditions (`if`), types (`type`), precedences (`prec`) and `{...}` holes in templates and
# messages are Python expressions, parsed with `ast` and evaluated over a whitelist: names,
# constants, attribute reads on an argument, comparisons, `and`/`or`/`not`, `a if c else b`,
# `+`/`-`, list and tuple literals, a constant subscript, and calls of the FUNCS below.

_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
            ast.BinOp, ast.Add, ast.Sub, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
            ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Name, ast.Load, ast.Constant, ast.Attribute,
            ast.Call, ast.keyword, ast.IfExp, ast.List, ast.Tuple, ast.Subscript, ast.Slice)
_compiled = {}


def _parse(src):
    tree = _compiled.get(src)
    if tree is None:
        tree = ast.parse(src, mode="eval")
        for n in ast.walk(tree):
            if not isinstance(n, _ALLOWED):
                raise RuleError("`%s`: %s is not allowed in a rule expression" % (src, type(n).__name__))
        _compiled[src] = tree
    return tree


class Env:
    """What a rule expression can see: the bound arguments (as Views), the rule's variables,
    and the type constants and functions the translator provides."""

    def __init__(self, ctx):
        self.ctx = ctx

    def lookup(self, name):
        c = self.ctx
        if name in c.views:
            return c.views[name]
        if name in c.values:
            return c.values[name]
        if name in c.vars:
            return c.vars[name]
        g = c.engine.globals
        if name in g:
            return g[name]
        raise RuleError("rule %s: unknown name `%s`" % (c.rule.id, name))


def evaluate(src, ctx):
    return _ev(_parse(src).body, Env(ctx))


def _ev(n, env):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        return env.lookup(n.id)
    if isinstance(n, ast.Attribute):
        base = _ev(n.value, env)
        if not isinstance(base, View):
            # a type the translator passed (a binop rule's `rt`): its fields
            if n.attr in ("kind", "rank", "py", "container") and hasattr(base, "is_arr"):
                return getattr(base, n.attr)
            raise RuleError("`.%s`: attributes are read from arguments only" % n.attr)
        return base.attr(n.attr)
    if isinstance(n, ast.BoolOp):
        if isinstance(n.op, ast.And):
            v = True
            for x in n.values:
                v = _ev(x, env)
                if not v:
                    return v
            return v
        v = False
        for x in n.values:
            v = _ev(x, env)
            if v:
                return v
        return v
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, env)
        return (not v) if isinstance(n.op, ast.Not) else -v
    if isinstance(n, ast.BinOp):
        a, b = _ev(n.left, env), _ev(n.right, env)
        return a + b if isinstance(n.op, ast.Add) else a - b
    if isinstance(n, ast.Compare):
        left = _ev(n.left, env)
        for op, rn in zip(n.ops, n.comparators):
            right = _ev(rn, env)
            ok = {ast.Eq: lambda x, y: x == y, ast.NotEq: lambda x, y: x != y,
                  ast.Lt: lambda x, y: x < y, ast.LtE: lambda x, y: x <= y,
                  ast.Gt: lambda x, y: x > y, ast.GtE: lambda x, y: x >= y,
                  ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y}[type(op)](left, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(n, ast.IfExp):
        return _ev(n.body, env) if _ev(n.test, env) else _ev(n.orelse, env)
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_ev(e, env) for e in n.elts]
    if isinstance(n, ast.Subscript):
        return _ev(n.value, env)[_ev(n.slice, env)]
    if isinstance(n, ast.Slice):
        return slice(*[_ev(x, env) if x is not None else None for x in (n.lower, n.upper, n.step)])
    if isinstance(n, ast.Call):
        if not isinstance(n.func, ast.Name) or n.func.id not in env.ctx.engine.funcs:
            raise RuleError("only the rule functions can be called: %s" % ", ".join(sorted(env.ctx.engine.funcs)))
        f = env.ctx.engine.funcs[n.func.id]
        args = [_ev(a, env) for a in n.args]
        kw = {k.arg: _ev(k.value, env) for k in n.keywords}
        return f(env.ctx, *args, **kw)
    raise RuleError("cannot evaluate %s" % type(n).__name__)


# ---------------------------------------------------------------- templates
#
# `{x}` is the Viv code of argument x; `{x:71}` the same, parenthesized unless its precedence
# is at least 71; `{x|0}` the code of x, or the text `0` when x was not passed; anything else
# in braces is an expression, and its value is written with str(). `{{` and `}}` are braces.

_HOLE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
_SIMPLE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?::(\d+)|\|(.*))?$")


def render(text, ctx):
    _check_once(text, ctx)

    def one(m):
        tok = m.group(0)
        if tok == "{{":
            return "{"
        if tok == "}}":
            return "}"
        inner = m.group(1).strip()
        sm = _SIMPLE.match(inner)
        if sm and sm.group(3) is not None and sm.group(1) not in ctx.views:
            return sm.group(3)  # {kw_rel_tol|1e-09}: a keyword that was not passed
        if sm and isinstance(ctx.views.get(sm.group(1)), View):
            v = ctx.views[sm.group(1)]
            if sm.group(3) is not None:
                return v.code_or(sm.group(3))
            if sm.group(2) is not None:
                return ctx.tr._paren(v.code, v.prec, int(sm.group(2)))
            return v.code
        return str(evaluate(inner, ctx))
    return _HOLE.sub(one, text)


def _check_once(text, ctx):
    """A template that writes an argument's code twice evaluates it twice. That is the same
    value only if evaluating it has no effect: refuse an argument whose Viv code calls a def
    of the program (which may print or write into an array), since Python evaluates it once."""
    counts = {}
    for m in _HOLE.finditer(text):
        sm = _SIMPLE.match((m.group(1) or "").strip())
        if sm and sm.group(3) is None and isinstance(ctx.views.get(sm.group(1)), View):
            counts[sm.group(1)] = counts.get(sm.group(1), 0) + 1
    for nm, k in counts.items():
        if k > 1:
            refuse_if_effectful(ctx, ctx.views[nm])


# The helpers that draw from a random generator (lib/nprandom): a call of one is an effect.
DRAWS = re.compile(r"(?<![\w.])(py_npr_|py_npg_|npr_)\w*\(")


def refuse_if_effectful(ctx, v):
    """Refuse argument v when its Viv code calls a def of the program, or draws random numbers
    (see _check_once)."""
    tr = ctx.tr
    if v.node is None or tr is None:
        return
    code = v.code
    if DRAWS.search(code):
        tr.refuse(v.node, "%s of an expression that draws random numbers" % display_name(ctx),
                  "The translation evaluates this argument more than once and Python once; assign it to a variable first.")
    vname = ctx.engine.globals.get("viv_name", lambda n: n)
    for f in _effect_names(tr):
        if re.search(r"(?<![\w.])%s\(" % re.escape(vname(f)), code):
            tr.refuse(v.node, "%s of an expression that calls `%s`" % (display_name(ctx), f),
                      "The translation evaluates this argument more than once and Python once; "
                      "assign it to a variable first.")


def display_name(ctx):
    """The name a refusal shows: `np.trace`, `abs`, or `.sum` for a method."""
    return "." + ctx.name if ctx.rule.kind in ("method", "stmt", "attr") else ctx.name


# ---------------------------------------------------------------- bound arguments


class View:
    """One bound argument of a call. Its Viv code and type are computed when first read
    (lazily) or by an `eval` step, once; its syntax can be read without translating it."""

    SYNTAX = {"given", "lit", "strlit", "syntax", "callname", "grows", "f_order", "transposed", "axis_lit"}
    TYPED = {"code", "prec", "t", "kind", "rank", "py", "container", "arr", "scalar", "words",
             "dims", "ndims", "dims_int"}

    def __init__(self, ctx, name, node, bind="expr", default=None, value=None):
        self.ctx = ctx
        self.name = name
        self.node = node
        self.bind = bind          # expr | shape | receiver
        self.default = default    # the pattern's default, as Viv text
        self._done = value        # (code, T, prec), already translated (a method's receiver)
        self.passed = node is not None or value is not None
        self.optional = True      # False: the pattern gives no default, so it must be passed
        self.default_node = None  # the default as a Python constant (translated when read)
        self._dims = None

    # translation
    def force(self):
        if self._done is None:
            tr = self.ctx.tr
            if self.node is None and self.default_node is not None:
                # a parameter not passed: its default, as if it had been written
                self._done = tr.expr(ast.copy_location(self.default_node, self.ctx.node))
                return self._done
            if self.node is None and self.optional:
                # a parameter whose default is None, not passed: its type is `none`, and its
                # code is not Viv (a rule that writes it without testing `.given` fails to build)
                self._done = ("<%s not passed>" % self.name, self.ctx.engine.globals["NONE"], 100)
                return self._done
            if self.node is None:
                # a required argument that was not passed: Python raises TypeError
                self.ctx.engine.refuse_missing(self.ctx, self.name)
            if self.bind == "shape":
                self._dims = self._shape()
                self._done = (", ".join(c for c, _ in self._dims), None, 100)
            elif self.bind == "seq":
                # a tuple or list literal of arrays (np.concatenate((a, b)))
                if not isinstance(self.node, (ast.Tuple, ast.List)):
                    tr.refuse(self.node, "a sequence of arrays that is not written out",
                              "Write the arrays in the call: np.concatenate((a, b)).")
                self._elts = [tr.expr(e) for e in self.node.elts]
                self._done = (", ".join(c for c, _, _ in self._elts), None, 100)
                for e, (_, t, _) in zip(self.node.elts, self._elts):
                    self._no_object(e, t)
            else:
                self._done = tr.expr(self.node)
                self._no_object(self.node, self._done[1])
        return self._done

    def _no_object(self, node, t):
        """No rule takes an object (a record: classes.py): Python raises TypeError for a class's
        object in a builtin, math or NumPy function, and a dict or named tuple is not an array."""
        if t is not None and t.kind == "rec":
            tr = self.ctx.tr
            tr.refuse(node, "`%s` of %s" % (self.ctx.name, tr._kind_words(t)),
                      "No builtin, math or NumPy function takes an object here; pass its fields.")
        if t is not None and t.kind in ("str", "strs") and self.ctx.rule.source != "strings.toml" and not self.ctx.rule.takes_text:
            # a string reaches a rule of builtins.toml or numpy.toml only when rules/strings.toml
            # has no translation for it: Python raises TypeError, or answers by an order of
            # strings py2viv does not follow (min, max)
            tr = self.ctx.tr
            tr.refuse(node, "`%s` of %s" % (display_name(self.ctx), tr._kind_words(t)),
                      "Python raises TypeError for a string here (min and max compare strings, which py2viv does not "
                      "translate); convert it first, or loop over it.")

    def _shape(self):
        """A shape: a number or a tuple (or list) of numbers, each with its type."""
        tr = self.ctx.tr
        node = self.node
        if isinstance(node, (ast.Tuple, ast.List)):
            out = []
            for e in node.elts:
                c, t, _ = tr.expr(e)
                out.append((c, t))
            return out
        c, t, _ = tr.expr(node)
        if t.kind == "num":
            return [(c, t)]
        if t.container == "shape":
            tr.refuse(node, "a shape held in a variable", "Pass the dimensions: np.zeros((n, m)).")
        tr.refuse(node, "this shape", "Pass a number or a tuple of numbers.")

    def code_or(self, default):
        if not self.passed:
            return default
        return self.force()[0]

    @property
    def code(self):
        if not self.passed and self.default is not None:
            return self.default
        return self.force()[0]

    @property
    def prec(self):
        if not self.passed and self.default is not None:
            return 100
        return self.force()[2]

    @property
    def t(self):
        return self.force()[1]

    def __getattr__(self, a):
        # for the rule functions in Python: x.rank reads as the expression `x.rank` does
        if a.startswith("_") or a not in self.SYNTAX | self.TYPED:
            raise AttributeError(a)
        return self.attr(a)

    def attr(self, a):
        if a in self.SYNTAX:
            n = self.node
            if a == "given":
                return self.passed
            if a == "lit":
                return n.value if isinstance(n, ast.Constant) else None
            if a == "strlit":
                return n.value.strip().lower() if isinstance(n, ast.Constant) and isinstance(n.value, str) else None
            if a == "syntax":
                return type(n).__name__ if n is not None else None
            if a == "callname":
                return self.ctx.tr._call_name(n) if n is not None else None
            if a == "f_order":
                # the value may be held in a transpose's memory order (views.Order)
                return n is not None and self.ctx.tr.order.f_order(n, self.ctx.tr.scope)
            if a == "transposed":
                # `X.T`, np.transpose(X), X.transpose() of a matrix X in row order
                return self.ctx.tr._transpose_of(n) is not None
            if a == "axis_lit":
                # an int literal (a negative one too), or None
                if isinstance(n, ast.Constant) and type(n.value) is int:
                    return n.value
                if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) \
                        and type(n.operand.value) is int:
                    return -n.operand.value
                return None
            if a == "grows":
                # a name bound only to `[]` and read nowhere an alias of it could be taken
                return isinstance(n, ast.Name) and self.ctx.tr._grow_refusal(n) is None
        if a in self.TYPED:
            if a == "code":
                return self.code
            if a == "prec":
                return self.prec
            if a == "dims":
                self.force()
                return [c for c, _ in self._dims]
            if a == "ndims":
                self.force()
                return len(self._dims)
            if a == "dims_int":
                self.force()
                return all(t.kind == "num" and t.py == "int" for _, t in self._dims)
            t = self.t
            if a == "t":
                return t
            if a == "arr":
                return t.is_arr()
            if a == "scalar":
                return t.scalar()
            if a == "words":
                return self.ctx.tr._kind_words(t)
            return getattr(t, a)
        raise RuleError("rule %s: an argument has no attribute `.%s`" % (self.ctx.rule.id, a))


class Ctx:
    """One application of one rule."""

    def __init__(self, engine, rule, tr, node, name):
        self.engine = engine
        self.rule = rule
        self.tr = tr
        self.node = node
        self.name = name
        self.views = {}
        self.values = {}   # dtype-bound parameters: their raw value
        self.vars = dict(rule.vars)
        self.vars["name"] = name
        self.vars["short"] = name.split(".")[-1]
        self.vars["module"] = name.split(".")[0] if "." in name else ""
        self.vars["in_grad"] = tr._in_grad() if tr is not None else False
        # the module calls np.random.seed somewhere (np.random's legacy draws need a seed)
        self.vars["np_seeded"] = getattr(tr, "np_seeded", False)
        # the module imports NumPy (from 3.12 the builtin sum of a list compensates exact Python
        # floats only, and a list may hold NumPy's float64)
        self.vars["np_imported"] = bool(getattr(tr, "np", None) or getattr(tr, "from_np", None))
        self.extra_pos = []
        self.unknown_kw = []
        self.pattern = None


DECLINE = object()


# ---------------------------------------------------------------- patterns


class Pattern:
    """`np.linalg.solve(A, b)`, `a.sum()`, `a.T`, `np.pi`, `q, r = divmod(a, b)`.

    The parameter list is a Python def signature: `a, /` positional only, `k=0` positional or
    keyword with a default, `*, dtype=None` keyword only, `**kw` keyword arguments not named
    in the signature."""

    # `x + y`: an operator of two arrays (the translator gives each array-array operation to
    # the operator rules, with both operands already translated)
    BINOPS = ("+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=")

    def __init__(self, text, rule_id, context="expr"):
        self.text = text
        self.targets = None
        self.item = None
        body = text.strip()
        bm = re.fullmatch(r"(\w+) (\+|-|\*|/|<=|>=|<|>|==|!=) (\w+)", body)
        if bm:
            self._structural("binop", bm.group(2), None, [bm.group(1), bm.group(3)])
            return
        im = re.fullmatch(r"(\w+)\[([\w:, ]+)\](?: = (\w+))?", body)
        items = [x.strip() for x in im.group(2).split(",")] if im else []
        if im and all(x == ":" or x.isidentifier() for x in items):
            # `a[i]`, `a[:, i]`, `a[i, j]`: a subscript whose named items are arrays (a mask or
            # integer indices, a plain int beside them) and whose other items are whole ranges
            # `:`; `a[i] = v` a store through one. `a[None]`: the axes a subscript adds.
            if items == ["None"]:
                self._structural("newaxis", "[None]", im.group(1), [])
                return
            if items[-1] == ":":
                raise RuleError("rule %s: bad subscript pattern %r (trailing whole ranges are left out, as "
                                "the translator leaves them out)" % (rule_id, text))
            key = "[%s]" % ",".join(":" if x == ":" else "#" for x in items)
            names = [x for x in items if x != ":"]
            if im.group(3):
                self._structural("index-store", key, im.group(1), names + [im.group(3)])
            else:
                self._structural("index", key, im.group(1), names)
            return
        cm = re.fullmatch(r"([\w.]+)\((.*)\)\[(\w+)\]", body)
        if cm:
            # `np.nonzero(a)[k]`: one item of a call's tuple result
            inner = Pattern("%s(%s)" % (cm.group(1), cm.group(2)), rule_id)
            self.__dict__.update(inner.__dict__)
            self.text = text
            self.kind = "item"
            self.item = cm.group(3)
            return
        sm = re.fullmatch(r"(\w+)\[(\w+):(\w+):(\w+)\]", body)
        if sm:
            # `a[lo:hi:step]`: one axis of a subscript, a range (the translator gives each
            # range of a subscript to the slice rules, with the axis and its length)
            self.callee = body
            self.is_call = False
            self.receiver = sm.group(1)
            self.key = "[:]"
            self.glob = False
            self.kind = "slice"
            self.params = [(nm, "poskw", None, True) for nm in sm.groups()[1:]]
            self.defaults = {nm: None for nm in sm.groups()[1:]}
            self.varkw = self.varargs = None
            self.anyargs = False
            return
        if " = " in body:
            lhs, body = body.split(" = ", 1)
            self.targets = [t.strip() for t in lhs.split(",")]
        if "(" in body:
            if not body.endswith(")"):
                raise RuleError("rule %s: bad pattern %r" % (rule_id, text))
            callee, params = body[:-1].split("(", 1)
            self.is_call = True
        else:
            callee, params = body, None
            self.is_call = False
        self.callee = callee.strip()
        parts = self.callee.split(".")
        self.receiver = None
        if len(parts) >= 2 and parts[0] not in MODULE_ROOTS and not self.targets:
            self.receiver = parts[0]
            self.key = ".".join(parts[1:])
        else:
            self.key = self.callee
        self.glob = "*" in self.key
        if self.targets:
            self.kind = "unpack"
        elif self.receiver and context == "stmt":
            self.kind = "stmt"
        elif self.receiver:
            self.kind = "method" if self.is_call else "attr"
        else:
            self.kind = "call" if self.is_call else "const"
        self.params = []   # (name, how, default text, has a default); how: pos | poskw | kw
        self.varkw = None
        self.varargs = None
        self.defaults = {}  # parameter -> its default's Python value (None: no default, or None)
        self.anyargs = params is not None and params.strip() == "..."
        if self.anyargs:
            pass  # any arguments, none bound (a refusal)
        elif params is not None and params.strip():
            try:
                fn = ast.parse("def _(%s): pass" % params).body[0]
            except SyntaxError as e:
                raise RuleError("rule %s: bad parameters in %r: %s" % (rule_id, text, e.msg))
            a = fn.args
            pos = [(p.arg, "pos") for p in a.posonlyargs] + [(p.arg, "poskw") for p in a.args]
            defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
            for (nm, how), d in zip(pos, defaults):
                self.params.append((nm, how, _default_text(d), d is not None))
                self.defaults[nm] = _default_value(d)
            for p, d in zip(a.kwonlyargs, a.kw_defaults):
                self.params.append((p.arg, "kw", _default_text(d), d is not None))
                self.defaults[p.arg] = _default_value(d)
            self.varkw = a.kwarg.arg if a.kwarg else None
            self.varargs = a.vararg.arg if a.vararg else None

    def _structural(self, kind, key, receiver, names):
        """An operator or a subscript: every parameter is positional and required."""
        self.callee = self.text
        self.is_call = False
        self.receiver = receiver
        self.key = key
        self.glob = False
        self.kind = kind
        self.params = [(nm, "pos", None, False) for nm in names]
        self.defaults = {nm: None for nm in names}
        self.varkw = self.varargs = None
        self.anyargs = False

    def matches(self, name):
        return fnmatch.fnmatchcase(name, self.key) if self.glob else name == self.key


def _default_value(d):
    if d is None:
        return None
    return d.value if isinstance(d, ast.Constant) else ast.literal_eval(d)


def _default_text(d):
    if d is None:
        return None
    v = d.value if isinstance(d, ast.Constant) else ast.literal_eval(d)
    if v is None:
        return None
    return v if isinstance(v, str) else repr(v)


# ---------------------------------------------------------------- rules


class Rule:
    FIELDS = {"id", "match", "doc", "context", "params", "when", "cases", "vars", "exactness",
              "rtol", "atol", "evidence", "gap", "lift", "guard", "tests", "note", "family", "setup"}
    CASE_FIELDS = {"if", "refuse", "hint", "gap", "at", "viv", "type", "prec", "use", "hook", "args",
                   "next", "const", "eval", "keywords", "stmts", "targets", "note", "exactness",
                   "rtol", "atol", "evidence", "inexact", "tmps", "hoist", "python"}
    TERMINAL = ("viv", "stmts", "hook", "const", "python")

    def __init__(self, d, source, family=None):
        extra = set(d) - self.FIELDS
        if extra:
            raise RuleError("rule %s (%s): unknown field(s) %s" % (d.get("id"), source, ", ".join(sorted(extra))))
        self.id = d.get("id") or (d["match"][0] if isinstance(d.get("match"), list) else d.get("match"))
        self.source = source
        self.family = family
        m = d["match"]
        self.context = d.get("context", "expr")
        if self.context not in ("expr", "stmt"):
            raise RuleError("rule %s: context is `expr` or `stmt`" % self.id)
        self.patterns = [Pattern(p, self.id, self.context) for p in ([m] if isinstance(m, str) else m)]
        kinds = {p.kind for p in self.patterns}
        if len(kinds) != 1:
            raise RuleError("rule %s: patterns of different kinds (%s)" % (self.id, ", ".join(sorted(kinds))))
        self.kind = kinds.pop()
        if "bind" in d:
            raise RuleError("rule %s: `bind` is gone: every rule binds its arguments strictly (RULES.md)" % self.id)
        self.when = d.get("when")
        self.cases = list(d.get("cases", []))
        for c in self.cases:
            bad = set(c) - self.CASE_FIELDS
            if bad:
                raise RuleError("rule %s: unknown case field(s) %s" % (self.id, ", ".join(sorted(bad))))
        self.vars = dict(d.get("vars", {}))
        self.params = dict(d.get("params", {}))
        for nm, how in self.params.items():
            if how not in ("expr", "shape", "dtype", "seq", "list", "literal"):
                raise RuleError("rule %s: parameter `%s` binds as expr, shape, dtype, seq, list or literal" % (self.id, nm))
        self.exactness = d.get("exactness")
        if self.exactness not in EXACTNESS:
            raise RuleError("rule %s: exactness must be one of %s" % (self.id, ", ".join(EXACTNESS)))
        self.rtol = d.get("rtol")
        self.atol = d.get("atol")
        if self.exactness == "tolerance" and (self.rtol is None or self.atol is None):
            raise RuleError("rule %s: a tolerance rule states rtol and atol" % self.id)
        if self.exactness == "refuse" and any(k in c for c in self.cases for k in self.TERMINAL):
            raise RuleError("rule %s: a refuse rule has no translating case" % self.id)
        if self.exactness == "refuse" and any(p.params for p in self.patterns):
            raise RuleError("rule %s: a refuse rule binds no arguments (write `f(...)`): it refuses before "
                            "translating any" % self.id)
        if self.exactness != "refuse" and not any(k in c for c in self.cases for k in self.TERMINAL + ("next",)):
            raise RuleError("rule %s: an %s rule needs a translating case" % (self.id, self.exactness))
        if self.exactness != "refuse" and any(p.glob for p in self.patterns):
            raise RuleError("rule %s: a `*` pattern may only refuse (a translating rule names each function it "
                            "maps, so a name with another module path in the middle cannot reach it)" % self.id)
        self.evidence = d.get("evidence")
        self.gap = d.get("gap")
        self.lift = d.get("lift")
        self.guard = d.get("guard")
        self.tests = d.get("tests", [])
        self.note = d.get("note")
        self.has_eval_step = any("eval" in c for c in self.cases)
        # a rule outside strings.toml that names strings in its steps takes them (list.append)
        self.takes_text = any("'str'" in c.get("if", "") for c in self.cases)
        # Python lines check_rules.py puts before each generated case (np.random's seed)
        self.setup = list(d.get("setup", []))


class Engine:
    def __init__(self, rules_dir=RULES_DIR, types=None):
        self.rules = []
        self.helpers = {}
        self.helper_stops = {}
        self.module_hints = {}
        self.hooks = dict(HOOKS)
        self.funcs = dict(FUNCS)
        self.globals = dict(types or {})
        for fname in RULE_FILES:
            with open(os.path.join(rules_dir, fname), "rb") as f:
                data = tomllib.load(f)
            for d in data.get("rule", []):
                self.rules.append(Rule(d, fname))
            for fam in data.get("family", []):
                shared = {k: v for k, v in fam.items() if k not in ("rule", "id", "doc")}
                for d in fam.get("rule", []):
                    merged = dict(shared)
                    merged.update(d)
                    merged["vars"] = dict(shared.get("vars", {}), **d.get("vars", {}))
                    self.rules.append(Rule(merged, fname, family=fam.get("id")))
        self.rules = self._file_order(rules_dir)
        with open(os.path.join(rules_dir, "helpers.toml"), "rb") as f:
            for name, h in tomllib.load(f).get("helper", {}).items():
                text = lib_module_text(h["file"]) if "file" in h else h["viv"]
                self.helpers[name] = (" ".join(h.get("deps", [])), text)
                # a stop whose message is the start of NumPy's line (the rest is computed:
                # shapes, an index), which diff.py compares as a prefix
                self.helper_stops[name] = list(h.get("stops", []))
        with open(os.path.join(rules_dir, "modules.toml"), "rb") as f:
            self.module_hints = dict(tomllib.load(f).get("refused_module_hint", {}))
        self._index()

    def _file_order(self, rules_dir):
        """Rules and family members in the order they appear in each file (TOML tables of
        arrays keep order within [[rule]] and within [[family]], not between them)."""
        out = []
        for fname in RULE_FILES:
            text = open(os.path.join(rules_dir, fname)).read()
            heads = re.findall(r"^\[\[(rule|family|family\.rule)\]\]", text, re.M)
            rules = [r for r in self.rules if r.source == fname and r.family is None]
            fams = [r for r in self.rules if r.source == fname and r.family is not None]
            ri = fi = 0
            for h in heads:
                if h == "rule":
                    out.append(rules[ri])
                    ri += 1
                elif h == "family.rule":
                    out.append(fams[fi])
                    fi += 1
        return out

    def _index(self):
        self.by_kind = {}
        for r in self.rules:
            self.by_kind.setdefault(r.kind, []).append(r)
        self.exact = {}
        self.globbed = {}
        for r in self.rules:
            for p in r.patterns:
                if p.glob:
                    self.globbed.setdefault(r.kind, []).append((p, r))
                else:
                    self.exact.setdefault((r.kind, p.key), []).append((p, r))

    def candidates(self, kind, name):
        out = list(self.exact.get((kind, name), []))
        for p, r in self.globbed.get(kind, []):
            if p.matches(name):
                out.append((p, r))
        return out

    # ------------------------------------------------ entry points (called by the Translator)
    def call(self, tr, node, name):
        """A call of a function (np.*, math.*, a builtin). Returns (code, T, prec), or None
        when no rule matches."""
        for pat, rule in self.candidates("call", name):
            ctx = Ctx(self, rule, tr, node, name)
            if self._bind(ctx, pat, node.args, node.keywords) is DECLINE:
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def method(self, tr, node, recv):
        """a.m(...) on an array; recv is the receiver's (code, T, prec). None if no rule."""
        m = node.func.attr
        for pat, rule in self.candidates("method", m):
            ctx = Ctx(self, rule, tr, node, m)
            ctx.views[pat.receiver] = View(ctx, pat.receiver, node.func.value, "receiver", value=recv)
            if self._bind(ctx, pat, node.args, node.keywords) is DECLINE:
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def method_takes_keywords(self, m):
        """Whether a method rule for `.m` names a parameter a keyword can bind (`a.sum(axis=0)`)."""
        return any(p.params for p, _ in self.candidates("method", m))

    def statement(self, tr, node):
        """a.m(...) as a statement (sort in place, fill, list.append refused). Returns the
        Viv statement, or None to translate the call as an expression."""
        m = node.func.attr
        for pat, rule in self.candidates("stmt", m):
            ctx = Ctx(self, rule, tr, node, m)
            ctx.views[pat.receiver] = View(ctx, pat.receiver, node.func.value, "receiver")
            if self._bind(ctx, pat, node.args, node.keywords) is DECLINE:
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r[0]
        return None

    def attribute(self, tr, node, recv):
        """a.T, a.shape, X.real, ... ; recv is the receiver's (code, T, prec)."""
        for pat, rule in self.candidates("attr", node.attr):
            ctx = Ctx(self, rule, tr, node, node.attr)
            ctx.views[pat.receiver] = View(ctx, pat.receiver, node.value, "receiver", value=recv)
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def constant(self, tr, node, name):
        """np.pi, math.inf, np.random.* (refused), ... ; None if no rule."""
        for pat, rule in self.candidates("const", name):
            ctx = Ctx(self, rule, tr, node, name)
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def unpack(self, tr, tgt, value, stmt, name):
        """`w, V = np.linalg.eigh(H)`: True when a rule emitted it."""
        for pat, rule in self.candidates("unpack", name):
            ctx = Ctx(self, rule, tr, value, name)
            ctx.stmt = stmt
            ctx.tgt = tgt
            names = tgt.elts
            ctx.vars["ntargets"] = len(names)
            ctx.vars["targets_named"] = all(isinstance(n, ast.Name) for n in names)
            ctx.vars["targets_plain"] = all(isinstance(n, (ast.Name, ast.Subscript)) for n in names)
            ctx.target_pats = pat.targets
            ctx.target_nodes = names
            if self._bind(ctx, pat, value.args, value.keywords) is DECLINE:
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return True
        return False

    def slice_item(self, tr, node, recv_node, recv, axis, n_code):
        """One range of a subscript, `lo:hi:step` (an ast.Slice) on axis `axis` of the array
        recv (its (code, T, prec)), whose length along that axis is the Viv code n_code.
        Returns the Viv text of that range, or None when no rule matches."""
        for pat, rule in self.candidates("slice", "[:]"):
            ctx = Ctx(self, rule, tr, node, "slice")
            ctx.views[pat.receiver] = View(ctx, pat.receiver, recv_node, "receiver", value=recv)
            for (nm, _, _, _), part in zip(pat.params, (node.lower, node.upper, node.step)):
                if isinstance(part, ast.Constant) and part.value is None:
                    part = None  # a[None:3] is a[:3]
                ctx.views[nm] = View(ctx, nm, part)
            ctx.vars["n"] = n_code
            ctx.vars["axis"] = axis
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r[0]
        return None

    def binop(self, tr, node, op, left, right, rt):
        """`x OP y` of two arrays: left and right are (node, (code, T, prec)), already
        translated; rt is the result's type as the operator's typing found it. Returns
        (code, T, prec), or None when no rule matches."""
        for pat, rule in self.candidates("binop", op):
            ctx = Ctx(self, rule, tr, node, op)
            for (nm, _, _, _), (n, val) in zip(pat.params, (left, right)):
                ctx.views[nm] = View(ctx, nm, n, value=val)
            ctx.vars["op"] = op
            ctx.vars["rt"] = rt
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def index(self, tr, node, key, recv, items, store=None):
        """A subscript with an array among its items (`a[m]`, `A[:, idx]`, `A[I, J]`), or a
        store through one (`store`: the value, (node, (code, T, prec))). recv is the array
        (node, (code, T, prec)); items the named items in order, each (node, (code, T, prec)).
        Returns (code, T, prec) (a store: the statement as code), or None."""
        kind = "index-store" if store is not None else "index"
        for pat, rule in self.candidates(kind, key):
            ctx = Ctx(self, rule, tr, node, key)
            ctx.views[pat.receiver] = View(ctx, pat.receiver, recv[0], "receiver", value=recv[1])
            vals = list(items) + ([store] if store is not None else [])
            for (nm, _, _, _), (n, val) in zip(pat.params, vals):
                ctx.views[nm] = View(ctx, nm, n, value=val)
            ctx.vars["axis"] = key.strip("[]").split(",").index("#")
            ctx.vars["nitems"] = len(key.strip("[]").split(","))
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def newaxis(self, tr, node, recv, axes):
        """A subscript with `None` (np.newaxis) among its items: recv is the value of the
        subscript without them, (node, (code, T, prec)); axes the positions, in the result, of
        the axes of length 1 they add."""
        for pat, rule in self.candidates("newaxis", "[None]"):
            ctx = Ctx(self, rule, tr, node, "[None]")
            ctx.views[pat.receiver] = View(ctx, pat.receiver, recv[0], "receiver", value=recv[1])
            ctx.vars["axes"] = "[%s]" % ", ".join(str(k) for k in axes)
            ctx.vars["nnew"] = len(axes)
            if rule.when and not evaluate(rule.when, ctx):
                continue
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def item(self, tr, node, name):
        """`np.nonzero(a)[k]`: node is the subscript, node.value the call. None if no rule."""
        call = node.value
        for pat, rule in self.candidates("item", name):
            ctx = Ctx(self, rule, tr, call, name)
            if self._bind(ctx, pat, call.args, call.keywords) is DECLINE:
                continue
            ctx.views[pat.item] = View(ctx, pat.item, node.slice)
            ctx.views[pat.item].optional = False
            r = self._run(ctx)
            if r is not DECLINE:
                return r
        return None

    def is_unpack(self, name):
        return bool(self.candidates("unpack", name))

    # ------------------------------------------------ binding
    def _bind(self, ctx, pat, args, keywords):
        """Bind the call's arguments to the pattern's parameters, as Python binds a call of a
        def with that signature. A starred argument, `**kwargs`, and a parameter passed twice
        are refused here. A positional argument past the parameters (and no `*rest`) and a
        keyword the signature does not name (and no `**kw`) are kept aside: the rule's steps
        may refuse them with their own message, and if none does, the rule refuses them before
        it translates (`_leftovers`). A required parameter that is missing is refused once the
        rule's `when` has accepted the call."""
        tr = ctx.tr
        rule = ctx.rule
        ctx.vars["nargs"] = len(args)
        ctx.vars["nkw"] = len(keywords)
        ctx.vars["kw_names"] = [k.arg for k in keywords]
        bound = {}
        extra_kw = {}
        ctx.extra_pos = []
        ctx.unknown_kw = []
        ctx.pattern = pat
        positional = [p for p in pat.params if p[1] in ("pos", "poskw")]
        by_kw = {p[0] for p in pat.params if p[1] in ("poskw", "kw")}
        for i, a in enumerate(args):
            if isinstance(a, ast.Starred):
                tr.refuse(a, "a starred argument", "Pass the values one by one.")
            if i < len(positional):
                bound[positional[i][0]] = a
            elif not pat.varargs and not pat.anyargs:
                ctx.extra_pos.append(a)
        for k in keywords:
            if k.arg is None:
                tr.refuse(k.value, "`**kwargs`", "Pass the arguments by name.")
            if k.arg in by_kw:
                if k.arg in bound:
                    tr.refuse(k.value, "`%s` passed twice (by position and as `%s=`)" % (k.arg, k.arg),
                              "Python raises TypeError; pass it once.")
                bound[k.arg] = k.value
            else:
                extra_kw[k.arg] = k.value
                if not pat.varkw and not pat.anyargs:
                    ctx.unknown_kw.append(k)
        ctx.bound = bound
        ctx.extra_kw = extra_kw
        ctx.vars["extra_kws"] = list(extra_kw)
        if pat.varkw:
            ctx.vars[pat.varkw] = extra_kw
        for nm, how, default, optional in pat.params:
            bind = rule.params.get(nm, "expr")
            node = bound.get(nm)
            if bind == "dtype":
                if node is None:
                    ctx.values[nm] = None
                else:
                    ctx.values[nm] = tr._call_name(node) if not isinstance(node, ast.Constant) else node.value
                continue
            ctx.views[nm] = View(ctx, nm, node, bind, default)
            ctx.views[nm].optional = optional
            dv = pat.defaults.get(nm)
            if dv is not None:
                ctx.views[nm].default_node = ast.Constant(value=dv)
        for nm, node in extra_kw.items():
            ctx.views["kw_" + str(nm)] = View(ctx, "kw_" + str(nm), node)
        if pat.varargs:
            npos = len(positional)
            ctx.views[pat.varargs] = [View(ctx, pat.varargs, a) for a in args[npos:]]
        if rule.when and not evaluate(rule.when, ctx):
            return DECLINE
        for nm, how, default, optional in pat.params:
            if not optional and nm not in bound:
                self.refuse_missing(ctx, nm)
        return None

    def refuse_missing(self, ctx, nm):
        ctx.tr.refuse(ctx.node, "%s without its argument `%s`" % (display_name(ctx), nm),
                      "Python raises TypeError (a missing argument); pass it.")

    @staticmethod
    def _ordinal(n):
        return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, "%dth" % n)

    def _leftovers(self, ctx):
        """Before a rule translates: every argument the call passed is one the rule read. A
        positional argument past the pattern's parameters, or a keyword it does not name, is
        refused (py2viv never drops an argument: NumPy's next parameter changes the result)."""
        pat = ctx.pattern
        if pat is None:
            return
        shown = "`%s`" % pat.text
        if ctx.extra_pos:
            a = ctx.extra_pos[0]
            k = ctx.vars["nargs"] - len(ctx.extra_pos) + 1
            ctx.tr.refuse(a, "%s with a %s positional argument" % (display_name(ctx), self._ordinal(k)),
                          "py2viv translates %s; the next parameter of the Python function is not in the "
                          "subset. Drop it, or write the computation out." % shown)
        if ctx.unknown_kw:
            k = ctx.unknown_kw[0]
            ctx.tr.refuse(k.value, "%s with `%s=`" % (display_name(ctx), k.arg),
                          "py2viv translates %s; this keyword is not in the subset." % shown)

    # ------------------------------------------------ steps
    def _run(self, ctx):
        rule = ctx.rule
        tr = ctx.tr
        if not rule.has_eval_step:
            # no eval step: the named parameters, in pattern order, before the first case
            for v in ctx.views.values():
                if isinstance(v, View) and v.bind != "receiver" and not v.name.startswith("kw_") \
                        and (v.node is not None or not v.optional):
                    v.force()
        for c in rule.cases:
            if "if" in c and not evaluate(c["if"], ctx):
                continue
            if "eval" in c:
                for nm in c["eval"]:
                    if nm not in ctx.views:
                        continue  # a **kw keyword that was not passed
                    for v in (ctx.views[nm] if isinstance(ctx.views[nm], list) else [ctx.views[nm]]):
                        if v.node is not None or v._done is not None or not v.optional:
                            v.force()
                continue
            if "keywords" in c:
                allowed = set(c["keywords"])
                for k in ctx.node.keywords:
                    if k.arg in allowed:
                        continue
                    ctx.vars["kw"] = str(k.arg)
                    tr.refuse(k.value, render(c["refuse"], ctx), render(c["hint"], ctx), c.get("gap"))
                continue
            if "inexact" in c:
                # an operation whose Viv result can differ in the last bits: the manifest lists it
                tr.inexact(ctx.node, render(c["inexact"], ctx))
            if c.get("next"):
                self._keep_translations(ctx)
                return DECLINE
            if "refuse" in c:
                tr.refuse(self._at(ctx, c.get("at")), render(c["refuse"], ctx), render(c["hint"], ctx),
                          c.get("gap", None))
            for u in c.get("use", []):
                tr.use(render(u, ctx))
            for stem in c.get("tmps", []):
                ctx.vars["tmp_" + stem] = tr.new_tmp(stem)
            for h in c.get("hoist", []):
                tr.pre.append(render(h, ctx))
            if not any(k in c for k in Rule.TERMINAL):
                continue  # a step: it recorded an inexact operation or used a helper
            self._leftovers(ctx)
            if "python" in c:
                return self._python(ctx, c["python"])
            if "hook" in c:
                return self.hooks[c["hook"]](ctx, *[render(a, ctx) for a in c.get("args", [])])
            if "const" in c:
                return tr._const_named(c["const"], ctx.node)
            if "stmts" in c:
                return self._stmts(ctx, c)
            code = render(c["viv"], ctx)
            t = self._type(evaluate(c["type"], ctx)) if "type" in c else None
            prec = c.get("prec", 100)
            prec = evaluate(prec, ctx) if isinstance(prec, str) else prec
            return code, t, prec
        return DECLINE

    @staticmethod
    def _keep_translations(ctx):
        """A rule that declines after translating arguments (to read their types) hands the
        translations to the next rule: each argument is translated once, so the statements it
        hoists and the inexact operations it records are not doubled."""
        memo = getattr(ctx.tr, "_xmemo", None)
        if memo is None:
            return
        for v in ctx.views.values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, View) and x.node is not None and x._done is not None and x.bind == "expr":
                    memo[id(x.node)] = x._done

    def _python(self, ctx, template):
        """Desugar: the call becomes the Python expression `template`, whose names are the
        bound arguments (the same nodes, so refusals point where they did); the new nodes take
        the call's location. The translator then translates that expression (or, for an
        unpacking rule, that assignment)."""
        tree = ast.parse(template, mode="eval").body
        nodes = {nm: v.node for nm, v in ctx.views.items() if isinstance(v, View) and v.node is not None}
        uses = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                uses[n.id] = uses.get(n.id, 0) + 1
        for nm, k in uses.items():
            if k > 1 and nm in nodes:
                refuse_if_effectful(ctx, ctx.views[nm])

        class Sub(ast.NodeTransformer):
            def visit_Name(self, n):
                if n.id in nodes:
                    return nodes[n.id]
                raise RuleError("rule %s: `%s` in a python template is not a bound argument" % (ctx.rule.id, n.id))

            def generic_visit(self, n):
                super().generic_visit(n)
                if hasattr(n, "lineno"):
                    ast.copy_location(n, ctx.node)
                return n
        new = Sub().visit(tree)
        if ctx.rule.kind == "unpack":
            ctx.tr._tuple_assign(ctx.tgt, new, ctx.stmt)
            return ("", None, 100)
        return ctx.tr.expr(new)

    @staticmethod
    def _type(v):
        return v.t if isinstance(v, View) else v

    def _at(self, ctx, at):
        if at is None:
            return ctx.node
        if at == "targets":
            return ctx.tgt
        if at == "func":
            return ctx.node.func
        v = ctx.views[at]
        return v.node if v.node is not None else ctx.node

    def _stmts(self, ctx, c):
        tr = ctx.tr
        types = c.get("targets", [])
        names = []
        for tn, node, ty in zip(ctx.target_pats, ctx.target_nodes, types):
            tr.set_var(node.id, self._type(evaluate(ty, ctx)), ctx.stmt)
            names.append((tn, node.id))
        for tn, pyname in names:
            ctx.vars[tn] = tr.vname(pyname)
        tr.flush_pre(ctx.stmt)
        for s in c["stmts"]:
            tr.emit(render(s, ctx), ctx.stmt)
        return ("", None, 100)

    # ------------------------------------------------ reports
    def summary(self):
        counts = {}
        for r in self.rules:
            counts[r.exactness] = counts.get(r.exactness, 0) + 1
        return counts


# ---------------------------------------------------------------- functions and hooks


def _arr(ctx, rank, py="float", container="nd"):
    return ctx.engine.globals["T"]("arr", rank, py, container)


def _T(ctx, kind, rank=0, py=None, container=None):
    return ctx.engine.globals["T"](kind, rank, py, container)


def _num_of(ctx, a, b=None):
    t = Engine._type
    return ctx.engine.globals["num_of"](t(a), None if b is None else t(b))


def _dims_like(ctx, x):
    c = x.code
    if x.rank > 1:
        refuse_if_effectful(ctx, x)
    return ", ".join("shape(%s)[%d]" % (c, i) if i else "len(%s)" % c for i in range(x.rank))


def _paren(ctx, code, prec, need):
    return ctx.tr._paren(code, prec, need)


def _len(ctx, x):
    return len(x)


def _max(ctx, *xs):
    return max(xs)


def _same_shape(ctx, x, y):
    """Two arguments whose values have one shape by their form (py2viv's _shape_key): an
    operation of them needs no broadcasting."""
    k = ctx.tr._shape_key(x.node) if x.node is not None else None
    return k is not None and k == ctx.tr._shape_key(y.node)


def _vec(ctx, x):
    """An argument's code as an array: a number written as a vector of one entry, which
    broadcasts to any shape (NumPy broadcasts a number the same way)."""
    return x.code if x.t.is_arr() else "[%s]" % x.code


def _pure(ctx, *xs):
    """No argument's Viv code calls a def of the program: evaluating it twice, or once less,
    changes nothing (a def may print, or write into an array)."""
    tr = ctx.tr
    vname = ctx.engine.globals.get("viv_name", lambda n: n)
    for x in xs:
        code = x.code if isinstance(x, View) else str(x)
        for f in _effect_names(tr):
            if re.search(r"(?<![\w.])%s\(" % re.escape(vname(f)), code):
                return False
    return True


def _effect_names(tr):
    """The Python names whose Viv call may have an effect: the program's defs, and the function
    parameters of the def being translated (py2viv.py's Translator._effect_names)."""
    names = getattr(tr, "_effect_names", None)
    return names() if names is not None else list(getattr(tr, "funcs", {}))


def _all_int(ctx, *items):
    """Every passed argument (lists of arguments flattened) has Python type int."""
    flat = []
    for x in items:
        flat.extend(x if isinstance(x, list) else [x])
    return all(v.attr("py") == "int" for v in flat if v.attr("given"))


def _size(ctx, x):
    """One array size, as NumPy takes it: a negative size raises ValueError in NumPy and gives
    an empty array in Viv, so a size that is not a literal, len(...), numel(...) or shape(...)
    goes through py_size, which stops as NumPy does."""
    return _guard_size(ctx, x.code if isinstance(x, View) else x)


def _guard_size(ctx, code):
    if re.fullmatch(r"\d+", code) or re.match(r"(len|numel|shape)\(", code) and code.endswith(")") \
            and _balanced_call(code):
        return code
    return "%s(%s)" % (ctx.tr.use("py_size"), code)


def _balanced_call(code):
    """`len(x)` is one call (not `len(x) - k`)."""
    depth = 0
    for i, ch in enumerate(code):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(code) - 1:
                return False
    return True


def _sizes(ctx, shape):
    """A shape's dimensions, each through _guard_size, joined."""
    return ", ".join(_guard_size(ctx, c) for c in shape.attr("dims"))


def _py_of(ctx, *items):
    """NumPy's result dtype of an elementwise operation on these real operands: int when
    every one is int, bool when every one is Bool, float otherwise."""
    pys = [(v.attr("py") if v.attr("kind") != "bool" else "bool") for v in items]
    if all(p == "int" for p in pys):
        return "int"
    if all(p == "bool" for p in pys):
        return "bool"
    return "float"


def _axk(ctx, axis, a):
    """The axis a literal `axis=` names on array a, from 0 to rank - 1 (a negative one counts
    from the end), or -1 when no axis is named (None, or not passed)."""
    k = axis.attr("axis_lit") if axis.attr("given") else None
    if k is None:
        return -1
    return k + a.attr("rank") if k < 0 else k


def _kd(ctx, keepdims):
    """1 for keepdims=True (a literal), else 0."""
    return 1 if keepdims.attr("given") and keepdims.attr("lit") else 0


def _elts(x):
    x.force()
    return x._elts


def _seq_n(ctx, x):
    """How many arrays a sequence argument holds."""
    return len(_elts(x))


def _seq_kinds(ctx, x):
    """The kinds of a sequence's entries, sorted (['arr'], ['arr', 'carr'], ['num'], ...)."""
    return sorted({t.kind for _, t, _ in _elts(x)})


def _seq_ranks(ctx, x):
    """The ranks of a sequence's entries, sorted."""
    return sorted({t.rank if t.is_arr() else 0 for _, t, _ in _elts(x)})


def _seq_py(ctx, x):
    """NumPy's dtype of the entries joined: bool when all are Bools, int when all are ints or
    Bools, mixed when one is a list mixing ints and floats, else float."""
    pys = {t.py for _, t, _ in _elts(x)}
    if "mixed" in pys:
        return "mixed"
    if pys == {"bool"}:
        return "bool"
    if pys <= {"int", "bool"}:
        return "int"
    return "float"


def _seq_codes(ctx, x, wrap=""):
    """The entries' Viv code, joined by commas, each inside wrap(...) when wrap is given."""
    codes = [c for c, _, _ in _elts(x)]
    if wrap:
        for h in wrap.split("."):
            ctx.tr.use(h)
        codes = ["%s(%s)" % (wrap.split(".")[0], c) for c in codes]
    return ", ".join(codes)


def _const_int(node):
    """The value of an int written with constants only (`42`, `2**40 + 3`, `-1`), or None."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_int(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        a, b = _const_int(node.left), _const_int(node.right)
        if a is None or b is None or isinstance(node.op, ast.Pow) and not 0 <= b <= 256:
            return None
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op)) if not isinstance(node.op, ast.Pow) else a ** b
    return None


def _u32_words(n):
    """_int_to_uint32_array: the 32-bit words of a non-negative int, lowest first ([0] for 0)."""
    out = [n & 0xFFFFFFFF]
    n >>= 32
    while n > 0:
        out.append(n & 0xFFFFFFFF)
        n >>= 32
    return out


def _seed_words(ctx, x):
    """SeedSequence's entropy words of a seed written with constants (an int, or a list or tuple
    of them: each one's words in turn, as _coerce_to_uint32_array concatenates them), as a Viv
    array literal; None when the seed is not such a constant or has a negative entry."""
    node = x.node
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    words = []
    for e in items:
        v = _const_int(e)
        if v is None or v < 0:
            return None
        words.extend(_u32_words(v))
    if not words:
        return None
    return "[%s]" % ", ".join(str(w) for w in words)


FUNCS = {"seed_words": _seed_words, "seq_n": _seq_n, "seq_kinds": _seq_kinds, "seq_ranks": _seq_ranks, "seq_py": _seq_py,
         "seq_codes": _seq_codes, "axk": _axk, "kd": _kd, "arr": _arr, "T": _T, "num_of": _num_of, "dims_like": _dims_like, "paren": _paren,
         "len": _len, "all_int": _all_int, "size": _size, "sizes": _sizes, "py_of": _py_of, "max": _max, "pure": _pure, "vec": _vec,
         "same_shape": _same_shape}


def hook_reduce_gen(ctx, arg, fn):
    """sum/all/any over a generator expression: an accumulator loop hoisted before the
    statement, with the generator's own `for` and `if` clauses. Needs code: it writes
    statements and nested loops, which a template cannot."""
    return ctx.tr._reduce_gen(ctx.views[arg].node, fn)


def hook_fold_minmax(ctx):
    """min(a, b, c, ...) over numbers: folded pairwise, refused at the first array or at an int
    beside a float. Needs code: a template has a fixed number of arguments."""
    tr = ctx.tr
    name = ctx.name
    codes = [tr.expr(x) for x in ctx.node.args]
    code = codes[0][0]
    t = codes[0][1]
    num_of = ctx.engine.globals["num_of"]
    for (c, tt, _), node in zip(codes, ctx.node.args):
        if tt.kind in ("bool", "cnum", "rec"):
            tr.refuse(node, "builtin %s of %s" % (name, tr._kind_words(tt)),
                      "Python returns the winning value itself (a Bool, or TypeError for complex numbers); compare numbers.")
    for c, tt, _ in codes[1:]:
        if tt.is_arr():
            tr.refuse(ctx.node, "builtin %s over arrays" % name,
                      "Use np.maximum/np.minimum for elementwise, or %s(a) for the largest entry." % name)
        if t.py == "int" and tt.py == "int":
            code = "%s(%s, %s)" % (name, code, c)
            t = num_of(t, tt)
        else:
            code = "%s(%s, %s)" % (tr.use("py_b%s2" % name), code, c)
            # an int beside a float: Python returns whichever wins, as it is, so the result is
            # an int on some runs and a float on others (typed so: printing it is refused)
            t = num_of(t, tt) if (t.py == "int") == (tt.py == "int") else ctx.engine.globals["NUM_MIXED"]
    return code, t, 100


def _np_maxmin(tr, fn, x, tx, px, y, ty, py_):
    """np.maximum / np.minimum: a NaN in either operand gives NaN (Viv's max and min skip it,
    as fmax does); otherwise the same value, +0 over -0 for max and -0 under +0 for min (with
    comparisons: Viv's max and min of two zeros differ between engines). Two ints need none of
    it."""
    if tx.py == "int" and ty.py == "int" and not (tx.is_arr() or ty.is_arr()):
        return "%s(%s, %s)" % (fn, x, y)
    if tx.is_arr() and ty.is_arr():
        return "%s(%s, %s)" % (tr.use("py_np%sb" % fn), x, y)  # broadcast (helpers.toml)
    if tx.is_arr() or ty.is_arr():
        arr_, num_ = (x, y) if tx.is_arr() else (y, x)
        return "%s(%s, %s)" % (tr.use("py_np%ss" % fn), arr_, num_)
    return "%s(%s, %s)" % (tr.use("py_np%s2" % fn), x, y)


def hook_np_maxmin(ctx, fn, xn, yn):
    """np.maximum(x, y), np.minimum: NaN-propagating, +0 over -0. Needs code: the helper
    depends on which operands are arrays, and two ints need none."""
    x, y = ctx.views[xn], ctx.views[yn]
    code = _np_maxmin(ctx.tr, fn, x.code, x.t, x.prec, y.code, y.t, y.prec)
    if x.t.is_arr() or y.t.is_arr():
        rt = _arr(ctx, max(x.t.rank if x.t.is_arr() else 0, y.t.rank if y.t.is_arr() else 0), _py_of(ctx, x, y))
    else:
        rt = ctx.engine.globals["num_of"](x.t, y.t)
    return code, rt, 100


def hook_grow_append(ctx, an, xn):
    """lst.append(x) of a list py2viv grows: `lst = append(lst, x)`. Needs code: the list's
    element type (int or float, which Python prints differently) is the translator's state,
    set by the first append and checked by every later one, and the `[]` reads it."""
    return ctx.tr._grow_append(ctx.views[an].node, ctx.views[xn].node, ctx.views[xn].t, ctx.views[xn].code), None, 100


def hook_list_method(ctx, method, *names):
    """A method of a Python list (funcs.py, FuncsMixin._list_method): pop, insert, extend, remove,
    clear, reverse, sort with a key, index, count. Needs code: the list's element type and
    shape are the translator's state (as for append), and the method rebinds the list's name."""
    views = [ctx.views.get(n) for n in names]
    return ctx.tr._list_method(ctx, method, views)


def hook_str_join(ctx, sep, x):
    """sep.join(x) (strs.py): x a list of strings, a string, or a generator or comprehension of
    strings, built as a list first. Needs code: a generator is translated as a comprehension."""
    return ctx.tr._str_join(ctx.node, ctx.views[sep], ctx.views[x])


def hook_str_text(ctx, conv, x):
    """str(x), repr(x) (strs.py): the text of a value as Python writes it, by its type (an int's
    digits, a float's repr, True and False, a list as Python prints it). Needs code: the choice
    follows the static type in more ways than a template's cases say well."""
    v = ctx.views[x]
    return ctx.tr._str_of(v.node, v.code, v.t, v.prec, conv)


def hook_str_format(ctx, x, spec):
    """format(x, spec) (strs.py): the spec is a literal, read as Python reads it."""
    v = ctx.views[x]
    sv = ctx.views[spec]
    text = ""
    if sv.passed:
        n = sv.node
        if not (isinstance(n, ast.Constant) and isinstance(n.value, str)):
            ctx.tr.refuse(n if n is not None else ctx.node, "a format spec that is not a literal", "Write the spec literally.")
        text = n.value
    if isinstance(v.node, ast.Constant) and isinstance(v.node.value, str):
        return ctx.tr._str_lit(ctx.tr._const_field(v.node.value, None, text, v.node), v.node), ctx.engine.globals["STR"], 100
    return ctx.tr._str_format(v.node, v.code, v.t, text, None, v.prec)


def hook_strs_method(ctx, method, *names):
    """A method of a list of strings py2viv owns (strs.py): pop, insert, extend, remove, clear,
    reverse, sort; each rebinds the list's name."""
    views = [ctx.views.get(n) for n in names]
    return ctx.tr._strs_method(ctx, method, views)


def hook_strs_find(ctx, method, a, x):
    """lst.index(x), lst.count(x) of a list of strings (strs.py)."""
    return ctx.tr._strs_index_count(ctx, method, ctx.views[a], ctx.views[x])


HOOKS = {"reduce_gen": hook_reduce_gen, "fold_minmax": hook_fold_minmax, "np_maxmin": hook_np_maxmin,
         "grow_append": hook_grow_append, "list_method": hook_list_method, "str_join": hook_str_join,
         "str_text": hook_str_text, "str_format": hook_str_format, "strs_method": hook_strs_method,
         "strs_find": hook_strs_find}


# ---------------------------------------------------------------- command line


def _cell(text):
    return (text or "").replace("|", "\\|").replace("\n", " ")


def table(engine):
    """SUBSET.md's rule tables, generated: one section per kind of rule, a row per rule, in
    file order (check_rules.py keeps SUBSET.md equal to this)."""
    sections = [
        ("Strings: builtins and the methods of str and of a list of strings",
         lambda r: r.source == "strings.toml" and r.exactness != "refuse"),
        ("Builtins and the math module", lambda r: r.source == "builtins.toml" and r.exactness != "refuse"),
        ("NumPy functions", lambda r: r.source == "numpy.toml" and r.kind in ("call", "unpack", "item") and r.exactness != "refuse"),
        ("Methods and attributes of an array", lambda r: r.source != "strings.toml" and r.kind in ("method", "stmt", "attr")
         and r.exactness != "refuse"),
        ("Module constants", lambda r: r.kind == "const" and r.exactness != "refuse"),
        ("Operators on two arrays (broadcasting)", lambda r: r.kind == "binop" and r.exactness != "refuse"),
        ("Indexing", lambda r: r.kind in ("slice", "newaxis", "index", "index-store") and r.exactness != "refuse"),
    ]
    out = []
    for title, keep in sections:
        rows = [r for r in engine.rules if keep(r)]
        out.append("")
        out.append("#### %s (%d rules)" % (title, len(rows)))
        out.append("")
        out.append("| Python | exactness | notes | tests |")
        out.append("|---|---|---|---|")
        for r in rows:
            cls = r.exactness if r.exactness != "tolerance" else "within rtol %g, atol %g" % (r.rtol, r.atol)
            notes = r.evidence or ""
            if r.guard:
                notes += "; holds for: " + r.guard
            if r.note:
                notes += "; " + r.note
            out.append("| %s | %s | %s | %d |" % (", ".join("`%s`" % p.text for p in r.patterns), cls, _cell(notes),
                                                  len(r.tests)))
    rows = [r for r in engine.rules if r.exactness == "refuse"]
    out.append("")
    out.append("#### Refused (%d rules)" % len(rows))
    out.append("")
    out.append("| Python | refusal | hint | lifted by |")
    out.append("|---|---|---|---|")
    for r in rows:
        c = next((c for c in r.cases if "refuse" in c), {})
        out.append("| %s | %s | %s | %s |" % (", ".join("`%s`" % p.text for p in r.patterns), _cell(c.get("refuse")),
                                            _cell(c.get("hint")), _cell(r.gap or r.lift or "")))
    return "\n".join(out).strip("\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sys.path.insert(0, HERE)
    import py2viv
    eng = py2viv.RULES
    if "--table" in argv:
        print(table(eng))
        return 0
    counts = eng.summary()
    kinds = {}
    for r in eng.rules:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    print("%d rules (%s); by kind: %s; %d helpers; %d hooks (%s)" % (
        len(eng.rules), ", ".join("%s %d" % (k, counts.get(k, 0)) for k in EXACTNESS),
        ", ".join("%s %d" % kv for kv in sorted(kinds.items())), len(eng.helpers), len(eng.hooks),
        ", ".join(sorted(eng.hooks))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

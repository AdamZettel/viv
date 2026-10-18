# SPDX-License-Identifier: MIT OR Apache-2.0
"""Expression graph for the translator: hash-consed nodes, constant folding at
40 digits, and the Viv emitter.

The emitter keeps libxc's branch structure: a piecewise node becomes an
if/else statement, and an intermediate used only inside one branch is
computed inside it, so a branch that libxc does not take is never
evaluated (and its infinities never reach the gradient's tape)."""
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 40


class Unsupported(Exception):
    pass


class Node:
    __slots__ = ("id", "op", "args", "val", "exact", "hint", "name")

    def __repr__(self):
        return f"<{self.op}#{self.id}>"


_table = {}
_nodes = []


def _mk(op, args=(), val=None, exact=None, name=None):
    key = (op, tuple(a.id for a in args), None if val is None else mp.nstr(val, 40), exact, name)
    n = _table.get(key)
    if n is not None:
        return n
    n = Node()
    n.id, n.op, n.args, n.val, n.exact, n.hint, n.name = len(_nodes), op, tuple(args), val, exact, None, name
    _nodes.append(n)
    _table[key] = n
    return n


# Maple simplifies 1 + zeta to 2 r0 / (r0 + r1) (and 1 - zeta to 2 r1 / (r0 + r1));
# libxc's threshold tests (1 + z <= zeta_threshold) are written on that form, so
# the translator mirrors it: set_zeta registers the zeta node and its two forms.
ZETA = {}


def set_zeta(z, opz, omz):
    ZETA.clear()
    ZETA.update(z=z.id, opz=opz, omz=omz)


SIMPLIFY = [0]   # > 0 inside a Maple simplify(...): only there does Maple rewrite 1 + zeta


def _zeta_rewrite(op, a, b):
    if not ZETA or not SIMPLIFY[0]:
        return None
    if op == "+" and is_c(a) and a.val == 1:
        a, b = b, a
    if not (is_c(b) and b.val == 1):
        return None
    if op == "+":
        if a.id == ZETA["z"]:
            return ZETA["opz"]
        if a.op == "neg" and a.args[0].id == ZETA["z"]:
            return ZETA["omz"]
    if op == "-" and False:
        return None
    return None


def rebuild(n, memo):
    """n rebuilt bottom-up through the constructors (so SIMPLIFY rewrites apply)."""
    if n.id in memo:
        return memo[n.id]
    if n.op in ("const", "var", "bool"):
        return n
    a = [rebuild(x, memo) for x in n.args]
    op = n.op
    if op in ("add", "sub", "mul", "div", "pow"):
        out = binop({"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[op], a[0], a[1])
    elif op == "cmp":
        out = binop(n.name, a[0], a[1])
    elif op == "neg":
        out = neg(a[0])
    elif op == "not":
        out = lnot(a[0])
    elif op in ("and", "or"):
        out = logic(op, a[0], a[1])
    elif op == "func":
        out = func(n.name, a[0])
    elif op == "pw":
        out = piecewise(a)
    else:
        raise Unsupported(f"rebuild {op}")
    if out.hint is None and n.hint is not None and out.op not in ("const", "var", "bool"):
        out.hint = n.hint
    memo[n.id] = out
    return out


def simplify(n):
    """Maple's simplify(e) acts on all of e, the arguments a call put into it
    included; the one rewrite libxc's thresholds see is 1 +- zeta."""
    if not ZETA or not isinstance(n, Node):
        return n
    SIMPLIFY[0] += 1
    try:
        return rebuild(n, {})
    finally:
        SIMPLIFY[0] -= 1


def reset():
    ZETA.clear()
    _table.clear()
    _nodes.clear()
    global PI, TRUE, FALSE
    PI = _mk("const", val=+mp.pi)
    TRUE = _mk("bool", exact=1)
    FALSE = _mk("bool", exact=0)


def const(x):
    if isinstance(x, Fraction):
        return _mk("const", val=mp.mpf(x.numerator) / x.denominator, exact=x)
    if isinstance(x, int):
        return const(Fraction(x))
    return _mk("const", val=mp.mpf(x))


def const_float(s):
    """A Maple float literal: Maple keeps its decimal digits."""
    return _mk("const", val=mp.mpf(s))


def var(name):
    return _mk("var", name=name)


def hint(n, name):
    if isinstance(n, Node) and n.hint is None and n.op not in ("const", "var", "bool") and not name.startswith("<"):
        n.hint = name


def is_c(n):
    return isinstance(n, Node) and n.op == "const"


def cval(n, v):
    return is_c(n) and n.val == v


def boolc(b):
    return TRUE if b else FALSE


def _need(n):
    if not isinstance(n, Node):
        raise Unsupported(f"not an expression: {type(n).__name__}")
    return n


def neg(a):
    a = _need(a)
    if is_c(a):
        return _mk("const", val=-a.val, exact=None if a.exact is None else -a.exact)
    if a.op == "neg":
        return a.args[0]
    return _mk("neg", (a,))


def binop(op, a, b):
    a, b = _need(a), _need(b)
    if op in ("<", "<=", ">", ">=", "=", "<>"):
        if is_c(a) and is_c(b):
            x, y = a.val, b.val
            return boolc({"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y, "=": x == y, "<>": x != y}[op])
        return _mk("cmp", (a, b), name=op)
    if is_c(a) and is_c(b):
        ea, eb = a.exact, b.exact
        if op == "+":
            return _mk("const", val=a.val + b.val, exact=None if ea is None or eb is None else ea + eb)
        if op == "-":
            return _mk("const", val=a.val - b.val, exact=None if ea is None or eb is None else ea - eb)
        if op == "*":
            return _mk("const", val=a.val * b.val, exact=None if ea is None or eb is None else ea * eb)
        if op == "/":
            if b.val == 0:
                raise Unsupported("constant division by zero")
            return _mk("const", val=a.val / b.val, exact=None if ea is None or eb is None else ea / eb)
        if op == "^":
            if ea is not None and eb is not None and eb.denominator == 1 and abs(eb) < 64 and (ea != 0 or eb > 0):
                e = ea ** int(eb)
                return _mk("const", val=mp.mpf(e.numerator) / e.denominator, exact=e)
            if a.val < 0 and (eb is None or eb.denominator != 1):
                raise Unsupported("negative constant to a fractional power")
            return _mk("const", val=mp.power(a.val, b.val))
    if op in ("+", "-"):
        if op == "-" and is_c(a) and a.val == 1 and ZETA and SIMPLIFY[0]:
            if b.id == ZETA["z"]:
                return ZETA["omz"]
            if b.op == "neg" and b.args[0].id == ZETA["z"]:
                return ZETA["opz"]
        r = _zeta_rewrite(op, a, b)
        if r is not None:
            return r
    if op == "+":
        if cval(a, 0):
            return b
        if cval(b, 0):
            return a
        if b.op == "neg":
            return _mk("sub", (a, b.args[0]))
        return _mk("add", (a, b))
    if op == "-":
        if cval(b, 0):
            return a
        if cval(a, 0):
            return neg(b)
        if b.op == "neg":
            return _mk("add", (a, b.args[0]))
        return _mk("sub", (a, b))
    if op == "*":
        if cval(a, 0) or cval(b, 0):
            return const(0)
        if cval(a, 1):
            return b
        if cval(b, 1):
            return a
        if cval(a, -1):
            return neg(b)
        if cval(b, -1):
            return neg(a)
        return _mk("mul", (a, b))
    if op == "/":
        if cval(b, 1):
            return a
        if cval(a, 0):
            return const(0)
        return _mk("div", (a, b))
    if op == "^":
        if cval(b, 1):
            return a
        if cval(b, 0):
            return const(1)
        if cval(a, 1):
            return const(1)
        return _mk("pow", (a, b))
    raise Unsupported(f"operator {op}")


def lnot(a):
    a = _need(a)
    if a.op == "bool":
        return boolc(not a.exact)
    return _mk("not", (a,))


def logic(op, a, b):
    a, b = _need(a), _need(b)
    if a.op == "bool":
        if op == "or":
            return TRUE if a.exact else b
        return b if a.exact else FALSE
    if b.op == "bool":
        if op == "or":
            return TRUE if b.exact else a
        return a if b.exact else FALSE
    return _mk(op, (a, b))


MPF = {"exp": mp.exp, "log": mp.log, "sqrt": mp.sqrt, "sin": mp.sin, "cos": mp.cos, "tan": mp.tan,
       "sinh": mp.sinh, "cosh": mp.cosh, "tanh": mp.tanh, "atan": mp.atan, "asin": mp.asin, "acos": mp.acos,
       "erf": mp.erf, "abs": abs, "asinh": mp.asinh, "atanh": mp.atanh, "erfc": mp.erfc,
       "lambertw": lambda x: mp.re(mp.lambertw(x)), "sech": mp.sech, "cbrt": mp.cbrt,
       "e1_scaled": lambda x: mp.exp(x) * mp.e1(x), "erfcx": lambda x: mp.exp(x * x) * mp.erfc(x),
       "signum": mp.sign, "heaviside": lambda x: mp.mpf(1) if x >= 0 else mp.mpf(0)}


def func(name, a):
    a = _need(a)
    if is_c(a):
        if name in ("sqrt", "log") and a.val < 0:
            raise Unsupported(f"{name} of a negative constant")
        if name == "abs" and a.exact is not None:
            return const(abs(a.exact))
        return _mk("const", val=mp.mpf(MPF[name](a.val)))
    return _mk("func", (a,), name=name)


def piecewise(vals):
    """Maple piecewise(c1, a1, c2, a2, ..., [else]) and my_piecewise3/5."""
    if len(vals) % 2 == 0:
        vals = list(vals) + [const(0)]
    if len(vals) == 1:
        return vals[0]
    c, a, rest = _need(vals[0]), vals[1], vals[2:]
    if c.op == "bool":
        return a if c.exact else piecewise(rest)
    b = piecewise(rest)
    if isinstance(a, list) or isinstance(b, list):
        raise Unsupported("piecewise over lists")
    a, b = _need(a), _need(b)
    if a is b:
        return a
    return _mk("pw", (c, a, b))


reset()


# ---------------------------------------------------------------- emitting Viv

VIV_FUNCS = {"exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asin", "acos", "erf", "abs"}
CORE_FUNCS = {"asinh": "xc_asinh", "atanh": "xc_atanh"}
PREC = {"or": 1, "and": 2, "not": 3, "cmp": 4, "add": 5, "sub": 5, "mul": 6, "div": 6, "neg": 7, "pow": 8}


def num_lit(n):
    """A constant as a Viv literal: small rationals stay readable as p / q."""
    e = n.exact
    if e is not None and e.denominator == 1 and abs(e.numerator) < 10**15:
        s = f"{e.numerator}.0"
        return (s, 9) if e >= 0 else (s, 7)
    if e is not None and e.denominator <= 64 and abs(e.numerator) < 10**6:
        s = f"{e.numerator}.0 / {e.denominator}.0"
        return s, 6
    x = float(n.val)
    s = repr(x)
    if "e" in s and "." not in s.split("e")[0]:
        m, ex = s.split("e")
        s = f"{m}.0e{ex}"
    elif "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    if "inf" in s or "nan" in s:
        raise Unsupported("non-finite constant")
    return s, (7 if x < 0 else 9)


def ident(s):
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s).strip("_").lower()
    if not out or out[0].isdigit():
        out = "v_" + out
    return out


VIV_RESERVED = {"if", "else", "for", "while", "def", "return", "and", "or", "not", "in", "to", "from", "by", "test",
                "show", "import", "true", "false", "break", "continue", "pi", "e", "fn", "match", "enum", "rung",
                "step", "take", "using", "with", "let", "print", "len", "zeros", "sum", "max", "min", "abs", "exp",
                "log", "sqrt", "x", "gamma", "beta", "type", "record", "phrase", "assert", "stop", "where", "is",
                "of", "at", "as", "end", "hold", "holds", "deriv", "grad", "range", "parallel", "mod"}


class Emitter:
    """Straight-line Viv for a set of roots, with branch-local intermediates."""

    def __init__(self, inputs):
        self.inputs = inputs          # var name -> Viv expression
        self.lines = []
        self.used = set(VIV_RESERVED) | set(inputs.values())
        self.counts = {}

    def refcount(self, roots):
        seen = set()
        stack = list(roots)
        while stack:
            n = stack.pop()
            for a in n.args:
                self.counts[a.id] = self.counts.get(a.id, 0) + 1
                if a.id not in seen:
                    seen.add(a.id)
                    stack.append(a)

    def fresh(self, base):
        base = ident(base) or "t"
        if base in self.used:
            k = 1
            while f"{base}_{k}" in self.used:
                k += 1
            base = f"{base}_{k}"
        self.used.add(base)
        return base

    def named(self, n):
        if n.op in ("const", "var", "bool"):
            return False
        if n.op == "pw":
            return True
        if n.op in ("cmp", "and", "or", "not"):
            return False
        if n.hint is not None:
            return True
        if self.counts.get(n.id, 0) > 1:
            return n.op != "neg"
        return False

    def emit(self, n, scope, out, ind):
        """Returns (text, precedence); appends any statements to `out`."""
        if n.id in scope:
            return scope[n.id], 9
        if n.op == "const":
            return num_lit(n)
        if n.op == "var":
            return self.inputs[n.name], 9
        if n.op == "bool":
            return ("true" if n.exact else "false"), 9
        if n.op == "pw":
            c, a, b = n.args
            cs, _ = self.emit(c, scope, out, ind)
            name = self.fresh(n.hint or "branch")
            out.append(f"{ind}if {cs} {{")
            for arm, last in ((a, False), (b, True)):
                inner = dict(scope)
                body = []
                s, _ = self.emit(arm, inner, body, ind + "    ")
                out.extend(body)
                out.append(f"{ind}    {name} = {s}")
                out.append(f"{ind}}}" if last else f"{ind}}}\n{ind}else {{")
            scope[n.id] = name
            return name, 9
        parts = [self.emit(a, scope, out, ind) for a in n.args]
        text, prec = self.fmt(n, parts)
        if self.named(n):
            name = self.fresh(n.hint or "t")
            out.append(f"{ind}{name} = {text}")
            scope[n.id] = name
            return name, 9
        return text, prec

    @staticmethod
    def wrap(p, need, strict=False):
        s, q = p
        if q < need or (strict and q == need):
            return f"({s})"
        return s

    def fmt(self, n, parts):
        op = n.op
        w = self.wrap
        if op == "add":
            return f"{w(parts[0], 5)} + {w(parts[1], 5)}", 5
        if op == "sub":
            return f"{w(parts[0], 5)} - {w(parts[1], 5, True)}", 5
        if op == "mul":
            return f"{w(parts[0], 6)} * {w(parts[1], 6)}", 6
        if op == "div":
            return f"{w(parts[0], 6)} / {w(parts[1], 6, True)}", 6
        if op == "neg":
            return f"-{w(parts[0], 8)}", 7
        if op == "pow":
            return f"{w(parts[0], 9)} ^ {w(parts[1], 9)}", 8
        if op == "cmp":
            o = {"=": "==", "<>": "!="}.get(n.name, n.name)
            return f"{w(parts[0], 5)} {o} {w(parts[1], 5)}", 4
        if op in ("and", "or"):
            p = PREC[op]
            return f"{w(parts[0], p)} {op} {w(parts[1], p)}", p
        if op == "not":
            return f"not {w(parts[0], 4)}", 3
        if op == "func":
            f = n.name
            if f in VIV_FUNCS:
                return f"{f}({parts[0][0]})", 9
            if f in CORE_FUNCS:
                self.core_used.add(CORE_FUNCS[f])
                return f"{CORE_FUNCS[f]}({parts[0][0]})", 9
            if f == "cbrt":
                return f"{w(parts[0], 9)} ^ (1.0 / 3.0)", 8
            if f == "sech":
                return f"1.0 / cosh({parts[0][0]})", 6
            raise Unsupported(f"special function {f} (no Viv builtin with an adjoint)")
        raise Unsupported(f"node {op}")

    def body(self, root, ind="    "):
        self.core_used = set()
        self.refcount([root])
        out = []
        s, _ = self.emit(root, {}, out, ind)
        return out, s

# SPDX-License-Identifier: MIT OR Apache-2.0
"""A reader and evaluator for the subset of Maple that libxc's maple/*.mpl use.

libxc writes each functional as Maple arrow functions over (rs, z, xt, xs0,
xs1). Maple evaluates a name when it is used, so a script can define `f`
before the helpers it calls and an included script's `params_a_*` can be
filled by the includer. The evaluator here does the same: assignments are
kept as syntax, and evaluating the final `mzk(r0, r1, ...)` call inlines
every arrow function into one expression graph (expr.py) over the libxc
input variables. Constants fold at 40 digits (Maple's maple2c uses 20)."""
import os
import re
from fractions import Fraction

import expr as E


Unsupported = E.Unsupported   # the script uses something this translator does not translate


# ---------------------------------------------------------------- preprocessing

def preprocess(path, incdirs, defines, seen=None):
    return preprocess_text(open(path, encoding="utf-8", errors="replace").read(), incdirs, defines)


def preprocess_text(text, incdirs, defines):
    """libxc's $include/$define/$ifdef/$elif/$else/$endif/$undef, in order."""
    out = []
    stack = []          # [active, any_taken]
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith("$"):
            parts = line.split(None, 1)
            d = parts[0]
            arg = parts[1].strip() if len(parts) > 1 else ""
            active = all(s[0] for s in stack)
            if d == "$ifdef":
                take = arg in defines
                stack.append([take, take])
            elif d == "$elif":
                outer = all(s[0] for s in stack[:-1])
                top = stack[-1]
                take = (not top[1]) and arg in defines
                top[0] = take and outer
                top[1] = top[1] or take
            elif d == "$else":
                top = stack[-1]
                top[0] = not top[1]
                top[1] = True
            elif d == "$endif":
                stack.pop()
            elif not active:
                pass
            elif d == "$define":
                defines.add(arg.split()[0])
            elif d == "$undef":
                defines.discard(arg.split()[0])
            elif d == "$include":
                name = arg.strip('"<> ')
                for inc in incdirs:
                    p = os.path.join(inc, name)
                    if os.path.exists(p):
                        out.append(preprocess(p, incdirs, defines))
                        break
                else:
                    raise Unsupported(f"$include {name} not found")
            else:
                raise Unsupported(f"preprocessor {d}")
            continue
        if all(s[0] for s in stack):
            out.append(raw)
    return "\n".join(out)


# ---------------------------------------------------------------- tokens

TOK = re.compile(r"""
  (?P<ws>\s+)
 |(?P<num>(?:\d+\.(?!\.)\d*|(?<!\.)\.\d+|\d+)(?:[eE][-+]?\d+)?)
 |(?P<name>[A-Za-z_][A-Za-z_0-9]*)
 |(?P<bq>`[^`]*`)
 |(?P<str>"[^"]*")
 |(?P<op>:=|->|<=|>=|<>|\.\.|\*\*|[-+*/^(),\[\]{}:;<>=!.~$|&@%])
""", re.X)


def strip_comments(text):
    out, i, depth = [], 0, 0
    n = len(text)
    while i < n:
        if text.startswith("(*", i):
            depth += 1
            i += 2
        elif depth and text.startswith("*)", i):
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif text[i] == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text[i] == '"':
            j = text.find('"', i + 1)
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def tokenize(text):
    toks = []
    pos = 0
    text = strip_comments(text).replace("\\\n", " ")
    while pos < len(text):
        m = TOK.match(text, pos)
        if not m:
            raise Unsupported(f"cannot read {text[pos:pos + 20]!r}")
        pos = m.end()
        k = m.lastgroup
        if k == "ws":
            continue
        toks.append((k, m.group(k)))
    toks.append(("eof", ""))
    return toks


# ---------------------------------------------------------------- syntax
# AST: ("num", str) ("name", s) ("str", s) ("call", f_ast, [args]) ("index", a, [i])
# ("bin", op, a, b) ("neg", a) ("not", a) ("arrow", [params], body) ("list", [items])
# ("range", a, b) ("proc", text)

KEYWORDS = {"and", "or", "not", "proc", "end", "if", "then", "else", "elif", "fi", "do", "od", "for", "from",
            "to", "by", "while", "local", "global", "option", "options", "description", "in", "mod", "use"}


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k]

    def take(self, val=None):
        tok = self.t[self.i]
        if val is not None and tok[1] != val:
            raise Unsupported(f"expected {val!r}, found {tok[1]!r}")
        self.i += 1
        return tok

    def at(self, val):
        return self.t[self.i][1] == val and self.t[self.i][0] in ("op", "name")

    def statements(self):
        out = []
        while self.peek()[0] != "eof":
            if self.at(":") or self.at(";"):
                self.take()
                continue
            start = self.i
            try:
                st = self.statement()
            except Unsupported as e:
                # skip to the statement's end: the name it assigns (if any) becomes unusable
                self.i = start
                target = self.peek()[1] if self.peek()[0] in ("name", "bq") else None
                self.skip_statement()
                st = ("bad", target, str(e))
            out.append(st)
        return out

    def skip_statement(self):
        depth = 0
        while self.peek()[0] != "eof":
            k, v = self.take()
            if v == "proc":
                depth += 1
            elif v == "end" and self.peek()[1] in ("proc",):
                self.take()
                depth -= 1
            elif depth == 0 and v in (":", ";") and k == "op":
                return

    def statement(self):
        k, v = self.peek()
        if k == "bq":
            raise Unsupported(f"backquoted name {v}")
        if v in ("if", "for", "while"):
            raise Unsupported(f"statement {v}")
        lhs = self.expr()
        if self.at(":="):
            self.take()
            rhs = self.expr()
            self.end_statement()
            if lhs[0] == "name":
                return ("assign", lhs[1], rhs)
            if lhs[0] == "index" and lhs[1][0] == "name":
                return ("assign_index", lhs[1][1], lhs[2], rhs)
            raise Unsupported("assignment target")
        self.end_statement()
        return ("expr", lhs)

    def end_statement(self):
        if self.at(":") or self.at(";"):
            self.take()
        elif self.peek()[0] != "eof":
            raise Unsupported(f"unexpected {self.peek()[1]!r}")

    def expr(self):
        return self.arrow()

    def arrow(self):
        # x -> e   or   (a, b) -> e
        k, v = self.peek()
        if k == "name" and v not in KEYWORDS and self.peek(1)[1] == "->":
            self.take()
            self.take("->")
            return ("arrow", [v], self.arrow())
        if v == "(":
            j = self.i + 1
            names = []
            ok = True
            while True:
                kk, vv = self.t[j]
                if kk == "name":
                    names.append(vv)
                    j += 1
                    if self.t[j][1] == ",":
                        j += 1
                        continue
                    if self.t[j][1] == ")":
                        j += 1
                        break
                elif vv == ")" and not names:
                    j += 1
                    break
                ok = False
                break
            if ok and self.t[j][1] == "->":
                self.i = j + 1
                return ("arrow", names, self.arrow())
        return self.lor()

    def lor(self):
        a = self.land()
        while self.at("or"):
            self.take()
            a = ("bin", "or", a, self.land())
        return a

    def land(self):
        a = self.lnot()
        while self.at("and"):
            self.take()
            a = ("bin", "and", a, self.lnot())
        return a

    def lnot(self):
        if self.at("not"):
            self.take()
            return ("not", self.lnot())
        return self.rel()

    def rel(self):
        a = self.rng()
        if self.peek()[0] == "op" and self.peek()[1] in ("<", "<=", ">", ">=", "=", "<>"):
            op = self.take()[1]
            a = ("bin", op, a, self.rng())
        return a

    def rng(self):
        a = self.add()
        if self.at(".."):
            self.take()
            a = ("range", a, self.add())
        return a

    def add(self):
        if self.at("-") or self.at("+"):
            op = self.take()[1]
            a = self.mul()
            a = ("neg", a) if op == "-" else a
        else:
            a = self.mul()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            a = ("bin", op, a, self.mul())
        return a

    def mul(self):
        a = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            a = ("bin", op, a, self.unary())
        return a

    def unary(self):
        if self.at("-"):
            self.take()
            return ("neg", self.unary())
        if self.at("+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        a = self.postfix()
        if self.at("^") or self.at("**"):
            self.take()
            if self.at("-"):
                self.take()
                b = ("neg", self.power())
            else:
                b = self.power()
            return ("bin", "^", a, b)
        return a

    def postfix(self):
        a = self.primary()
        while True:
            if self.at("("):
                self.take()
                args = self.args(")")
                a = ("call", a, args)
            elif self.at("["):
                self.take()
                a = ("index", a, self.args("]"))
            else:
                return a

    def args(self, close):
        out = []
        if self.at(close):
            self.take()
            return out
        while True:
            out.append(self.expr())
            if self.at(","):
                self.take()
                continue
            self.take(close)
            return out

    def primary(self):
        k, v = self.peek()
        if k == "num":
            self.take()
            return ("num", v)
        if k == "str":
            self.take()
            return ("str", v[1:-1])
        if k == "name":
            if v == "proc":
                raise Unsupported("proc")
            if v in KEYWORDS:
                raise Unsupported(f"keyword {v}")
            self.take()
            return ("name", v)
        if v == "(":
            self.take()
            e = self.expr()
            self.take(")")
            return e
        if v == "[":
            self.take()
            return ("list", self.args("]"))
        if v == "{":
            raise Unsupported("set")
        if k == "bq":
            raise Unsupported(f"backquoted name {v}")
        raise Unsupported(f"unexpected {v!r}")


# ---------------------------------------------------------------- evaluation

class Closure:
    def __init__(self, name, params, body):
        self.name, self.params, self.body = name, params, body


MATH1 = {"exp": "exp", "log": "log", "ln": "log", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
         "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arctan": "atan", "arcsin": "asin", "arccos": "acos",
         "erf": "erf", "abs": "abs", "arcsinh": "asinh", "arctanh": "atanh", "erfc": "erfc", "LambertW": "lambertw",
         "xc_E1_scaled": "e1_scaled", "xc_erfcx": "erfcx", "sech": "sech", "cbrt": "cbrt", "signum": "signum",
         "Heaviside": "heaviside", "xc_bessel_I0": "bessel_i0", "xc_bessel_I1": "bessel_i1"}


class Env:
    def __init__(self, statements, params, consts):
        """params: name -> value (number or list of numbers) for the functional's
        params_a_*; consts: p_a_* thresholds and other names maple2c leaves free."""
        self.defs = {}
        self.bad = {}
        for st in statements:
            if st[0] == "assign":
                self.defs[st[1]] = st[2]
                self.bad.pop(st[1], None)
            elif st[0] == "assign_index":
                # params_a_x[2] := v: keep as an override of one list entry
                self.defs.setdefault(("idx", st[1]), []).append((st[2], st[3]))
            elif st[0] == "bad" and st[1]:
                self.bad[st[1]] = st[2]
                self.defs.pop(st[1], None)
        self.params = params
        self.consts = consts
        self.cache = {}
        self.used_params = set()
        self.calls = {}
        self.rs = None      # (rs node, dens node, r0, r1, polarized): see set_density

    def set_density(self, rs, dens, r0, r1, polarized):
        """Maple's simplify turns n_spin(rs, z) back into r0 (or r1, or dens/2) and
        n_total(rs) into dens; libxc screens on those exact values."""
        self.rs = (rs, dens, r0, r1, polarized)

    def lookup(self, name):
        if name in self.cache:
            return self.cache[name]
        if name in self.defs:
            ast = self.defs[name]
            if ast[0] == "arrow":
                v = Closure(name, ast[1], ast[2])
            else:
                self.cache[name] = None       # cycle guard
                v = self.eval(ast, {})
                if ("idx", name) in self.defs and isinstance(v, list):
                    v = list(v)
                    for idx, rhs in self.defs[("idx", name)]:
                        k = self.int_of(self.eval(idx[0], {}))
                        v[k - 1] = self.eval(rhs, {})
            self.cache[name] = v
            return v
        if name in self.bad:
            raise Unsupported(f"{name}: {self.bad[name]}")
        if name.startswith("params_a_"):
            key = name[len("params_a_"):]
            m = re.fullmatch(r"(\w+?)_(\d+)_", key)     # params_a_CC_0_ is the C params->CC[0]
            if key not in self.params and m and isinstance(self.params.get(m.group(1)), list):
                rows = self.params[m.group(1)]
                k = int(m.group(2))
                if k < len(rows) and isinstance(rows[k], list):
                    self.used_params.add(m.group(1))
                    v = [E.const(x) for x in rows[k]]
                    self.cache[name] = v
                    return v
            if key in self.params:
                self.used_params.add(key)
                val = self.params[key]
                if isinstance(val, list) and val and isinstance(val[0], list):
                    raise Unsupported(f"{name} is a matrix")
                v = [E.const(x) for x in val] if isinstance(val, list) else E.const(val)
                self.cache[name] = v
                return v
            raise Unsupported(f"no value for {name}")
        if name in self.consts:
            v = E.const(self.consts[name])
            self.cache[name] = v
            return v
        if name == "Pi":
            return E.PI
        if name == "true":
            return E.TRUE
        if name == "false":
            return E.FALSE
        if name == "infinity":
            raise Unsupported("infinity")
        raise Unsupported(f"undefined name {name}")

    def int_of(self, v):
        if isinstance(v, E.Node) and v.op == "const" and v.exact is not None and v.exact.denominator == 1:
            return int(v.exact)
        raise Unsupported("index is not an integer constant")

    def eval(self, ast, local):
        k = ast[0]
        if k == "num":
            s = ast[1]
            if re.fullmatch(r"\d+", s):
                return E.const(Fraction(int(s)))
            return E.const_float(s)
        if k == "str":
            return ast[1]
        if k == "name":
            if ast[1] in local:
                return local[ast[1]]
            return self.lookup(ast[1])
        if k == "list":
            return [self.eval(a, local) for a in ast[1]]
        if k == "arrow":
            # an arrow inside an expression closes over the current locals
            c = Closure("<lambda>", ast[1], ast[2])
            c.local = dict(local)
            return c
        if k == "neg":
            return E.neg(self.eval(ast[1], local))
        if k == "not":
            return E.lnot(self.eval(ast[1], local))
        if k == "bin":
            op = ast[1]
            if op == "or" or op == "and":
                a = self.eval(ast[2], local)
                b = self.eval(ast[3], local)
                return E.logic(op, a, b)
            a = self.eval(ast[2], local)
            b = self.eval(ast[3], local)
            if op in ("+", "-", "*", "/") and (isinstance(a, list) or isinstance(b, list)):
                raise Unsupported("list arithmetic")
            return E.binop(op, a, b)
        if k == "index":
            a = self.eval(ast[1], local)
            idx = [self.eval(i, local) for i in ast[2]]
            if not isinstance(a, list) or len(idx) != 1:
                raise Unsupported("indexing a non-list")
            i = self.int_of(idx[0])
            if not (1 <= i <= len(a)):
                raise Unsupported("index out of range")
            return a[i - 1]
        if k == "call":
            return self.call(ast, local)
        if k == "range":
            raise Unsupported("range outside add")
        raise Unsupported(f"syntax {k}")

    def call(self, ast, local):
        f = ast[1]
        args = ast[2]
        if f[0] == "name" and f[1] not in local:
            n = f[1]
            if n in ("add", "sum", "mul", "seq"):
                return self.reduce(n, args, local)
            if n in ("simplify", "evalf", "eval", "expand", "combine", "factor", "normal") and n not in self.defs:
                if n == "eval" and len(args) > 1:
                    raise Unsupported("eval with substitution")
                if n != "simplify":
                    return self.eval(args[0], local)
                return E.simplify(self.eval(args[0], local))
            if n in ("my_piecewise3", "my_piecewise5", "piecewise") and n not in self.defs:
                vals = [self.eval(a, local) for a in args]
                return E.piecewise(vals)
            if n in MATH1 and n not in self.defs:
                if len(args) != 1:
                    if n == "log" or n == "LambertW":
                        raise Unsupported(f"{n} with {len(args)} arguments")
                    raise Unsupported(f"{n} arity")
                return E.func(MATH1[n], self.eval(args[0], local))
            if n in ("max", "min") and n not in self.defs:
                vals = [self.eval(a, local) for a in args]
                out = vals[0]
                for v in vals[1:]:
                    out = E.piecewise([E.binop(">" if n == "max" else "<", out, v), out, v])
                return out
            if n in ("BesselK", "BesselI", "GAMMA", "Zeta", "dilog", "polylog", "Ei", "Ci", "Si", "series",
                     "convert", "taylor", "int", "diff", "limit", "solve", "fsolve", "printf", "assert"):
                if n not in self.defs:
                    raise Unsupported(f"Maple function {n}")
        fn = self.eval(f, local)
        if not isinstance(fn, Closure):
            raise Unsupported(f"calling a non-function {f[1] if f[0] == 'name' else f[0]}")
        vals = [self.eval(a, local) for a in args]
        return self.apply(fn, vals)

    def apply(self, fn, vals):
        if self.rs is not None and fn.name in ("n_spin", "n_total") and vals and vals[0] is self.rs[0]:
            rs, dens, r0, r1, pol = self.rs
            if fn.name == "n_total":
                return dens
            z = vals[1]
            if not pol and E.is_c(z) and z.val == 0:
                return E.binop("/", dens, E.const(2))
            if pol and E.ZETA and z.id == E.ZETA["z"]:
                return r0
            if pol and E.ZETA and z.op == "neg" and z.args[0].id == E.ZETA["z"]:
                return r1
        if len(vals) != len(fn.params):
            raise Unsupported(f"{fn.name}: {len(vals)} arguments for {len(fn.params)} parameters")
        key = None
        if fn.name != "<lambda>" and all(isinstance(v, E.Node) for v in vals):
            key = (fn.name, tuple(v.id for v in vals))
            if key in self.calls:
                return self.calls[key]
        loc = dict(getattr(fn, "local", {}))
        loc.update(zip(fn.params, vals))
        out = self.eval(fn.body, loc)
        if isinstance(out, E.Node) and fn.name != "<lambda>":
            E.hint(out, fn.name)
        if key is not None:
            self.calls[key] = out
        return out

    def reduce(self, n, args, local):
        if len(args) != 2 or args[1][0] != "bin" or args[1][1] != "=" or args[1][2][0] != "name" \
                or args[1][3][0] != "range":
            raise Unsupported(f"{n} form")
        var = args[1][2][1]
        lo = self.int_of(self.eval(args[1][3][1], local))
        hi = self.int_of(self.eval(args[1][3][2], local))
        items = []
        for i in range(lo, hi + 1):
            loc = dict(local)
            loc[var] = E.const(Fraction(i))
            items.append(self.eval(args[0], loc))
        if n == "seq":
            return items
        if not items:
            return E.const(Fraction(0 if n in ("add", "sum") else 1))
        out = items[0]
        for it in items[1:]:
            out = E.binop("+" if n in ("add", "sum") else "*", out, it)
        return out


def load_text(text, incdirs, defines=()):
    text = preprocess_text(text, incdirs, set(defines))
    return Parser(tokenize(text)).statements()

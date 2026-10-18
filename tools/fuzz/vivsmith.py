#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivsmith: random, well-typed, terminating Viv programs, Csmith-style.

    python3 tools/fuzz/vivsmith.py SEED [--size N]     the program, on stdout
    python3 tools/fuzz/vivsmith.py SEED --cover        the constructs it used

Every program is valid by construction and has one meaning, so the three
engines (interpreter, C, wasm) must print the same bytes (tools/fuzz/run.py
compares them):

- Values stay bounded. Each Int variable and Int array element holds at most
  IB in magnitude and each number at most NB; the generator carries a bound for
  every expression it builds (interval arithmetic on magnitudes) and reduces a
  value back into range (`% M`, `clip`, `tanh`) before it is stored, so no Int
  operation overflows and no number reaches inf or nan.
- Divisors are never zero: an Int divisor is `(e % 7 + 1)` or `(e % 7 - 9)`
  (Viv's `%` is Python's floor modulo), a number divisor `(abs(e) + 1.0)`.
- Indices are in range: a loop index over `len(a)`, a literal below the
  length, or `(e % len(a))`; slices have literal bounds inside the array.
- Loops are bounded: counted loops over literals, Int constants or `len`,
  `while` loops with a counter, recursion with a depth below 12.
- Arrays alias on assignment and as parameters (`viv features
  aliasing-and-copies`); the generator keeps lengths fixed, so aliasing never
  changes what is in range, and the engines must agree on what it changes.

The program ends by printing every top-level value with `fmt`, the shortest
digits that read back as the same double, so any differing bit shows.

Growth (from a second random stream, so a seed's other statements are what
they were): defs that append to an [Int], an Arr or a record's [Int] and [Str]
fields they were passed and hand them back, called up to 200 times (B340);
records of lists made with empty `[]` (B341); variables named after C
keywords and macros (B345); slice bounds that are Ints (B342); and an [Int]
joined by `extend`, then summed, sliced and sorted (B343).

Not generated yet: strings beyond a list of names, maps, 2-D arrays and
matrix builtins (`@`, solves, axis reductions), complex numbers, enums and
match, optionals, lambdas and closures, grad and the other autodiff forms,
phrases, arrays of records and nested records, imports, test blocks and
holds, and programs meant to stop with a run-time error (index out of range,
Int overflow), whose message the engines should also agree on. One open
front-end bug shapes what is generated: an if-expression whose branches are
both literals, len(...) or int(...) gets an Int zero added (B111).
"""
import random
import sys
from collections import Counter

IB = 2 ** 20            # an Int variable's bound
IMOD = 1048573          # `% IMOD` brings an Int back into [0, IB)
ILIM = 2 ** 60          # the largest bound an Int expression may reach
NB = 1.0e6              # a number variable's bound
NLIM = 1.0e15           # the largest bound a number expression may reach
# numbers at the edges of the format and of the libm, each inside something that
# keeps the result bounded: negative zero, subnormals, ties for round, huge
# arguments for argument reduction, results that underflow
NUM_EDGES = [("(-0.0)", 1e-300), ("5.0e-324", 1e-300), ("1.0e-310", 1e-300), ("2.5", 2.5), ("(-2.5)", 2.5),
             ("(-0.5)", 0.5), ("0.49999999999999994", 1.0), ("tanh(1.0e300)", 1.0), ("atan(1.0e300)", 2.0),
             ("sin(1.0e22)", 1.0), ("cos(1.0e15)", 1.0), ("exp(-745.0)", 1e-300), ("exp(-708.5)", 1e-300),
             ("log(1.0e-310)", 714.0), ("(1.0e300 % 7.0)", 7.0), ("sqrt(1.0e-310)", 1e-150), ("(0.1 + 0.2)", 1.0),
             ("(-1.0e-300 * 1.0e-300)", 1e-300), ("atan2((-0.0), (-1.0))", 4.0), ("round((-0.4))", 1.0),
             ("ceil((-0.5))", 1.0), ("(9007199254740993.0 - 9007199254740992.0)", 2.0)]
# Ints at the edges of 64 bits, used only under // and % by a small divisor or in wrap_*
INT_EDGES = ["4611686018427387904", "(-4611686018427387904)", "9223372036854775807", "(-9223372036854775807)",
             "9007199254740993", "(-9007199254740993)"]
# names that are keywords or macros in C (C89 to C23, GCC and Clang
# extensions, predefined macros), each a plain variable in Viv (B345)
C_NAMES = ["asm", "auto", "register", "volatile", "restrict", "inline", "typedef", "extern", "static", "union",
           "struct", "unsigned", "signed", "short", "long", "char", "goto", "switch", "sizeof", "const", "do",
           "void", "typeof", "alignas", "alignof", "constexpr", "nullptr", "thread_local", "static_assert",
           "noreturn", "fortran", "unix", "linux", "i386", "_Bool", "_Complex", "_Atomic", "_Generic", "_Noreturn",
           "_Thread_local", "y0", "j1", "index", "read", "write", "select", "signal"]
# the most times a growth loop calls its def: past the 64 elements where B340 trapped
GROW_MAX = 200
F32_LITS = ["0.0", "1.0", "-1.0", "0.5", "2.0", "0.1", "3.0", "1.0e-3", "250.0", "-0.75", "1.0e30", "1.0e-40", "65504.0", "3.4028235e38"]
NUM_LITS = ["0.0", "1.0", "-1.0", "0.5", "2.0", "-2.5", "3.0", "0.1", "1.0e-3", "250.0", "-0.75", "1.5e2",
            "0.3333333333333333", "7.25", "-1.0e-5", "1024.0"]


class Var:
    def __init__(self, name, ty, mutable=True):
        self.name, self.ty, self.mutable = name, ty, mutable


class Def:
    def __init__(self, name, params, ret, recursive=False):
        self.name, self.params, self.ret, self.recursive = name, params, ret, recursive


def tyname(ty):
    if ty == "N":
        return "Num"
    if ty == "I":
        return "Int"
    if ty[0] == "A":
        return "Arr"
    if ty[0] == "IA":
        return "[Int]"
    return ty[1]


class Gen:
    def __init__(self, seed, size=1.0):
        self.R = random.Random(seed)
        self.seed = seed
        self.size = size
        self.out = []
        self.cover = Counter()
        self.n = 0
        self.defs = []
        self.records = {}       # name -> [(field, ty)]
        self.scopes = [[]]
        self.ind = 0
        self.loop_depth = 0
        self.in_def = None
        self.budget = int(60 * size)   # statements left
        self.call_depth = 0
        self.weak = set()
        self.zero = "z0"
        # the growth constructs draw from a stream of their own
        self.G = random.Random(seed * 7919 + 17)
        # the Float32 section's own stream (float32s)
        self.F = random.Random(seed * 104729 + 32)
        self.c_names = list(C_NAMES)
        self.G.shuffle(self.c_names)
        self.grow = False

    # ------------------------------------------------------------ helpers
    def fresh(self, p="v"):
        self.n += 1
        if p == "v" and self.c_names and self.G.random() < 0.05:
            self.c("c-keyword-name")
            return self.c_names.pop()
        return f"{p}{self.n}"

    def ibound(self, k):
        """The slice bound k, sometimes as an Int expression (B342)."""
        if self.G.random() < 0.3:
            self.c("slice-int-bound")
            return f"({k} + {self.zero})"
        return str(k)

    def emit(self, s):
        self.out.append("    " * self.ind + s)

    def c(self, tag):
        self.cover[tag] += 1

    def vars(self, pred=lambda v: True):
        return [v for sc in self.scopes for v in sc if pred(v)]

    def declare(self, v):
        self.scopes[-1].append(v)

    def pick(self, xs):
        return self.R.choice(xs)

    def chance(self, p):
        return self.R.random() < p

    def arr_len(self):
        return self.R.randint(1, 6)

    # ------------------------------------------------------------ Int expressions
    def int_leaf(self):
        R = self.R
        opts = ["lit", "lit"]
        ivars = self.vars(lambda v: v.ty == "I")
        if ivars:
            opts += ["var"] * 4
        arrs = self.vars(lambda v: v.ty[0] in ("A", "IA"))
        if arrs:
            opts.append("len")
        iarrs = self.vars(lambda v: v.ty[0] == "IA")
        if iarrs:
            opts += ["ielem"] * 2
        recs = [(v, f) for v in self.vars(lambda v: v.ty[0] == "R") for f, t in self.records[v.ty[1]] if t == "I"]
        if recs:
            opts.append("field")
        if self.callable("I"):
            opts.append("call")
        k = self.pick(opts)
        if k == "var":
            v = self.pick(ivars)
            self.c("int-var")
            return v.name, IB
        if k == "len":
            v = self.pick(arrs)
            self.c("len")
            e = f"len({v.name})"
            self.weak.add(e)
            return e, v.ty[1]
        if k == "ielem":
            v = self.pick(iarrs)
            return f"{v.name}[{self.index(v)}]", IB
        if k == "field":
            v, f = self.pick(recs)
            self.c("record-field-read")
            return f"{v.name}.{f}", IB
        if k == "call":
            return self.call("I")
        self.c("int-literal")
        x = self.pick([R.randint(-9, 9), R.randint(-100, 100), R.randint(-IB + 1, IB - 1)])
        e = f"({x})" if x < 0 else str(x)
        self.weak.add(e)
        return e, abs(x)

    def int_expr(self, d=None):
        if d is None:
            d = self.R.randint(0, 3)
        if d <= 0 or self.chance(0.2):
            return self.int_leaf()
        R = self.R
        k = self.pick(["+", "-", "*", "*", "//", "%", "neg", "abs", "minmax", "pow", "ifx", "wrap", "fromnum", "edge"])
        if k == "edge":
            big = self.pick(INT_EDGES)
            self.weak.add(big)
            if self.chance(0.5):
                b = self.strong(self.int_expr(d - 1)[0])
                self.c("int-edge-wrap")
                return f"({self.pick(['wrap_add', 'wrap_mul'])}({self.strong(big)}, {b}) % {IMOD})", IMOD
            b, _ = self.int_expr(d - 1)
            op = self.pick(["//", "%"])
            div = f"({b} % 7 + 1)" if self.chance(0.5) else f"({b} % 7 - 9)"
            self.c("int-edge-" + ("floordiv" if op == "//" else "mod"))
            e = self.weak_if(f"({big} {op} {div})", big, b)
            if op == "//":
                r = f"({e} % {IMOD})"
                if e in self.weak:
                    self.weak.add(r)
                return r, IMOD
            return e, 9
        if k in ("+", "-", "*"):
            a, ba = self.int_expr(d - 1)
            b, bb = self.int_expr(d - 1)
            bound = ba * bb if k == "*" else ba + bb
            if bound > ILIM:
                a, ba = f"({a} % 1021)", 1021
                b, bb = f"({b} % 1021)", 1021
                bound = ba * bb if k == "*" else ba + bb
            self.c("int-" + {"+": "add", "-": "sub", "*": "mul"}[k])
            return self.weak_if(f"({a} {k} {b})", a, b), bound
        if k in ("//", "%"):
            a, ba = self.int_expr(d - 1)
            b, _ = self.int_expr(d - 1)
            if self.chance(0.5):
                div, bd = f"({b} % 7 + 1)", 7
                self.c("int-divisor-positive")
            else:
                div, bd = f"({b} % 7 - 9)", 9
                self.c("int-divisor-negative")
            self.c("int-floordiv" if k == "//" else "int-mod")
            return self.weak_if(f"({a} {k} {div})", a, b), (ba + 1 if k == "//" else bd)
        if k == "neg":
            a, ba = self.int_expr(d - 1)
            self.c("int-neg")
            return self.weak_if(f"(-{a})", a), ba
        if k == "abs":
            a, ba = self.int_expr(d - 1)
            self.c("int-abs")
            return self.weak_if(f"abs({a})", a), ba
        if k == "minmax":
            a, ba = self.int_expr(d - 1)
            b, bb = self.int_expr(d - 1)
            f = self.pick(["min", "max"])
            self.c("int-" + f)
            return self.weak_if(f"{f}({a}, {b})", a, b), max(ba, bb)
        if k == "pow":
            ivars = self.vars(lambda v: v.ty == "I")
            if not ivars:
                return self.int_leaf()
            # `^` of literals (and len) is a number (viv features int-type), so the
            # base holds an Int variable
            a, ba = f"({self.pick(ivars).name} % 1000)", 1000
            e = R.randint(0, 3)
            self.c("int-pow")
            return f"({a} ^ {e})", max(1, ba ** e)
        if k == "ifx":
            cond = self.cond(d - 1)
            a, ba = self.int_expr(d - 1)
            b, bb = self.int_expr(d - 1)
            self.c("if-expression-int")
            if a in self.weak and b in self.weak:
                a = self.strong(a)
            return f"({a} if {cond} else {b})", max(ba, bb)
        if k == "wrap":
            a = self.strong(self.int_expr(d - 1)[0])
            b = self.strong(self.int_expr(d - 1)[0])
            f = self.pick(["wrap_add", "wrap_mul"])
            self.c(f)
            return f"({f}({a}, {b}) % {IMOD})", IMOD
        # an Int from a number: floor, ceil or round, then int()
        x, bx = self.num_expr(d - 1)
        if bx > 2.0 ** 50:
            x, bx = f"tanh({x})", 1.0
        f = self.pick(["floor", "ceil", "round", "int"])
        self.c("int-from-num-" + f)
        # int(...) is weak too: it adapts as a literal does
        e = f"int({x})" if f == "int" else f"int({f}({x}))"
        self.weak.add(e)
        return e, int(bx) + 1

    def weak_if(self, e, *parts):
        """Integer literals and len() are numbers unless they stand where an Int
        is expected (viv features int-type). Arithmetic, abs, min and max of such
        weak parts stay weak and take the Int type from an Int beside them or
        from an Int position; an if-expression or `^` of weak parts alone is a
        number (B111). So an expression is weak when all its parts are, and an
        if-expression of weak parts alone gets one part made an Int with
        strong()."""
        if all(p in self.weak for p in parts):
            self.weak.add(e)
        return e

    def strong(self, e):
        """e as an Int: plus the scope's Int zero, for a weak e where no Int is
        expected."""
        if e in self.weak:
            self.c("int-of-weak")
            return f"({e} + {self.zero})"
        return e

    def int_fit(self, e, b):
        """e reduced into an Int variable's range, and an Int."""
        if b < IB:
            return self.strong(e)
        self.c("int-reduce")
        r = f"({e} % {IMOD})"
        if e in self.weak:
            self.weak.add(r)
        return self.strong(r)

    # ------------------------------------------------------------ number expressions
    def num_leaf(self):
        opts = ["lit", "lit"]
        nvars = self.vars(lambda v: v.ty == "N")
        if nvars:
            opts += ["var"] * 4
        arrs = self.vars(lambda v: v.ty[0] == "A")
        if arrs:
            opts += ["elem"] * 2 + ["reduce"] * 2
        if self.vars(lambda v: v.ty == "I"):
            opts.append("float")
        recs = [(v, f) for v in self.vars(lambda v: v.ty[0] == "R") for f, t in self.records[v.ty[1]] if t == "N"]
        if recs:
            opts.append("field")
        if self.callable("N"):
            opts.append("call")
        opts.append("edge")
        k = self.pick(opts)
        if k == "edge":
            e, b = self.pick(NUM_EDGES)
            self.c("num-edge")
            return e, b
        if k == "var":
            self.c("num-var")
            return self.pick(nvars).name, NB
        if k == "elem":
            v = self.pick(arrs)
            self.c("array-index")
            return f"{v.name}[{self.index(v)}]", NB
        if k == "reduce":
            v = self.pick(arrs)
            n = v.ty[1]
            f = self.pick(["sum", "mean", "max", "min", "dot", "prod", "norm"])
            self.c("reduce-" + f)
            if f == "dot":
                w = self.pick([u for u in arrs if u.ty[1] == n])
                return f"dot({v.name}, {w.name})", n * NB * NB
            if f == "prod":
                return f"prod(tanh({v.name}))", 1.0
            if f == "norm":
                return f"norm({v.name})", n * NB
            return f"{f}({v.name})", n * NB
        if k == "float":
            e, b = self.int_expr(1)
            self.c("float-of-int")
            return f"float({e})", float(b)
        if k == "field":
            v, f = self.pick(recs)
            self.c("record-field-read")
            return f"{v.name}.{f}", NB
        if k == "call":
            return self.call("N")
        self.c("num-literal")
        s = self.pick(NUM_LITS)
        return (f"({s})" if s.startswith("-") else s), max(abs(float(s)), 1e-300)

    def num_expr(self, d=None):
        if d is None:
            d = self.R.randint(0, 3)
        if d <= 0 or self.chance(0.2):
            return self.num_leaf()
        k = self.pick(["+", "-", "*", "*", "/", "neg", "unary", "unary", "sqrt", "exp", "log", "mod", "round",
                       "minmax", "atan2", "pow", "ifx", "fdiv"])
        if k == "fdiv":
            a, ba = self.num_expr(d - 1)
            b, _ = self.num_expr(d - 1)
            self.c("num-floordiv")
            return f"({a} // (abs({b}) + 0.5))", 2 * ba + 1
        if k in ("+", "-", "*"):
            a, ba = self.num_expr(d - 1)
            b, bb = self.num_expr(d - 1)
            bound = ba * bb if k == "*" else ba + bb
            if bound > NLIM:
                a, ba = f"tanh({a})", 1.0
                bound = ba * bb if k == "*" else ba + bb
            self.c("num-" + {"+": "add", "-": "sub", "*": "mul"}[k])
            return f"({a} {k} {b})", bound
        if k == "/":
            a, ba = self.num_expr(d - 1)
            b, _ = self.num_expr(d - 1)
            self.c("num-div")
            return f"({a} / (abs({b}) + 1.0))", ba
        if k == "neg":
            a, ba = self.num_expr(d - 1)
            self.c("num-neg")
            return f"(-{a})", ba
        if k == "unary":
            a, _ = self.num_expr(d - 1)
            f = self.pick(["sin", "cos", "tanh", "atan"])
            self.c(f)
            return f"{f}({a})", 2.0
        if k == "sqrt":
            a, ba = self.num_expr(d - 1)
            self.c("sqrt")
            return f"sqrt(abs({a}))", ba ** 0.5 + 1
        if k == "exp":
            a, _ = self.num_expr(d - 1)
            self.c("exp")
            return f"exp({a} % 8.0)", 2981.0
        if k == "log":
            a, ba = self.num_expr(d - 1)
            self.c("log")
            return f"log(abs({a}) + 1.0)", 40.0
        if k == "mod":
            a, _ = self.num_expr(d - 1)
            m = self.pick(["1.0", "2.5", "1000.0", "0.1", "(-2.5)", "(-1.0)"])
            self.c("num-mod")
            return f"({a} % {m})", abs(float(m.strip("()")))
        if k == "round":
            a, ba = self.num_expr(d - 1)
            f = self.pick(["floor", "ceil", "round", "abs"])
            self.c("num-" + f)
            return f"{f}({a})", ba + 1
        if k == "minmax":
            a, ba = self.num_expr(d - 1)
            b, bb = self.num_expr(d - 1)
            f = self.pick(["min", "max"])
            self.c("num-" + f)
            return f"{f}({a}, {b})", max(ba, bb)
        if k == "atan2":
            a, _ = self.num_expr(d - 1)
            b, _ = self.num_expr(d - 1)
            self.c("atan2")
            return f"atan2({a}, {b})", 4.0
        if k == "pow":
            a, ba = self.num_expr(d - 1)
            if self.chance(0.5):
                self.c("num-pow-half")
                return f"(abs({a}) ^ 0.5)", ba ** 0.5 + 1
            if ba > 1e7:
                a, ba = f"tanh({a})", 1.0
            self.c("num-pow-2")
            return f"({a} ^ 2.0)", ba * ba
        cond = self.cond(d - 1)
        a, ba = self.num_expr(d - 1)
        b, bb = self.num_expr(d - 1)
        self.c("if-expression")
        return f"({a} if {cond} else {b})", max(ba, bb)

    def num_fit(self, e, b):
        if b <= NB:
            return e
        f = self.pick(["mod", "clip", "tanh"])
        self.c("num-reduce-" + f)
        if f == "mod":
            return f"({e} % 1000.0)"
        if f == "clip":
            return f"min(max({e}, -1000.0), 1000.0)"
        return f"tanh({e})"

    # ------------------------------------------------------------ conditions
    def cond(self, d=1):
        k = self.pick(["num", "num", "int", "int", "and", "or", "not", "chain", "in"])
        if d <= 0 and k in ("and", "or", "not"):
            k = "int"
        if k in ("and", "or"):
            self.c("bool-" + k)
            return f"({self.cond(d - 1)} {k} {self.cond(d - 1)})"
        if k == "not":
            self.c("bool-not")
            return f"(not {self.cond(d - 1)})"
        op = self.pick(["<", "<=", ">", ">=", "==", "!="])
        if k == "chain":
            self.c("chained-comparison")
            a, _ = self.int_expr(1)
            b, _ = self.int_expr(1)
            c, _ = self.int_expr(1)
            return f"({a} < {b} <= {c})"
        if k == "in" and self.vars(lambda v: v.ty[0] == "A"):
            v = self.pick(self.vars(lambda v: v.ty[0] == "A"))
            a, _ = self.num_expr(1)
            self.c("in-operator")
            return f"({a} {self.pick(['in', 'not in'])} {v.name})"
        if k == "num":
            self.c("compare-num")
            a, _ = self.num_expr(d)
            b, _ = self.num_expr(d)
        else:
            self.c("compare-int")
            a, _ = self.int_expr(d)
            b, _ = self.int_expr(d)
        return f"({a} {op} {b})"

    # ------------------------------------------------------------ arrays
    def index(self, v):
        """An index into array v (length known), always in range."""
        n = v.ty[1]
        loops = self.vars(lambda u: not u.mutable and 0 < getattr(u, "upto", 0) <= n)
        k = self.pick(["lit", "neg", "mod", "loop", "loop"] if loops else ["lit", "neg", "mod"])
        if k == "loop":
            self.c("index-loop-var")
            return self.pick(loops).name
        if k == "lit":
            self.c("index-literal")
            return str(self.R.randrange(n))
        if k == "neg":
            self.c("negative-index")
            return f"-{self.R.randint(1, n)}"
        e, _ = self.int_expr(1)
        self.c("index-mod-len")
        return f"({e} % len({v.name}))"

    def arr_expr(self, n, d=None):
        """A number array of length n, elements bounded by NB after fitting; (text, bound)."""
        if d is None:
            d = self.R.randint(0, 2)
        same = self.vars(lambda v: v.ty == ("A", n))
        longer = self.vars(lambda v: v.ty[0] == "A" and v.ty[1] > n)
        opts = ["lit", "zeros"]
        if same:
            opts += ["var"] * 3
        if longer:
            opts.append("slice")
        if same and d > 0:
            opts += ["binop", "binop", "scalar", "math", "comp", "rev", "cumsum", "where"]
        if self.callable(("A", n)):
            opts.append("call")
        if n >= 2:
            opts.append("linspace")
        recs = [(v, f) for v in self.vars(lambda v: v.ty[0] == "R") for f, t in self.records[v.ty[1]] if t == ("A", n)]
        if recs:
            opts.append("field")
        k = self.pick(opts)
        if k == "var":
            self.c("array-var")
            return self.pick(same).name, NB
        if k == "zeros":
            self.c("zeros")
            return f"zeros({n})", 0.0
        if k == "linspace":
            self.c("linspace")
            a, ba = self.num_expr(0)
            b, bb = self.num_expr(0)
            return f"linspace({a}, {b}, {n})", max(ba, bb)
        if k == "slice":
            v = self.pick(longer)
            m = v.ty[1]
            if self.chance(0.3) and m >= 2 * n - 1 and n >= 1:
                lo = self.R.randint(0, m - (2 * n - 1))
                self.c("slice-step")
                return f"{v.name}[{self.ibound(lo)}:{self.ibound(lo + 2 * n - 1)}:2]", NB
            lo = self.R.randint(0, m - n)
            self.c("slice")
            return f"{v.name}[{self.ibound(lo)}:{self.ibound(lo + n)}]", NB
        if k == "rev":
            self.c("slice-reverse")
            return f"{self.pick(same).name}[::-1]", NB
        if k == "binop":
            a, ba = self.arr_expr(n, d - 1)
            b, bb = self.arr_expr(n, d - 1)
            op = self.pick(["+", "-", "*"])
            # a one-element list literal beside a comprehension reads as a
            # Python list join, which the front end refuses (lower.rs
            # list_join_trap): the literal goes in as a product
            if n == 1:
                a, b = (f"({x} * 1.0)" if x.startswith("[") and " for " not in x else x for x in (a, b))
            if op == "*" and ba * bb > NLIM:
                a, ba = f"tanh({a})", 1.0
            self.c("array-elementwise-" + op)
            return f"({a} {op} {b})", ba * bb if op == "*" else ba + bb
        if k == "scalar":
            a, ba = self.arr_expr(n, d - 1)
            s, bs = self.num_expr(1)
            if bs > 1e6:
                s, bs = f"tanh({s})", 1.0
            op = self.pick(["+", "*", "-"])
            self.c("array-scalar-" + op)
            if self.chance(0.5):
                return f"({a} {op} {s})", ba * bs if op == "*" else ba + bs
            return f"({s} {op} {a})", ba * bs if op == "*" else ba + bs
        if k == "math":
            a, ba = self.arr_expr(n, d - 1)
            f = self.pick(["sin", "cos", "tanh", "abs", "sqrtabs", "exp"])
            self.c("array-math-" + f)
            if f == "sqrtabs":
                return f"sqrt(abs({a}))", ba ** 0.5 + 1
            if f == "abs":
                return f"abs({a})", ba
            if f == "exp":
                return f"exp(tanh({a}))", 3.0
            return f"{f}({a})", 1.0
        if k == "comp":
            v = self.pick(same)
            x = self.fresh("x")
            self.declare(Var(x, "N", mutable=False))
            e, be = self.num_expr(2)
            self.scopes[-1].pop()
            self.c("list-comprehension")
            return f"[{self.num_fit(e, be)} for {x} in {v.name}]", max(be, NB) if be <= NB else 1000.0
        if k == "cumsum":
            self.c("cumsum")
            return f"cumsum({self.pick(same).name})", n * NB
        if k == "where":
            v = self.pick(same)
            w = self.pick(same)
            t, _ = self.num_expr(0)
            self.c("where-mask")
            return f"where({v.name} > {t}, {v.name}, {w.name})", NB
        if k == "call":
            return self.call(("A", n))
        if k == "field":
            v, f = self.pick(recs)
            self.c("record-field-read")
            return f"{v.name}.{f}", NB
        self.c("array-literal")
        items = []
        for _ in range(n):
            e, _ = self.num_leaf() if self.chance(0.4) else (self.pick(NUM_LITS), 0)
            items.append(e)
        return "[" + ", ".join(items) + "]", NB

    def arr_fit(self, e, b):
        if b <= NB:
            return e
        self.c("array-clip")
        return f"clip({e}, -1000.0, 1000.0)"

    def iarr_expr(self, n):
        same = self.vars(lambda v: v.ty == ("IA", n))
        if same and self.chance(0.4):
            self.c("int-array-var")
            return self.pick(same).name
        self.c("int-array-literal")
        return "[" + ", ".join(self.int_fit(*self.int_expr(1)) for _ in range(n)) + "]"

    # ------------------------------------------------------------ records
    def rec_expr(self, rname):
        fields = self.records[rname]
        same = self.vars(lambda v: v.ty == ("R", rname))
        if same and self.chance(0.3):
            base = self.pick(same).name
            f, t = self.pick(fields)
            self.c("record-update")
            return f"{rname} {{ {f}: {self.value(t)}, ..{base} }}"
        if self.callable(("R", rname)) and self.chance(0.3):
            return self.call(("R", rname))[0]
        self.c("record-literal")
        return f"{rname} {{ " + ", ".join(f"{f}: {self.value(t)}" for f, t in fields) + " }"

    def value(self, ty):
        """An expression of type ty, fit for storing."""
        if ty == "N":
            return self.num_fit(*self.num_expr())
        if ty == "I":
            return self.int_fit(*self.int_expr())
        if ty[0] == "A":
            return self.arr_fit(*self.arr_expr(ty[1]))
        if ty[0] == "IA":
            return self.iarr_expr(ty[1])
        return self.rec_expr(ty[1])

    # ------------------------------------------------------------ calls
    def callable(self, ret):
        if self.call_depth >= 2:
            return []
        return [d for d in self.defs if d.ret == ret and (self.in_def is None or d is not self.in_def)]

    def call(self, ret):
        d = self.pick(self.callable(ret))
        self.call_depth += 1
        args = []
        for pname, t in d.params:
            if t[0] == "A" or t[0] == "R" or t[0] == "IA":
                cands = self.vars(lambda v: v.ty == t)
                if cands and self.chance(0.7):
                    args.append(self.pick(cands).name)
                    self.c("pass-alias")
                    continue
            if d.recursive and t == "I":
                args.append(str(self.R.randint(0, 11)))
                continue
            args.append(self.value(t))
        if self.chance(0.2) and len(d.params) > 0:
            # a keyword argument for the last parameter
            self.c("keyword-argument")
            args[-1] = f"{d.params[-1][0]}={args[-1]}"
        self.call_depth -= 1
        self.c("call-recursive" if d.recursive else "call")
        b = {"N": NB, "I": IB}.get(ret, NB)
        return f"{d.name}({', '.join(args)})", b

    # ------------------------------------------------------------ statements
    def new_type(self):
        R = self.R
        opts = ["N", "N", "I", "I", "A", "A", "IA"]
        if self.records:
            opts.append("R")
        k = self.pick(opts)
        if k == "A":
            return ("A", self.arr_len())
        if k == "IA":
            return ("IA", self.arr_len())
        if k == "R":
            return ("R", self.pick(sorted(self.records)))
        return k

    def decl(self):
        ty = self.new_type()
        name = self.fresh()
        val = self.value(ty)
        if ty == "I":
            self.emit(f"{name}: Int = {val}")
            self.c("decl-int")
        elif ty[0] == "IA":
            self.emit(f"{name}: [Int] = {val}")
            self.c("decl-int-array")
        else:
            self.emit(f"{name} = {val}")
            self.c("decl-" + tyname(ty).lower())
        self.declare(Var(name, ty))

    def assign(self):
        vs = self.vars(lambda v: v.mutable)
        if not vs:
            return self.decl()
        v = self.pick(vs)
        t = v.ty
        R = self.R
        if t == "N":
            if self.chance(0.4):
                op = self.pick(["+=", "-=", "*="])
                e, b = self.num_expr(2)
                if op == "*=":
                    e = f"tanh({e})"
                self.emit(f"{v.name} {op} {e}")
                self.c("compound-assign")
                self.emit(f"{v.name} = {self.num_fit(v.name, NLIM)}")
            else:
                self.emit(f"{v.name} = {self.value('N')}")
                self.c("assign-num")
        elif t == "I":
            if self.chance(0.4):
                op = self.pick(["+=", "-=", "*="])
                e, b = self.int_expr(2)
                if b > 1000 and op == "*=":
                    e = f"({e} % 1000)"
                elif b > ILIM // 2:
                    e = f"({e} % {IMOD})"
                self.emit(f"{v.name} {op} {self.strong(e)}")
                self.c("compound-assign-int")
                self.emit(f"{v.name} = {v.name} % {IMOD}")
            else:
                self.emit(f"{v.name} = {self.value('I')}")
                self.c("assign-int")
        elif t[0] == "A":
            k = self.pick(["elem", "elem", "slice", "whole", "alias", "sort"])
            if k == "sort":
                # in place, through every alias of the array
                self.emit(f"sort({v.name})")
                self.c("sort-in-place")
                return
            n = t[1]
            if k == "elem":
                self.emit(f"{v.name}[{self.index(v)}] = {self.value('N')}")
                self.c("array-element-write")
            elif k == "slice" and n >= 2:
                lo = R.randint(0, n - 1)
                hi = R.randint(lo + 1, n)
                self.emit(f"{v.name}[{lo}:{hi}] = {self.arr_fit(*self.arr_expr(hi - lo))}")
                self.c("slice-write")
            elif k == "alias":
                others = self.vars(lambda u: u.ty == t and u.mutable and u is not v)
                if others:
                    self.emit(f"{v.name} = {self.pick(others).name}")
                    self.c("array-alias")
                else:
                    self.emit(f"{v.name} = copy({self.arr_fit(*self.arr_expr(n))})")
                    self.c("copy")
            else:
                self.emit(f"{v.name} = {self.value(t)}")
                self.c("assign-array")
        elif t[0] == "IA":
            self.emit(f"{v.name}[{self.index(v)}] = {self.value('I')}")
            self.c("int-array-element-write")
        else:
            fields = self.records[t[1]]
            f, ft = self.pick(fields)
            if ft[0] == "A" and self.chance(0.5):
                fake = Var(f"{v.name}.{f}", ft)
                self.emit(f"{v.name}.{f}[{self.index(fake)}] = {self.value('N')}")
                self.c("record-field-element-write")
            elif self.chance(0.6):
                self.emit(f"{v.name}.{f} = {self.value(ft)}")
                self.c("record-field-write")
            else:
                self.emit(f"{v.name} = {self.rec_expr(t[1])}")
                self.c("assign-record")

    def block(self, body):
        self.ind += 1
        self.scopes.append([])
        body()
        self.scopes.pop()
        self.ind -= 1

    def stmts(self, k):
        for _ in range(k):
            if self.budget <= 0:
                return
            self.budget -= 1
            self.stmt()

    def stmt(self):
        R = self.R
        deep = self.ind >= 4
        opts = ["decl", "assign", "assign", "assign"]
        if not deep:
            opts += ["if", "for", "for", "while", "forin"]
        if self.loop_depth > 0:
            opts += ["break"]
        if self.defs and any(d.ret not in ("N", "I") for d in self.defs):
            opts.append("callstmt")
        if [d for d in self.defs if d.ret == "N2"]:
            opts.append("unpack")
        opts.append("print")
        k = self.pick(opts)
        if k == "decl":
            self.decl()
        elif k == "assign":
            self.assign()
        elif k == "print":
            vs = self.vars(lambda v: v.ty in ("N", "I"))
            if vs:
                v = self.pick(vs)
                if v.ty == "N":
                    form = self.pick(["fmt({})", "{}", "fmt({}, 17)", "fmt({}, \".3e\")", "fmt({}, \"12.4f\")",
                                      "f\"{{{}:.4f}}\"", "f\"{{{}:g}}\"", "f\"{{{}}}\"", "fmt({}, \"g\")"])
                else:
                    form = self.pick(["{}", "f\"{{{}}}\"", "fmt({}, \"d\")", "f\"{{{}:6d}}\""])
                self.emit(f"print({form.format(v.name)})")
                self.c("print-" + form.replace("{}", "x").replace("{{", "{").replace("}}", "}"))
        elif k == "if":
            self.emit(f"if {self.cond(2)} {{")
            self.block(lambda: self.stmts(R.randint(1, 3)))
            r = R.random()
            if r < 0.3:
                self.emit(f"}} else if {self.cond(1)} {{")
                self.block(lambda: self.stmts(R.randint(1, 2)))
                self.c("else-if")
            if r < 0.6:
                self.emit("} else {")
                self.block(lambda: self.stmts(R.randint(1, 2)))
                self.c("else")
            self.emit("}")
            self.c("if")
        elif k == "for":
            self.for_loop()
        elif k == "forin":
            self.forin_loop()
        elif k == "while":
            self.while_loop()
        elif k == "break":
            self.emit(f"if {self.cond(1)} {{")
            self.ind += 1
            self.emit(self.pick(["break", "continue"]))
            self.ind -= 1
            self.emit("}")
            self.c("break-continue")
        elif k == "callstmt":
            d = self.pick([d for d in self.defs if d.ret not in ("N", "I", "N2")] or self.defs)
            if d.ret == "N2":
                return self.unpack()
            name = self.fresh()
            e, _ = self.call(d.ret)
            self.emit(f"{name} = {e}" if d.ret != "I" else f"{name}: Int = {e}")
            self.declare(Var(name, d.ret))
        elif k == "unpack":
            self.unpack()

    def unpack(self):
        d = self.pick([d for d in self.defs if d.ret == "N2"])
        a, b = self.fresh(), self.fresh()
        args = ", ".join(self.value(t) for _, t in d.params)
        self.emit(f"{a}, {b} = {d.name}({args})")
        self.declare(Var(a, "N"))
        self.declare(Var(b, "I"))
        self.c("multiple-return-unpack")

    def loop_var(self, upto=0):
        v = Var(self.fresh("i"), "I", mutable=False)
        v.upto = upto
        return v

    def for_loop(self):
        R = self.R
        k = self.pick(["len", "const", "from", "reversed", "lit"])
        arrs = self.vars(lambda v: v.ty[0] in ("A", "IA"))
        lv = self.loop_var()
        if k == "len" and arrs:
            a = self.pick(arrs)
            lv.upto = a.ty[1]
            lv.ty = "N"
            # an index of every array at least this long
            self.emit(f"for {lv.name} to len({a.name}) {{")
            self.c("for-len")
        elif k == "from":
            lo = R.randint(-3, 3)
            hi = lo + R.randint(0, 6)
            by = R.choice([1, 2, 3])
            if R.random() < 0.3:
                self.emit(f"for {lv.name} from {hi} to {lo} by -{by} {{")
                self.c("for-from-by-negative")
            else:
                self.emit(f"for {lv.name} from {lo} to {hi} by {by} {{")
                self.c("for-from-by")
            lv.ty = "N"
        elif k == "reversed":
            n = R.randint(0, 6)
            cn = self.fresh("n")
            self.emit(f"{cn}: Int = {n}")
            self.declare(Var(cn, "I", mutable=False))
            lv.upto = n
            self.emit(f"for {lv.name} to {cn} reversed {{")
            self.c("for-reversed")
        elif k == "lit":
            n = R.randint(0, 6)
            self.emit(f"for {lv.name} to {n} {{")
            lv.ty = "N"
            self.c("for-literal-bound")
        else:
            n = R.randint(0, 6)
            cn = self.fresh("n")
            self.emit(f"{cn}: Int = {n}")
            self.declare(Var(cn, "I", mutable=False))
            lv.upto = n
            self.emit(f"for {lv.name} to {cn} {{")
            self.c("for-int-bound")
        self.loop_depth += 1
        self.block(lambda: (self.declare(lv), self.stmts(R.randint(1, 4))))
        self.loop_depth -= 1
        self.emit("}")

    def forin_loop(self):
        R = self.R
        arrs = self.vars(lambda v: v.ty[0] == "A")
        if not arrs:
            return self.for_loop()
        a = self.pick(arrs)
        x = Var(self.fresh("x"), "N", mutable=False)
        if self.chance(0.4):
            i = self.loop_var(a.ty[1])
            if self.chance(0.5):
                # a literal start makes the index an Int (viv features enumerate-start)
                self.emit(f"for {i.name}, {x.name} in enumerate({a.name}, start=0) {{")
                self.c("for-enumerate-start")
            else:
                i.ty = "N"
                self.emit(f"for {i.name}, {x.name} in enumerate({a.name}) {{")
                self.c("for-enumerate")
            body = lambda: (self.declare(i), self.declare(x), self.stmts(R.randint(1, 3)))
        else:
            self.emit(f"for {x.name} in {a.name} {{")
            self.c("for-in")
            body = lambda: (self.declare(x), self.stmts(R.randint(1, 3)))
        self.loop_depth += 1
        self.block(body)
        self.loop_depth -= 1
        self.emit("}")

    def while_loop(self):
        R = self.R
        w = self.fresh("w")
        self.emit(f"{w}: Int = 0")
        self.declare(Var(w, "I", mutable=False))
        lim = R.randint(0, 8)
        extra = f" and {self.cond(1)}" if self.chance(0.5) else ""
        self.emit(f"while {w} < {lim}{extra} {{")
        self.c("while")

        def body():
            self.emit(f"{w} += 1")
            self.stmts(R.randint(1, 3))
        self.loop_depth += 1
        self.block(body)
        self.loop_depth -= 1
        self.emit("}")

    def show(self, v):
        if v.ty == "N":
            return f"fmt({v.name})"
        return v.name

    # ------------------------------------------------------------ program
    def record_decls(self):
        for r in range(self.R.randint(0, 2)):
            name = f"Rec{r}"
            fields = []
            for j in range(self.R.randint(1, 4)):
                t = self.pick(["N", "N", "I", "A"])
                if t == "A":
                    t = ("A", self.arr_len())
                fields.append((f"f{j}", t))
            self.records[name] = fields
            self.emit(f"record {name} {{")
            for f, t in fields:
                self.emit(f"    {f}: {tyname(t)}")
            self.emit("}")
            self.c("record-decl")

    def def_decls(self):
        R = self.R
        for k in range(R.randint(0, 4)):
            name = f"fn{k}"
            if self.chance(0.15):
                d = Def(name, [("n", "I"), ("x", "N")], "N", recursive=True)
                self.emit(f"def {name}(n: Int, x: Num) -> Num {{")
                self.ind += 1
                self.scopes = [[Var("n", "I", mutable=False), Var("x", "N", mutable=False)]]
                self.in_def = d      # its own body never calls it with a fresh depth
                self.zero = self.fresh("z")
                self.emit(f"{self.zero}: Int = 0")
                self.emit("if n <= 0 {")
                e, b = self.num_expr(2)
                self.emit(f"    return {self.num_fit(e, b)}")
                self.emit("}")
                step, bs = self.num_expr(2)
                self.emit(f"return {name}(n - 1, {self.num_fit(step, bs)})")
                self.defs.append(d)
                self.in_def = None
                self.ind -= 1
                self.emit("}")
                self.c("def-recursive")
                continue
            params = []
            for j in range(R.randint(0, 3)):
                t = self.new_type()
                params.append((f"p{j}", t))
            ret = self.pick(["N", "N", "I", ("A", self.arr_len()), "N2"] + (["R"] if self.records else []))
            if ret == "R":
                ret = ("R", self.pick(sorted(self.records)))
            d = Def(name, params, ret)
            sig = ", ".join(f"{p}: {tyname(t)}" for p, t in params)
            arrow = "" if ret == "N2" else f" -> {tyname(ret)}"
            self.emit(f"def {name}({sig}){arrow} {{")
            self.ind += 1
            self.in_def = d
            # a def does not change a record it was given (viv features field-assignment)
            self.scopes = [[Var(p, t, mutable=t[0] != "R") for p, t in params]]
            self.zero = self.fresh("z")
            self.emit(f"{self.zero}: Int = 0")
            saved = self.budget
            self.budget = R.randint(1, 8)
            self.stmts(self.budget)
            self.budget = saved
            if ret == "N2":
                if self.chance(0.3):
                    # a slot holding what int() gives is an Int (B305)
                    x, bx = self.num_expr(1)
                    islot = f"int({x})" if bx < IB else f"int(tanh({x}) * 1000.0)"
                    self.c("multiple-return-int-call")
                else:
                    islot = self.value('I')
                self.emit(f"return {self.value('N')}, {islot}")
                self.c("multiple-return")
            else:
                self.emit(f"return {self.value(ret)}")
            self.ind -= 1
            self.emit("}")
            self.c("def")
            self.defs.append(d)
            self.in_def = None
        self.scopes = [[]]
        self.zero = "z0"

    def contract_decls(self):
        """A contract with 2 or 3 methods, each a def
        with the contract's signature and its own random body. The methods give
        different numbers, all within NB, so `agree ... within absolute 1.0e7` always
        holds; an Int output is the same literal in every method, compared
        `exactly`. The calls the statements make reach it like any def; the
        program also calls it with `using <method>` and switches to `using
        <contract> = every method`."""
        R = self.R
        if not self.chance(0.5):
            return None
        name = "ct0"
        params = [(f"q{j}", self.new_type()) for j in range(R.randint(1, 3))]
        two = self.chance(0.5)
        ret = "N2" if two else "N"
        nm = R.randint(2, 3)
        lit = R.randint(0, 99)
        sig = ", ".join(f"{p}: {tyname(t)}" for p, t in params)
        methods = []
        given = []
        for k in range(nm):
            mname = f"{name}_m{k}"
            # a setting: bound in the methods line, or a default in the
            # signature, sometimes given by a top-level `settings` line
            setting = self.chance(0.4)
            default = setting and self.chance(0.5)
            extra = (", s: Num = 1.5" if default else ", s: Num") if setting else ""
            self.emit(f"def {mname}({sig}{extra}) {{")
            self.ind += 1
            d = Def(mname, params, ret)
            self.in_def = d
            self.scopes = [[Var(p, t, mutable=t[0] != "R") for p, t in params] + ([Var("s", "N", mutable=False)] if setting else [])]
            self.zero = self.fresh("z")
            self.emit(f"{self.zero}: Int = 0")
            saved = self.budget
            self.budget = R.randint(1, 5)
            self.stmts(self.budget)
            self.budget = saved
            e, b = self.num_expr(2)
            v = self.num_fit(e, b)
            # `+ z` keeps the literal an Int in a multiple return
            self.emit(f"return {v}, {lit} + {self.zero}" if two else f"return {v}")
            self.ind -= 1
            self.emit("}")
            self.in_def = None
            if default and self.chance(0.6):
                given.append(f"settings {mname}(s = {R.choice(['0.5', '2.0', '(-1.25)'])})")
                self.c("contract-settings-line")
            bound = setting and not default
            methods.append(f"{mname}(s = {R.choice(['0.5', '2.0', '(-1.25)'])})" if bound else mname)
        outs = "out0: Num, out1: Int" if two else "out0: Num"
        self.emit(f"contract {name}({sig}) -> {outs} {{")
        self.emit(f'    intent "a random contract, seed {self.seed}"')
        groups = None
        if nm == 3 and self.chance(0.5):
            # methods in two labeled groups; `using every <label> method`
            groups = [("one", methods[:1]), ("two", methods[1:])]
            self.emit("    methods")
            for label, ms in groups:
                self.emit(f"        {label}: {', '.join(ms)}")
            self.c("contract-groups")
        else:
            self.emit(f"    methods {', '.join(methods)}")
        self.contract_groups = [label for label, _ in groups] if groups else []
        if self.chance(0.3):
            self.emit(f"    default {name}_m{R.randint(0, nm - 1)}")
            self.c("contract-default")
        self.emit("    agree out0 within absolute 1.0e7")
        if two:
            self.emit("    agree out1 exactly")
        elif self.chance(0.3):
            self.emit("    consensus median")
            self.c("contract-median")
        if self.chance(0.3):
            self.emit("    using every method")
            self.c("contract-start-every")
        self.emit("}")
        for line in given:
            self.emit(line)
        self.scopes = [[]]
        self.zero = "z0"
        self.c("contract")
        d = Def(name, params, ret)
        return d, [f"{name}_m{k}" for k in range(nm)]

    def contract_calls(self, d, methods):
        """Call the contract with each method, then switch to every method."""
        def call(using):
            args = ", ".join(self.value(t) for _, t in d.params)
            if d.ret == "N2":
                a, b = self.fresh(), self.fresh()
                self.emit(f"{a}, {b} = {d.name}({args}){using}")
                self.declare(Var(a, "N"))
                self.declare(Var(b, "I"))
            else:
                a = self.fresh()
                self.emit(f"{a} = {d.name}({args}){using}")
                self.declare(Var(a, "N"))
        call("")
        for m in methods:
            if self.chance(0.6):
                call(f" using {m}")
                self.c("contract-call-using")
        for label in self.contract_groups:
            if self.chance(0.6):
                call(f" using every {label} method")
                self.c("contract-call-every-group")
        if self.chance(0.6):
            self.emit(f"using {d.name} = every method")
            self.c("contract-using-every")
            call("")
        self.defs.append(d)

    def grow_decls(self):
        """Defs that grow what they were passed and hand it back (B340), and a
        record of lists (B341)."""
        self.grow = self.G.random() < 0.5
        if not self.grow:
            return
        self.emit("record Lists {")
        self.emit("    names: [Str]")
        self.emit("    ids: [Int]")
        self.emit("}")
        self.emit("def grow_i(out: [Int], v: Int) -> [Int] {")
        self.emit("    out = append(out, v)")
        self.emit("    return out")
        self.emit("}")
        self.emit("def grow_n(out: Arr, v: Num) -> Arr {")
        self.emit("    out = append(out, v)")
        self.emit("    return out")
        self.emit("}")
        self.emit("def grow_r(s: Lists, v: Int) -> Lists {")
        self.emit("    t = s")
        self.emit("    t.ids = append(t.ids, v)")
        self.emit("    t.names = append(t.names, f\"n{v % 10}\")")
        self.emit("    return t")
        self.emit("}")
        self.c("append-parameter-defs")

    def growth(self):
        """At the top level, after the program: the growth defs called many
        times, then an [Int] joined from the program's Int arrays and summed,
        sliced and sorted (B343). Every length is known here, so every bound
        is in range; the Int values stay below IMOD, so no sum overflows."""
        if not self.grow:
            return
        G = self.G
        ivars = [v.name for v in self.scopes[0] if v.ty == "I"]
        n = self.fresh("g")
        calls = G.choice([0, 1, 5, G.randint(1, GROW_MAX), G.randint(65, GROW_MAX)])
        seed = G.choice(ivars) if ivars else str(G.randint(0, 99))
        self.emit(f"{n}: [Int] = []")
        self.emit(f"{n}_x: Arr = []")
        empty = G.random() < 0.5
        names = "[]" if empty else '["a", "b"]'
        ids = "[]" if empty else "[3]"
        self.emit(f"{n}_r = Lists {{ names: {names}, ids: {ids} }}")
        self.emit(f"for k to {calls} {{")
        self.emit(f"    {n}_v: Int = (int(k) * 37 + {seed}) % {IMOD}")
        self.emit(f"    {n} = grow_i({n}, {n}_v)")
        self.emit(f"    {n}_x = grow_n({n}_x, float({n}_v) / 7.0)")
        self.emit(f"    {n}_r = grow_r({n}_r, {n}_v)")
        self.emit("}")
        self.c("append-parameter-many" if calls > 64 else "append-parameter")
        self.emit(f'print("{n}", len({n}), sum({n}), len({n}_x), fmt(sum({n}_x)), len({n}_r.names), sum({n}_r.ids))')
        if calls > 0:
            self.emit(f'print("{n} ends", min({n}), max({n}), {n}[{calls - 1}], {n}_r.names[{calls - 1 + (0 if empty else 2)}])')
        # an [Int] joined by extend from the program's Int arrays and the grown one
        e = f"{n}_e"
        self.emit(f"{e}: [Int] = []")
        total = 0
        parts = [v for v in self.scopes[0] if v.ty[0] == "IA"]
        G.shuffle(parts)
        for v in parts[:3]:
            self.emit(f"{e} = extend({e}, {v.name})")
            total += v.ty[1]
        if G.random() < 0.5:
            self.emit(f"{e} = extend({e}, {n})")
            total += calls
        self.c("int-array-extend")
        self.emit(f'print("{e}", len({e}), sum({e}))')
        if total > 0:
            lo = G.randint(0, total - 1)
            hi = G.randint(lo, total)
            self.emit(f"{e}_lo: Int = {lo}")
            self.emit(f"{e}_s = {e}[{e}_lo:{hi}]")
            self.emit(f"sort({e})")
            self.emit(f'print("{e} slice", {e}_s, sum({e}_s), min({e}), max({e}), {e}[{self.ibound(0)}:{self.ibound(min(total, 8))}])')
            self.c("int-array-slice-sum-sort")

    def f32_expr(self, names, d):
        """A Float32 expression over `names`: IEEE operations are total, so
        no value needs a bound (an inf or a nan prints the same everywhere)."""
        G = self.F
        if d <= 0 or G.random() < 0.3:
            r = G.random()
            if r < 0.5 and names:
                return G.choice(names)
            if r < 0.75:
                return G.choice(F32_LITS)
            nv = [v.name for v in self.scopes[0] if v.ty == "N"]
            return f"float32({G.choice(nv)})" if nv else G.choice(F32_LITS)
        k = G.random()
        a = self.f32_expr(names, d - 1)
        if k < 0.55:
            b = self.f32_expr(names, d - 1)
            # a literal beside a literal is a number, not a Float32
            if a in F32_LITS and b in F32_LITS:
                a = f"float32({a})"
            return f"({a} {G.choice(['+', '-', '*', '/'])} {b})"
        if k < 0.65:
            b = self.f32_expr(names, d - 1)
            if a in F32_LITS and b in F32_LITS:
                a = f"float32({a})"
            return f"{G.choice(['min', 'max'])}({a}, {b})"
        if a in F32_LITS:
            a = f"float32({a})"
        return G.choice(["sqrt(abs({}))", "abs({})", "exp({})", "log(abs({}))", "sin({})", "cos({})", "tanh({})", "bfloat16({})", "float16({})", "(-{})"]).format(a)

    def float32s(self):
        """At the top level, after the program (from a third random stream,
        so a seed's other statements are what they were): Float32 scalars,
        a [Float32] filled in a loop and summed in NumPy's order, its dot
        with another, its bfloat16 and float16 copies, and an [Int8] of
        values kept in range."""
        G = self.F
        if G.random() < 0.3:
            return
        names = []
        for _ in range(G.randint(1, 4)):
            n = self.fresh("f")
            e = self.f32_expr(names, 3)
            if e in F32_LITS:
                e = f"float32({e})"
            self.emit(f"{n}: Float32 = {e}" if G.random() < 0.5 else f"{n} = {e}")
            names.append(n)
        self.c("float32-scalars")
        size = G.choice([0, 1, 7, 9, 130, G.randint(1, 300), 8200])
        w, u = self.fresh("fw"), self.fresh("fu")
        self.emit(f"{w} = float32_zeros({size})")
        self.emit(f"{u} = float32_zeros({size})")
        self.emit(f"for k to {size} {{")
        self.emit(f"    {w}[k] = float32(k * {G.randint(1, 9999)} % {G.randint(2, 1000)}) / {G.choice(['3', '7.5', '0.1', '997'])}")
        self.emit(f"    {u}[k] = {w}[k] * {G.choice(names)} - {G.choice(F32_LITS)}")
        self.emit("}")
        h = self.fresh("fh")
        self.emit(f"{h} = {G.choice(['bfloat16', 'float16'])}({u})")
        q = self.fresh("fq")
        self.emit(f"{q} = int8_zeros({size})")
        self.emit(f"for k to {size} {{")
        self.emit(f"    {q}[k] = (int(k) * {G.randint(1, 999)} + {G.randint(0, 99)}) % 256 - 128")
        self.emit("}")
        self.c("float32-arrays")
        self.emit(f'print("{w}", fmt(sum({w})), fmt(sum({u})), fmt(dot({w}, {u})), fmt(sum({h})), sum({q}))')
        if size > 0:
            self.emit(f'print("{w} ends", fmt({w}[{size - 1}]), fmt({u}[0]), fmt({h}[{size // 2}]), {q}[{size - 1}])')
        self.emit("print(" + ", ".join(f"fmt({n})" for n in names) + ")")

    def program(self):
        self.record_decls()
        self.def_decls()
        ct = self.contract_decls()
        self.grow_decls()
        self.emit("z0: Int = 0")
        if ct:
            self.contract_calls(*ct)
        for _ in range(3):
            self.decl()
        self.stmts(self.budget)
        self.growth()
        self.float32s()
        self.emit("# every top-level value, each bit")
        for v in self.scopes[0]:
            self.final_print(v.name, v.ty)
        return "\n".join(self.out) + "\n"

    def final_print(self, name, ty):
        if ty == "N":
            self.emit(f'print("{name}", fmt({name}))')
        elif ty == "I":
            self.emit(f'print("{name}", {name})')
        elif ty[0] == "A":
            self.emit(f'print("{name}", ' + ", ".join(f"fmt({name}[{i}])" for i in range(ty[1])) + ")")
        elif ty[0] == "IA":
            self.emit(f'print("{name}", ' + ", ".join(f"{name}[{i}]" for i in range(ty[1])) + ")")
        else:
            for f, t in self.records[ty[1]]:
                self.final_print(f"{name}.{f}", t)


def generate(seed, size=1.0):
    g = Gen(seed, size)
    src = g.program()
    return f"# vivsmith seed {seed} size {size}\n" + src, g.cover


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = int(args[0]) if args else 0
    size = float(args[args.index("--size") + 1]) if "--size" in args else 1.0
    src, cover = generate(seed, size)
    if "--cover" in args:
        for k, n in sorted(cover.items()):
            print(f"{n:5d}  {k}")
    else:
        sys.stdout.write(src)

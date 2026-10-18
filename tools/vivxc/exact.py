# SPDX-License-Identifier: MIT OR Apache-2.0
"""The translated formula evaluated at 60 digits (mpmath), with derivatives by
mpmath's numerical differentiation at that precision: the reference the
oracle falls back to where libxc's double-precision result is itself decided
by rounding (it never replaces libxc where the two agree)."""
import mpmath as mp

import expr as E
import gen

F = {"exp": mp.exp, "log": mp.log, "sqrt": mp.sqrt, "abs": abs, "asinh": mp.asinh, "atanh": mp.atanh,
     "atan": mp.atan, "tanh": mp.tanh, "sinh": mp.sinh, "cosh": mp.cosh, "erf": mp.erf, "sin": mp.sin,
     "cos": mp.cos, "tan": mp.tan, "asin": mp.asin, "acos": mp.acos, "cbrt": mp.cbrt, "sech": mp.sech}
CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "=": lambda a, b: a == b, "<>": lambda a, b: a != b}


def evaluate(root, args):
    """root over the variables named in args (dict name -> mpf)."""
    memo = {}

    def ev(n):
        if n.id in memo:
            return memo[n.id]
        o = n.op
        if o == "const":
            r = n.val
        elif o == "var":
            r = args[n.name]
        elif o == "bool":
            r = bool(n.exact)
        elif o == "pw":
            r = ev(n.args[1]) if ev(n.args[0]) else ev(n.args[2])
        elif o == "and":
            r = ev(n.args[0]) and ev(n.args[1])
        elif o == "or":
            r = ev(n.args[0]) or ev(n.args[1])
        else:
            a = [ev(c) for c in n.args]
            if o == "add":
                r = a[0] + a[1]
            elif o == "sub":
                r = a[0] - a[1]
            elif o == "mul":
                r = a[0] * a[1]
            elif o == "div":
                r = a[0] / a[1]
            elif o == "pow":
                r = mp.power(a[0], a[1])
            elif o == "neg":
                r = -a[0]
            elif o == "not":
                r = not a[0]
            elif o == "cmp":
                r = CMP[n.name](a[0], a[1])
            else:
                r = F[n.name](a[0])
        memo[n.id] = r
        return r
    return ev(root)


def reference(row, libxc_src, fam, pol, x):
    """[zk, d mf / d x_k ...] at the libxc-screened input x (the libxc variable order)."""
    variant = "pol" if pol else "unpol"
    with mp.workdps(60):
        root, _ = gen.graph(row, libxc_src, variant)
        amap = gen.ARG_INPUT[(fam, variant)]
        xs = [mp.mpf(float(v)) for v in x]

        def zk(*v):
            return evaluate(root, {a: v[i] for a, i in amap.items()})

        def mf(*v):
            d = v[0] + v[1] if pol else v[0]
            return d * zk(*v)
        out = [zk(*xs)]
        for k in range(len(xs)):
            h = mp.mpf(10) ** -25 * max(abs(xs[k]), mp.mpf(10) ** -30)
            out.append(mp.diff(mf, xs, tuple(1 if j == k else 0 for j in range(len(xs))), h=h))
        return [float(v) for v in out]

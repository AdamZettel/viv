# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Emit lib/hfref/boys.viv: the Boys function F_n(x) from mpmath, for lib/hf/boys to be held against.

    python3 tools/boys_reference.py

F_n(x) = int_0^1 t^(2n) exp(-x t^2) dt for n = 0..16 on a set of x from 0 to 200.
mpmath works at 40 digits and each value is rounded once, to the nearest double,
and written with repr(), which round-trips. So the table is correct to the last
bit of a double and the only error in a comparison is Viv's.

The values come from the lower incomplete gamma function,
F_n(x) = gamma(n + 1/2, x) / (2 x^(n + 1/2)), and before anything is written the
generator checks them three ways at 40 digits: against mpmath's own quadrature of
the integral, against the downward recursion F_(n-1) = (2x F_n + e^-x) / (2n - 1),
and against F_n(0) = 1/(2n + 1). A table that fails any of these is not written.

The x values bracket every place where lib/hf/boys changes method: its switch
x_n between the series and the asymptotic form for each order n (read from
boys_switch in that file, with x_n - 0.1, x_n and x_n + 0.1 all tabulated), and
the old fixed switch at 50 in lib/gto_md.

Each x_n is also recomputed here, and the table is not written unless the list
in lib/hf/boys matches. The rule: x_n is the smallest multiple of 0.1 such that,
for every order m <= n, (a) the erfc tail that sqrt(pi/x)/2 drops from F_0,
carried up to F_m by the exact upward recursion, is at most 2^-53 of F_m, and
(b) 2x > 2m + 1, so the upward step does not amplify rounding. The list is also
written into the table as boys_ref_switch(), so the Viv tests catch a change to
either side made without rerunning this script.

The output is deterministic, so running this twice leaves git with no diff.
"""
import os
import re

import mpmath

mpmath.mp.dps = 40

NMAX = 16
CHUNK = 512  # values per literal; larger literals cost superlinear cc -O2 time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "lib", "hf", "boys.viv")
OUT = os.path.join(ROOT, "lib", "hfref", "boys.viv")


def boys_switch_lib():
    """The per-order switch list, as boys_switch in lib/hf/boys.viv holds it."""
    with open(LIB) as f:
        text = f.read()
    found = re.findall(r"^def boys_switch\(n\) \{\n    xs = \[([0-9., ]+)\]$", text, re.M)
    assert len(found) == 1, "lib/hf/boys.viv must define boys_switch exactly once"
    xs = [s.strip() for s in found[0].split(",")]
    assert len(xs) == NMAX + 1, "boys_switch needs one entry per order 0..16"
    return xs


def tail_share(m, x):
    """The erfc tail dropped from F_0, carried exactly up to F_m, over F_m."""
    d = mpmath.sqrt(mpmath.pi / x) / 2 * mpmath.erfc(mpmath.sqrt(x))
    for k in range(m):
        d = d * (2 * k + 1) / (2 * x)
    return d / boys_mp(m, x)


def switch_ok(n, x):
    return all(tail_share(m, x) <= mpmath.mpf(2) ** -53 and 2 * x > 2 * m + 1
               for m in range(n + 1))


def boys_switch_mp():
    """x_n for n = 0..NMAX by the rule in the module docstring, as decimal strings."""
    out = []
    k = 10
    for n in range(NMAX + 1):
        # x_n never falls with n, since the rule for n includes the rule for n - 1
        while not switch_ok(n, mpmath.mpf(k) / 10):
            k += 1
        out.append(mpmath.mpf(k) / 10)
    return out


def boys_mp(n, x):
    if x == 0:
        return mpmath.mpf(1) / (2 * n + 1)
    half = n + mpmath.mpf(1) / 2
    return mpmath.gammainc(half, 0, x) / (2 * x ** half)


def x_points(switches):
    pts = [mpmath.mpf(0), mpmath.mpf("1e-8"), mpmath.mpf("1e-4"), mpmath.mpf("1e-3")]
    # a log grid, 8 points per decade, from 0.01 to 178
    for k in range(-16, 19):
        pts.append(mpmath.mpf(10) ** (mpmath.mpf(k) / 8))
    # the series regime, where the Taylor grid lives: points at node midpoints
    # (x = 0.05 mod 0.1) are the farthest a grid lookup ever reaches
    for s in ("0.5", "2.5", "7.05", "12.35", "20", "25", "30", "33.3", "40", "45"):
        pts.append(mpmath.mpf(s))
    # each order's switch and 0.1 either side, and the old fixed switch at 50
    for sw in switches:
        for s in (sw - mpmath.mpf("0.1"), sw, sw + mpmath.mpf("0.1")):
            pts.append(s)
    for s in ("49.9", "50", "50.1", "60", "70", "80", "100", "120", "150", "200"):
        pts.append(mpmath.mpf(s))
    # round every x to a double first: the table is F at the double Viv will see
    xs = sorted(set(float(p) for p in pts))
    return xs


def check(xs, F):
    """Three independent checks on the table at full working precision."""
    for i, x in enumerate(xs):
        xm = mpmath.mpf(x)
        ex = mpmath.exp(-xm)
        for n in range(1, NMAX + 1):
            lhs = F[i][n - 1]
            rhs = (2 * xm * F[i][n] + ex) / (2 * n - 1)
            assert abs(lhs - rhs) <= mpmath.mpf("1e-35") * lhs, ("recursion", x, n)
        if x == 0:
            for n in range(NMAX + 1):
                assert F[i][n] == mpmath.mpf(1) / (2 * n + 1)
    # Quadrature is slow, so it checks a spread of points, not all. At x = 200
    # the integrand is a spike of width 0.07, so the interval is cut into 40
    # pieces for mpmath's tanh-sinh rule to resolve it. mpmath stops on an
    # absolute error near 10^-dps, and F_16(200) is 2.8e-26, so at 40 digits it
    # stops at 7e-21 relative; the quadrature runs at 70 digits instead.
    cuts = [mpmath.mpf(k) / 40 for k in range(41)]
    for x in (xs[1], 0.5, 12.35, 45.0, 200.0):
        xm = mpmath.mpf(x)
        i = xs.index(x)
        for n in (0, 5, 16):
            with mpmath.workdps(70):
                q = mpmath.quad(lambda t: t ** (2 * n) * mpmath.exp(-xm * t * t), cuts)
            assert abs(q - F[i][n]) <= mpmath.mpf("1e-30") * F[i][n], ("quad", x, n)


def num(v):
    return repr(float(v))


def main():
    lib = boys_switch_lib()
    switches = boys_switch_mp()
    want = [mpmath.nstr(sw, 3) for sw in switches]
    assert [mpmath.mpf(s) for s in lib] == switches, (
        "boys_switch in lib/hf/boys.viv must be " + ", ".join(want))
    # the continuity test steps one ulp below x_n and takes that ulp as 2^-47
    assert all(32 <= sw < 64 for sw in switches)
    xs = x_points(switches)
    F = [[boys_mp(n, mpmath.mpf(x)) for n in range(NMAX + 1)] for x in xs]
    check(xs, F)

    flat = [v for row in F for v in row]
    chunks = [flat[i:i + CHUNK] for i in range(0, len(flat), CHUNK)]
    nx = len(xs)
    lines = []
    lines.append("###")
    lines.append("post: The Boys function from mpmath, for lib/hf/boys to be held against")
    lines.append("")
    lines.append("Generated by `tools/boys_reference.py` from mpmath at 40 digits. Do not edit.")
    lines.append("")
    lines.append(f"`F_n(x)` for `n = 0..{NMAX}` at {nx} values of `x` from 0 to 200, each rounded")
    lines.append("once to the nearest double. The `x` values include both sides of the")
    lines.append("series/asymptotic switch `x_n` of every order in lib/hf/boys (`x_n` and")
    lines.append("`x_n +- 0.1`) and of the old fixed switch at 50. Before writing, the")
    lines.append("generator recomputes each `x_n` and checks the table against")
    lines.append("mpmath quadrature, the downward recursion and `F_n(0) = 1/(2n+1)`.")
    lines.append("")
    lines.append("`boys_ref_table()` returns `F[ix, n]`; `boys_ref_x()` returns the `x` values;")
    lines.append("`boys_ref_switch()` returns `x_n` for `n = 0..16`.")
    lines.append("###")
    lines.append("")
    lines.append(f"BOYS_REF_NX = {nx}")
    lines.append(f"BOYS_REF_NMAX = {NMAX}")
    lines.append("")
    lines.append("#| The switch `x_n` of each order, recomputed from mpmath.")
    lines.append("def boys_ref_switch() {")
    lines.append("    return [" + ", ".join(num(sw) for sw in switches) + "]")
    lines.append("}")
    lines.append("")
    lines.append("#| The arguments, in increasing order.")
    lines.append("def boys_ref_x() {")
    lines.append("    return [" + ", ".join(num(x) for x in xs) + "]")
    lines.append("}")
    for c, vals in enumerate(chunks):
        lines.append("")
        lines.append(f"#| Values {c * CHUNK} to {c * CHUNK + len(vals) - 1} of the table, flattened by row.")
        lines.append(f"def boys_ref_chunk{c}() {{")
        lines.append("    return [" + ", ".join(num(v) for v in vals) + "]")
        lines.append("}")
    lines.append("")
    lines.append(f"#| The table as `F[ix, n]`, `ix` indexing `boys_ref_x()` and `n = 0..{NMAX}`.")
    lines.append("#| Built when called, so a program that only uses the module pays nothing.")
    lines.append("def boys_ref_table() {")
    lines.append("    ncol = BOYS_REF_NMAX + 1")
    lines.append("    out = zeros(BOYS_REF_NX, ncol)")
    lines.append("    # base: where the current chunk starts in the flattened table")
    lines.append("    base = 0")
    for c, vals in enumerate(chunks):
        lines.append(f"    part{c} = boys_ref_chunk{c}()")
        lines.append(f"    for k to len(part{c}) {{")
        lines.append("        flat = base + k")
        lines.append(f"        out[floor(flat / ncol), mod(flat, ncol)] = part{c}[k]")
        lines.append("    }")
        if c < len(chunks) - 1:
            lines.append(f"    base = base + len(part{c})")
    lines.append("    return out")
    lines.append("}")
    lines.append("")
    lines.append('test "the table falls with the order and with the argument" {')
    lines.append("    boys_rt = boys_ref_table()")
    lines.append("    boys_rx = boys_ref_x()")
    lines.append("    for boys_ri to BOYS_REF_NX {")
    lines.append("        for boys_rn to BOYS_REF_NMAX {")
    lines.append("            if boys_rx[boys_ri] > 0.0 {")
    lines.append("                assert_lt(boys_rt[boys_ri, boys_rn + 1], boys_rt[boys_ri, boys_rn])")
    lines.append("            }")
    lines.append("            if boys_ri > 0 {")
    lines.append("                assert_lt(boys_rt[boys_ri, boys_rn], boys_rt[boys_ri - 1, boys_rn])")
    lines.append("            }")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"boys  nx {nx}  nmax {NMAX}  values {len(flat)}  chunks {len(chunks)}  switch " + " ".join(want))


if __name__ == "__main__":
    main()

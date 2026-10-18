#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""python_cases.py: Viv's Int against Python's int and NumPy's int64.

    python3 tools/ints/python_cases.py          write the case files
    python3 tools/ints/python_cases.py --check  exit 1 if the committed files differ

Writes, under lang/viv/tests/ints/:

  python_exact.viv    one `print("<id>", <expr>)` per case, on Int operands
  python_exact.txt    what Python computes for each (an Int as its digits, a
                      number as Viv's `print` writes a double, a boolean as
                      true/false)
  python_stops.tsv    one case per line: an expression whose exact result
                      leaves 64 bits (or divides by zero), and the first line
                      of the error every engine must stop with

lang/viv/tests/int_enum.rs runs python_exact.viv in the three engines (each
must print python_exact.txt byte for byte) and each python_stops.tsv case as a
program of its own.

Python's int is exact, so it is the reference for every case whose result
fits; NumPy's int64 is checked here to agree on those cases (floor division
and modulo of negative operands, the unsigned helpers on uint64). Where the
exact result does not fit, NumPy wraps around and Viv stops: those cases go to
python_stops.tsv, with Viv's message. The generator is seeded; the files are
the same on every run.
"""

import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "lang", "viv", "tests", "ints")
MIN, MAX = -(2 ** 63), 2 ** 63 - 1
EXACT = 2 ** 53


def fits(x):
    return MIN <= x <= MAX


def lit(n):
    """An Int literal in Viv source: the most negative one is an expression."""
    if n == MIN:
        return "(-9223372036854775807 - 1)"
    return f"({n})" if n < 0 else str(n)


def rt(x):
    """A double at full precision as Viv's errors write it (interp.rs, format_rt):
    the shortest of %.15g, %.16g, %.17g that reads back."""
    for p in (15, 16):
        s = g(x, p)
        if float(s) == x:
            return s
    return g(x, 17)


def g(x, p):
    s = "%.*g" % (p, x)
    if "e" in s:
        m, e = s.split("e")
        e = int(e)
        s = f"{m}e{'-' if e < 0 else '+'}{abs(e):02d}"
    return s


def g6(x):
    """A double as Viv's bare print writes it: %g, six significant digits, but
    a whole number below 10^16 as its digits."""
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 1e16:
        return "%d" % int(x)
    s = "%.6g" % x
    if "e" in s:
        m, e = s.split("e")
        e = int(e)
        s = f"{m}e{'-' if e < 0 else '+'}{abs(e):02d}"
    return s


def main():
    rnd = random.Random(20261001)
    special = [0, 1, -1, 2, -2, 3, -3, 7, -7, 10, -10, 255, 2 ** 31, -(2 ** 31), 2 ** 32 + 1, EXACT, -EXACT, EXACT + 1, -(EXACT + 1),
               3037000499, 3037000500, -3037000500, MAX, MIN, MAX - 1, MIN + 1, 10 ** 18, -(10 ** 18)]
    vals = special + [rnd.randint(-1000, 1000) for _ in range(20)] + [rnd.randint(MIN, MAX) for _ in range(20)] + [rnd.randint(-(2 ** 40), 2 ** 40) for _ in range(20)]
    exact, stops = [], []
    k = 0

    def case(expr, want):
        nonlocal k
        k += 1
        exact.append((f"c{k}", expr, want))

    def stop(expr, msg):
        stops.append((expr, msg))

    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    pairs = [(a, b) for a in special for b in special] + [(rnd.choice(vals), rnd.choice(vals)) for _ in range(300)]
    for a, b in pairs:
        A, B = lit(a), lit(b)
        decl = f"x: Int = {A}\n"
        for sym, f in ops:
            r = f(a, b)
            if fits(r):
                case(f"{A} {sym} y({B})", str(r))
            elif len(stops) < 400 and rnd.random() < 0.15:
                stop(f"{A} {sym} y({B})", f"integer overflow: {a} {sym} {b} does not fit in an Int")
        if b != 0:
            q, m = a // b, a % b
            # NumPy's int64 agrees with Python wherever the quotient fits
            if fits(q):
                assert int(np.floor_divide(np.int64(a), np.int64(b))) == q, (a, b)
                assert int(np.remainder(np.int64(a), np.int64(b))) == m, (a, b)
                case(f"{A} // y({B})", str(q))
                case(f"{A} % y({B})", str(m))
            else:
                stop(f"{A} // y({B})", f"integer overflow: {a} // {b} does not fit in an Int")
        elif rnd.random() < 0.3:
            stop(f"{A} // y({B})", "division by zero in `a // b`")
            stop(f"{A} % y({B})", "division by zero in `a % b`")
        # comparisons, Int with Int and Int with the double nearest b
        case(f"{A} < y({B})", "true" if a < b else "false")
        case(f"{A} == y({B})", "true" if a == b else "false")
        fb = float(b)
        case(f"y({A}) <= {repr(fb)}", "true" if a <= fb else "false")
        case(f"y({A}) > {repr(fb)}", "true" if a > fb else "false")
        case(f"y({A}) == {repr(fb)}", "true" if a == fb else "false")
    for a in vals:
        A = lit(a)
        if fits(-a):
            case(f"-y({A})", str(-a))
            case(f"abs(y({A}))", str(abs(a)))
        else:
            stop(f"-y({A})", f"integer overflow: -({a}) does not fit in an Int")
            stop(f"abs(y({A}))", f"integer overflow: abs({a}) does not fit in an Int")
        if -EXACT <= a <= EXACT:
            case(f"y({A}) * 0.5", g6(a * 0.5))
        else:
            stop(f"y({A}) * 0.5", f"converting {a} to a number would change it; write float(n) to round it")
        case(f"float(y({A}))", g6(float(a)))
        case(f"str(y({A})) + \"|\"", f"{a}|")
        u = a & (2 ** 64 - 1)
        case(f"ustr(y({A}))", str(u))
        for s in [0, 1, 7, 31, 32, 63, 64, 70]:
            ua = np.uint64(u)
            want_shr = 0 if s >= 64 else int(ua >> np.uint64(s))
            want_shl = 0 if s >= 64 else int((ua << np.uint64(s)) & np.uint64(2 ** 64 - 1))
            want_sar = (-1 if a < 0 else 0) if s >= 64 else (a >> s)
            sgn = lambda x: x - 2 ** 64 if x >= 2 ** 63 else x
            case(f"shr(y({A}), y({s}))", str(sgn(want_shr)))
            case(f"shl(y({A}), y({s}))", str(sgn(want_shl)))
            case(f"sar(y({A}), y({s}))", str(want_sar))
    for _ in range(300):
        a, b = rnd.choice(vals), rnd.choice(vals)
        A, B = lit(a), lit(b)
        ua, ub = a & (2 ** 64 - 1), b & (2 ** 64 - 1)
        sgn = lambda x: x - 2 ** 64 if x >= 2 ** 63 else x
        with np.errstate(over="ignore"):
            assert int(np.uint64(ua) * np.uint64(ub)) == (ua * ub) % 2 ** 64
        case(f"wrap_mul(y({A}), y({B}))", str(sgn((ua * ub) % 2 ** 64)))
        case(f"wrap_add(y({A}), y({B}))", str(sgn((ua + ub) % 2 ** 64)))
        case(f"wrap_sub(y({A}), y({B}))", str(sgn((ua - ub) % 2 ** 64)))
        case(f"umulhi(y({A}), y({B}))", str(sgn((ua * ub) >> 64)))
        case(f"ult(y({A}), y({B}))", "true" if ua < ub else "false")
        case(f"band(y({A}), y({B}))", str(a & b))
        case(f"bxor(y({A}), y({B}))", str(a ^ b))
    for a in [0, 1, -1, 2, -2, 3, -3, 10, -7, 2 ** 31, 3037000499]:
        for e in [0, 1, 2, 3, 5, 10, 31, 39, 40, 62, 63, 64]:
            r = a ** e
            if fits(r):
                case(f"y({lit(a)}) ^ {e}", str(r))
            else:
                stop(f"y({lit(a)}) ^ {e}", f"integer overflow: {a} ^ {e} does not fit in an Int")
    stop("y(2) ^ (-1)", "an Int to a negative power is not an Int; write float(n) ^ k")
    stop("shl(y(1), y(-1))", "`shl` takes a shift count of 0 or more, not -1")
    # a number computed where an Int is needed: NaN, an infinity, or too big
    stop("y(round(0.0 / 0.0))", "cannot convert float NaN to integer")
    stop("y(round(1.0 / 0.0))", "cannot convert float infinity to integer")
    stop("y(round(1e300))", f"{rt(1e300)} does not fit in an Int")
    stop("y(round(9223372036854775808.0))", f"{rt(2.0 ** 63)} does not fit in an Int")
    stop("y(round(-9223372036854777856.0))", f"{rt(-9223372036854777856.0)} does not fit in an Int")
    # the most negative double that fits
    case("y(round(-9223372036854775808.0))", str(MIN))
    # widths and formats
    for a in [0, 7, -7, 1234567, MAX, MIN]:
        A = lit(a)
        case(f"fmt(y({A}), \"8d\")", "%8d" % a)
        case(f"fmt(y({A}))", str(a))
    case('fmt(y(12), ".2f")', "12.00")
    case('fmt(y(1234567), ".3e")', "1.235e+06")

    # the cases in defs of a hundred: one function of 15,000 calls takes a C
    # compiler minutes, a hundred small ones seconds
    prog = ["# generated by tools/ints/python_cases.py; do not edit", "", "def y(v: Int) -> Int {", "    return v", "}", ""]
    chunk = 100
    parts = [exact[i:i + chunk] for i in range(0, len(exact), chunk)]
    for n, part in enumerate(parts):
        prog.append(f"def part{n}() {{")
        for cid, expr, _ in part:
            prog.append(f'    print("{cid}", {expr})')
        prog.append("}")
        prog.append("")
    for n in range(len(parts)):
        prog.append(f"part{n}()")
    # one test, so tools/three-engines.sh counts the program as tested; Python's
    # -7 // 2, -7 % 2 and 7 % -2
    title = "floor division and modulo take the divisor's sign, as in Python"
    prog += ["", f'test "{title}" {{', "    a: Int = -7", "    b: Int = 2", "    assert_true(a // b == -4)",
             "    assert_true(a % b == 1)", "    assert_true(7 % (0 - b) == -1)", "}"]
    assert (-7 // 2, -7 % 2, 7 % -2) == (-4, 1, -1)
    txt = [f"{cid} {want}" for cid, _, want in exact] + [f"PASS {title}"]
    tsv = [f"{expr}\t{msg}" for expr, msg in stops if msg]
    files = {
        "python_exact.viv": "\n".join(prog) + "\n",
        "python_exact.txt": "\n".join(txt) + "\n",
        "python_stops.tsv": "\n".join(tsv) + "\n",
    }
    check = "--check" in sys.argv
    bad = False
    for name, text in files.items():
        path = os.path.join(OUT, name)
        if check:
            old = open(path).read() if os.path.exists(path) else None
            if old != text:
                print(f"{path}: differs from what the generator writes", file=sys.stderr)
                bad = True
        else:
            with open(path, "w") as f:
                f.write(text)
    if not check:
        print(f"{len(exact)} exact cases, {len(tsv)} stopping cases")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

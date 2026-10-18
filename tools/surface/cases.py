#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The test cases of Viv's Python-like surface (comprehensions, `a if c else
b`, negative indices, `in`, `key=`, `enumerate(xs, start)`, `zip`, `rows`,
`print`'s separator), each a Viv program and the Python program it stands
for, with the output CPython prints.

    python3 tools/surface/cases.py          # rewrites tools/surface/cases.json
    python3 tools/surface/cases.py --check  # exits 1 if it would change

lang/viv/tests/surface.rs runs every Viv program in the interpreter, as C and
as wasm, and compares what it prints with the output recorded here, byte for
byte. Nothing in the file is written by hand: the output is what this CPython
prints for the Python side. The Python side prints what Viv prints for the
same values, so a boolean is printed through `b()` (`true`, `false`) and a
float through `repr`, where Viv writes `fmt(x)`.

A case that stops has `stops`: the words Viv's error begins with (its own
wording, as NumPy's differs for an index out of range); CPython must raise, and
the output before the stop must match.
"""
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

ROOT = __file__.rsplit("/tools/", 1)[0]
OUT = ROOT + "/tools/surface/cases.json"

PRELUDE = '''
def b(x):
    return "true" if x else "false"
'''

CASES = []


def case(name, viv, py, stops=None):
    CASES.append({"name": name, "viv": viv.strip("\n") + "\n", "py": py.strip("\n") + "\n", "stops": stops})


# ---- comprehensions --------------------------------------------------------------------------

case("comp_map", """
xs = [3, 1, 4, 1, 5, 9, 2, 6]
print([x * x for x in xs])
print([x + 0.5 for x in xs[:3]])
""", """
xs = [3, 1, 4, 1, 5, 9, 2, 6]
print([x * x for x in xs])
print([x + 0.5 for x in xs[:3]])
""")

case("comp_filter", """
xs = [3, 1, 4, 1, 5, 9, 2, 6]
print([x for x in xs if x % 2 == 0])
print([x for x in xs if x > 100], len([x for x in xs if x > 100]))
print([x for x in xs if x > 1 if x < 6])
""", """
xs = [3, 1, 4, 1, 5, 9, 2, 6]
print([x for x in xs if x % 2 == 0])
print([x for x in xs if x > 100], len([x for x in xs if x > 100]))
print([x for x in xs if x > 1 if x < 6])
""")

case("comp_empty_source", """
e = []
print([x + 1 for x in e], len([c for c in ""]))
""", """
e = []
print([x + 1 for x in e], len([c for c in ""]))
""")

case("comp_ranges", """
print([i for i to 5])
print([i for i from 2 to 12 by 3])
print([i for i to 5 reversed])
print([i * j for i to 4 for j to i])
""", """
print([i for i in range(5)])
print([i for i in range(2, 12, 3)])
print([i for i in reversed(range(5))])
print([i * j for i in range(4) for j in range(i)])
""")

case("comp_nested_rows", """
m = [[i * 3 + j for j to 3] for i to 2]
print(m)
print([[x * 10 for x in r] for r in rows(m)])
print([sum(r) for r in rows(m)])
""", """
m = [[i * 3 + j for j in range(3)] for i in range(2)]
print(m)
print([[x * 10 for x in r] for r in m])
print([sum(r) for r in m])
""")

case("comp_strings", """
print([c for c in "hello" if c != "l"])
print([upper(w) for w in split("a bb ccc")])
print([len(w) for w in split("a bb ccc")])
print(join("", [lower(c) for c in "Was It A Car" if isalnum(c)]))
""", """
print([c for c in "hello" if c != "l"])
print([w.upper() for w in "a bb ccc".split()])
print([len(w) for w in "a bb ccc".split()])
print("".join(c.lower() for c in "Was It A Car" if c.isalnum()))
""")

case("comp_enumerate_zip", """
xs = [10, 20, 30]
ys = [1, 2, 3, 4]
print([i * x for i, x in enumerate(xs)])
print([i for i, x in enumerate(xs, 1)])
print([a + b for a, b in zip(xs, ys)])
print([a * b * c for a, b, c in zip(xs, ys, [2, 2])])
""", """
xs = [10, 20, 30]
ys = [1, 2, 3, 4]
print([i * x for i, x in enumerate(xs)])
print([i for i, x in enumerate(xs, 1)])
print([a + b for a, b in zip(xs, ys)])
print([a * b * c for a, b, c in zip(xs, ys, [2, 2])])
""")

case("comp_scope", """
x = 100
k = 3
ys = [x + k for x in [1, 2]]
print(ys, x)
zs = [[x + y for y in [10, 20]] for x in ys]
print(zs, x)
""", """
x = 100
k = 3
ys = [x + k for x in [1, 2]]
print(ys, x)
zs = [[x + y for y in [10, 20]] for x in ys]
print(zs, x)
""")

case("comp_in_def", """
def squares(n) {
    return [i * i for i to n if i % 2 == 1]
}
def count_big(xs, t) {
    return len([x for x in xs if x > t])
}
print(squares(8), count_big([5, 1, 7], 4))
""", """
def squares(n):
    return [i * i for i in range(n) if i % 2 == 1]
def count_big(xs, t):
    return len([x for x in xs if x > t])
print(squares(8), count_big([5, 1, 7], 4))
""")

case("comp_order", """
def say(x) {
    print("say", x)
    return x
}
v = say(1) + sum([say(x) for x in [2, 3]])
print(v)
""", """
def say(x):
    print("say", x)
    return x
v = say(1) + sum([say(x) for x in [2, 3]])
print(v)
""")

case("genexp", """
xs = [3, 1, 4]
print(sum(x * x for x in xs), max(x - 1 for x in xs))
print(all(x > 0 for x in xs), any(x > 3 for x in xs), all(x > 1 for x in xs))
e = []
print(all(x > 0 for x in e), any(x > 0 for x in e))
print(all(i * j < 10 for i to 4 for j to 4), any(i * j == 6 for i to 4 for j to 4))
""", PRELUDE + """
xs = [3, 1, 4]
print(sum(x * x for x in xs), max(x - 1 for x in xs))
print(b(all(x > 0 for x in xs)), b(any(x > 3 for x in xs)), b(all(x > 1 for x in xs)))
e = []
print(b(all(x > 0 for x in e)), b(any(x > 0 for x in e)))
print(b(all(i * j < 10 for i in range(4) for j in range(4))), b(any(i * j == 6 for i in range(4) for j in range(4))))
""")

case("genexp_stops_early", """
def big(x) {
    print("saw", x)
    return x > 1
}
print(all(big(x) for x in [2, 0, 5]))
print(any(big(x) for x in [0, 3, 7]))
""", PRELUDE + """
def big(x):
    print("saw", x)
    return x > 1
print(b(all(big(x) for x in [2, 0, 5])))
print(b(any(big(x) for x in [0, 3, 7])))
""")

# ---- if-expressions --------------------------------------------------------------------------

case("ifexpr", """
for x in [-2, 0, 3] {
    print(x, "neg" if x < 0 else "zero" if x == 0 else "pos", -x if x < 0 else x)
}
y = 1.5 if len("ab") > 1 else 2.5
print(y, [x if x > 1 else 0 for x in [0, 1, 2, 3]])
""", """
for x in [-2, 0, 3]:
    print(x, "neg" if x < 0 else "zero" if x == 0 else "pos", -x if x < 0 else x)
y = 1.5 if len("ab") > 1 else 2.5
print(y, [x if x > 1 else 0 for x in [0, 1, 2, 3]])
""")

case("ifexpr_lazy", """
def say(s) {
    print("ran", s)
    return 1
}
a = say("then") if 2 > 1 else say("else")
b = say("then") if 2 < 1 else say("else")
xs = [1, 2]
n = 0
ok = n > 0 and (xs[5] if n > 1 else 0) > 1
print(ok, a + b)
k = 0
while (xs[k] if k < len(xs) else 0) > 0 {
    k = k + 1
}
print(k)
""", PRELUDE + """
def say(s):
    print("ran", s)
    return 1
a = say("then") if 2 > 1 else say("else")
b_ = say("then") if 2 < 1 else say("else")
xs = [1, 2]
n = 0
ok = n > 0 and (xs[5] if n > 1 else 0) > 1
print(b(ok), a + b_)
k = 0
while (xs[k] if k < len(xs) else 0) > 0:
    k = k + 1
print(k)
""")

# ---- negative indices ------------------------------------------------------------------------

case("negative_index", """
a = [10, 11, 12, 13, 14]
print(a[-1], a[-5], a[:-1], a[-2:], a[-3:-1], a[::-1 * -2])
M = [[1, 2, 3], [4, 5, 6]]
print(M[-1, :], M[:, -1], M[-1, -1], M[-2, 0])
a[-1] = 99
a[-2:] = 7
print(a)
""", """
import numpy as np
a = [10, 11, 12, 13, 14]
print(a[-1], a[-5], a[:-1], a[-2:], a[-3:-1], a[::-1 * -2])
M = np.array([[1, 2, 3], [4, 5, 6]])
print(M[-1, :].tolist(), M[:, -1].tolist(), M[-1, -1], M[-2, 0])
a[-1] = 99
a[-2:] = [7, 7]
print(a)
""")

case("negative_index_past_start", """
a = [10, 11, 12]
print(a[-3])
print(a[-4])
""", """
import numpy as np
a = np.array([10, 11, 12])
print(a[-3])
print(a[-4])
""", stops="error: index -4 out of bounds for dimension 0 (size 3)")

# ---- in ----------------------------------------------------------------------------------------

case("membership", """
a = [1, 2.5, 3]
print(2.5 in a, 4 in a, 4 not in a, 0 / 0 in [0 / 0], -0.0 in [0])
M = [[1, 2], [3, 4]]
print(3 in M, 5 in M)
m = map()
m["x"] = 1
print("x" in m, "y" in m, "y" not in m)
print("b" in ["a", "b"], "ell" in "hello", "z" in "hello")
""", PRELUDE + """
import numpy as np
a = np.array([1, 2.5, 3])
print(b(2.5 in a), b(4 in a), b(4 not in a), b(float("nan") in np.array([float("nan")])), b(-0.0 in np.array([0.0])))
M = np.array([[1, 2], [3, 4]])
print(b(3 in M), b(5 in M))
m = {}
m["x"] = 1
print(b("x" in m), b("y" in m), b("y" not in m))
print(b("b" in ["a", "b"]), b("ell" in "hello"), b("z" in "hello"))
""")

# ---- key= --------------------------------------------------------------------------------------

case("key_max_min", """
xs = [3, -7, 4, -4, 7]
print(max(xs, key=abs), min(xs, key=abs), max(xs, key=fn(v) -> -v), min(xs, key=fn(v) -> v * v))
words = split("pear fig banana kiwi plum")
print(max(words, key=len), min(words, key=len))
def last(w) {
    return w[-1]
}
print(max(words, key=last), min(words, key=fn(w) -> w[-1]))
""", """
xs = [3, -7, 4, -4, 7]
print(max(xs, key=abs), min(xs, key=abs), max(xs, key=lambda v: -v), min(xs, key=lambda v: v * v))
words = "pear fig banana kiwi plum".split()
print(max(words, key=len), min(words, key=len))
def last(w):
    return w[-1]
print(max(words, key=last), min(words, key=lambda w: w[-1]))
""")

case("key_nan", """
nan = 0 / 0
print(max([2, nan, 5], key=fn(v) -> v), min([2, nan, 1], key=fn(v) -> v))
print(fmt(max([nan, 5, 9], key=fn(v) -> v)), fmt(min([nan, 1], key=fn(v) -> v)))
""", """
nan = float("nan")
print(max([2, nan, 5], key=lambda v: v), min([2, nan, 1], key=lambda v: v))
print(repr(max([nan, 5.0, 9.0], key=lambda v: v)), repr(min([nan, 1.0], key=lambda v: v)))
""")

case("key_empty", """
print("before")
print(max([], key=fn(v) -> v))
""", """
print("before")
print(max([], key=lambda v: v))
""", stops="error: max() arg is an empty sequence")

case("key_sorted", """
xs = [5, 3, 8, 1, 9, 2, 3, 5]
print(sorted(xs, key=fn(v) -> v % 3))
print(sorted(xs, key=fn(v) -> v % 3, reverse=true))
print(sorted(xs, reverse=true))
print(argsort(xs, key=fn(v) -> v))
print(argsort(xs, key=fn(v) -> v, reverse=true))
words = split("pear fig banana kiwi plum fig")
print(sorted(words, key=len))
print(sorted(words, key=len, reverse=true))
print(sorted(words, key=fn(w) -> w[-1]))
e = []
print(sorted(e, key=fn(v) -> v), argsort(e, key=fn(v) -> v))
M = [[3, 1], [1, 2], [2, 0]]
print(sorted(rows(M), key=fn(r) -> r[1]))
""", """
xs = [5, 3, 8, 1, 9, 2, 3, 5]
print(sorted(xs, key=lambda v: v % 3))
print(sorted(xs, key=lambda v: v % 3, reverse=True))
print(sorted(xs, reverse=True))
print(sorted(range(len(xs)), key=lambda i: xs[i]))
print(sorted(range(len(xs)), key=lambda i: xs[i], reverse=True))
words = "pear fig banana kiwi plum fig".split()
print(sorted(words, key=len))
print(sorted(words, key=len, reverse=True))
print(sorted(words, key=lambda w: w[-1]))
e = []
print(sorted(e, key=lambda v: v), sorted(range(len(e)), key=lambda i: e[i]))
M = [[3, 1], [1, 2], [2, 0]]
print(sorted(M, key=lambda r: r[1]))
""")

case("key_in_def", """
def pivot(col, k) {
    return max([i for i from k to len(col)], key=fn(i) -> abs(col[i]))
}
def by_len(words) {
    return sorted(words, key=len)
}
print(pivot([1, -7, 3, 7], 0), pivot([1, -7, 3, 7], 2), by_len(split("ccc a bb")))
""", """
def pivot(col, k):
    return max([i for i in range(k, len(col))], key=lambda i: abs(col[i]))
def by_len(words):
    return sorted(words, key=len)
print(pivot([1, -7, 3, 7], 0), pivot([1, -7, 3, 7], 2), by_len("ccc a bb".split()))
""")

case("key_ifexpr", """
xs = [4, -1, 3, -8]
print(sorted(xs, key=fn(v) -> v if v > 0 else -2 * v))
""", """
xs = [4, -1, 3, -8]
print(sorted(xs, key=lambda v: v if v > 0 else -2 * v))
""")

# ---- loops ---------------------------------------------------------------------------------------

case("loop_heads", """
M = [[1, 2], [3, 4], [5, 6]]
for row in rows(M) {
    print(row, sum(row))
}
for i, row in enumerate(rows(M), 1) {
    print(i, row[-1])
}
for i, c in enumerate("abc", 10) {
    print(i, c)
}
for a, w in zip([1, 2, 3], split("x y")) {
    print(a, w)
}
""", """
M = [[1, 2], [3, 4], [5, 6]]
for row in M:
    print(row, sum(row))
for i, row in enumerate(M, 1):
    print(i, row[-1])
for i, c in enumerate("abc", 10):
    print(i, c)
for a, w in zip([1, 2, 3], "x y".split()):
    print(a, w)
""")

# ---- print -----------------------------------------------------------------------------------

case("print_sep", """
x = 1.5
s = "; "
print(1, 2, 3)
print("x =", x)
print("a", "b", sep="")
print("a", 1, x, sep=", ")
print("a", "", "b")
print(x, sep="")
print("a", "b", sep=s)
print()
""", """
x = 1.5
s = "; "
print(1, 2, 3)
print("x =", x)
print("a", "b", sep="")
print("a", 1, x, sep=", ")
print("a", "", "b")
print(x, sep="")
print("a", "b", sep=s)
print()
""")


def run_py(src):
    buf = io.StringIO()
    err = None
    with redirect_stdout(buf):
        try:
            exec(compile(src, "<case>", "exec"), {"__name__": "__case__"})
        except Exception as e:  # noqa: BLE001 -- a case that stops
            err = "%s: %s" % (type(e).__name__, e)
    return buf.getvalue(), err


def build():
    out = []
    for c in CASES:
        text, err = run_py(c["py"])
        if c["stops"] is None and err is not None:
            sys.exit("%s: CPython raised %s" % (c["name"], err))
        if c["stops"] is not None and err is None:
            sys.exit("%s: CPython did not raise" % c["name"])
        e = {"name": c["name"], "viv": c["viv"], "out": text}
        if c["stops"] is not None:
            e["stops"] = c["stops"]
            e["python_error"] = err
        out.append(e)
    return json.dumps({"python": sys.version.split()[0], "cases": out}, indent=1, ensure_ascii=False) + "\n"


def main():
    text = build()
    try:
        old = open(OUT, encoding="utf-8").read()
    except FileNotFoundError:
        old = None
    if "--check" in sys.argv:
        if old != text:
            print("tools/surface/cases.json is out of date: run python3 tools/surface/cases.py")
            return 1
        return 0
    if old != text:
        open(OUT, "w", encoding="utf-8").write(text)
        print("wrote %s (%d cases)" % (OUT, len(CASES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

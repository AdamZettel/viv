#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Generate the string test cases from CPython.

    python3 tools/strings/cases.py          # rewrites the three outputs
    python3 tools/strings/cases.py --check  # exits 1 if any would change

Writes:

- tools/strings/cases.viv: a Viv program that prints one line per case;
- tools/strings/cases.out: what it must print, computed here by CPython
  (lang/viv/tests/strings.rs runs the program in all three engines and
  compares the output with this file byte for byte);
- lang/viv/src/interp/strings_cases.rs: the cases that stop (float, int,
  chr, ord, index, split with an empty separator, ...), with CPython's
  message, for the unit tests of interp/strings.rs.

Every expected value is what this CPython computes; nothing is written by
hand. A number is printed as `fmt(x, "d")` when Python's result is an int and
as `repr(x)` (Python's repr of a float) when it is a float; a boolean as Viv
writes one (`true`, `false`).
"""
import math
import random
import sys
import unicodedata

ROOT = __file__.rsplit("/tools/", 1)[0]
VIV = ROOT + "/tools/strings/cases.viv"
OUT = ROOT + "/tools/strings/cases.out"
RS = ROOT + "/lang/viv/src/interp/strings_cases.rs"

lines_viv = []
lines_out = []
errors = []  # (kind, args, message)


def lit(s):
    """A Viv string literal for s."""
    out = ['"']
    for c in s:
        o = ord(c)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F or (0x80 <= o < 0xA0) or unicodedata.category(c) in ("Zl", "Zp", "Cf", "Co", "Cn", "Cs") or c in "  \x85":
            out.append("\\u{%x}" % o)
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def num(x):
    """A Viv number literal for an int or a float argument."""
    if isinstance(x, bool):
        raise TypeError
    if x < 0:
        return "-" + num(-x)
    if isinstance(x, int):
        return "%d" % x
    return repr(x)


def vstr(x):
    """`str(x)` of a float as Viv writes it: Python's, but a whole number below
    10^16 as its digits, the way Python writes an int (a zero of either sign
    `0`). Not CPython's: the one place
    the cases say what Viv chose."""
    if math.isfinite(x) and x == int(x) and abs(x) < 1e16:
        return "%d" % int(x)
    return str(x)


def show(v):
    """What Viv prints for Python's result v, and the Viv wrapper that prints it."""
    if isinstance(v, bool):
        return ("true" if v else "false"), "%s"
    if isinstance(v, int):
        return str(v), 'fmt(%s, "d")'
    if isinstance(v, float):
        return repr(v), "repr(%s)"
    if isinstance(v, str):
        return repr(v), "repr(%s)"
    if isinstance(v, list):
        return repr(v), "repr(%s)"
    raise TypeError(type(v))


def case(viv_expr, py_thunk, kind=None, args=None):
    try:
        v = py_thunk()
    except (ValueError, IndexError, TypeError, OverflowError) as e:
        if kind is not None:
            errors.append((kind, args, str(e)))
        return
    text, wrap = show(v)
    lines_viv.append("print(%s)" % (wrap % viv_expr))
    lines_out.append(text)


def section(title):
    lines_viv.append("# ---- %s" % title)


# ---- float ----------------------------------------------------------------------

section("float")
FLOATS = [
    "0", "-0", "+0", "0.0", "-0.0", "1", "1.5", "-1.5", "  1.5  ", "\t2\n", "1_000", "1_000.5", "1e1_0", "1_0e5", ".5", "5.", "-.5e-3",
    "1.e3", "1e", "1e+", "1e-", "e5", ".", "", " ", "_1", "1_", "1__0", "1_.5", "1._5", "0x10", "0X1p3", "inf", "-inf", "+inf", "INF", "Infinity",
    "-iNfInItY", "infin", "nan", "-nan", "+NaN", "nan(1)", "1e400", "-1e400", "1e-400", "2.4703282292062327e-324", "2.4703282292062328e-324",
    "4.9406564584124654e-324", "5e-324", "2.2250738585072011e-308", "2.2250738585072014e-308", "1.7976931348623157e308",
    "1.7976931348623158e308", "1.7976931348623159e308", "1.797693134862315807e308", "0.1", "0.2", "0.3", "0.30000000000000004",
    "123456789012345678901234567890", "1e22", "1e23", "9007199254740993", "9007199254740992.5", "1" + "0" * 400, "0." + "0" * 400 + "1",
    "1e999999999999999999", "1e-999999999999999999", "0e999999999999", "١٢.٣", "１２", " 1.5 ", "1.5​",
    "١", "abc", "1.5x", "1 2", "--1", "+-1", "1e5.5", "१२", "12\x00", "0000.000001", "-000123", "1E5", "1e05", "3.14159265358979323846264338327950288",
    "7.2057594037927933e16", "6.02214076e23", "-2.220446049250313e-16", "1.0000000000000002", "0.9999999999999999", "  -0.0  ",
]
rnd = random.Random(20260929)
for _ in range(400):
    x = rnd.choice([rnd.random(), rnd.uniform(-1e6, 1e6), 10 ** rnd.uniform(-320, 308), rnd.getrandbits(64) / 7.0])
    fm = rnd.choice(["%.17g", "%.20e", "%.12f", "%.3e", "%r"])
    FLOATS.append(repr(x) if fm == "%r" else fm % x)
for _ in range(200):
    # random 30-digit decimals near halfway points
    d = "".join(rnd.choice("0123456789") for _ in range(30))
    FLOATS.append("%s.%se%d" % (d[0], d[1:], rnd.randint(-330, 310)))
for s in FLOATS:
    case("float(%s)" % lit(s), lambda s=s: float(s), "float", [s])

# ---- int ------------------------------------------------------------------------

section("int")
INTS = [
    ("0", 10), ("-0", 10), ("42", 10), ("  42  ", 10), ("+42", 10), ("-42", 10), ("1_000", 10), ("1__0", 10), ("_1", 10), ("1_", 10),
    ("007", 10), ("0x1f", 16), ("0X1F", 16), ("1f", 16), ("0x_1f", 16), ("0b101", 2), ("0o17", 8), ("0b102", 2), ("z", 36), ("Z", 36),
    ("10", 2), ("10", 36), ("0x1f", 0), ("0o17", 0), ("0b11", 0), ("017", 0), ("000", 0), ("0", 0), ("12", 0), ("1.5", 10), ("", 10),
    (" ", 10), ("abc", 10), ("abc", 16), ("9007199254740993", 10), ("18446744073709551616", 10), ("123456789012345678901234567890", 10),
    ("f" * 40, 16), ("1" * 300, 10), ("1" + "0" * 308, 10), ("1" + "0" * 309, 10), ("١٢", 10), ("４２", 10), ("4 ", 10),
    ("12", 1), ("12", 37), ("x" * 300, 10), ("9" * 4301, 10), ("1" * 4300, 10), ("7fffffffffffffff", 16), ("1" + "0" * 100, 2), ("-0x10", 16),
    ("0x", 16), ("0b", 0), ("1 2", 10), ("1e5", 10), ("2" * 20, 3), ("z" * 12, 36),
]
for s, b in INTS:
    viv = "int(%s)" % lit(s) if b == 10 else "int(%s, %d)" % (lit(s), b)

    def f(s=s, b=b):
        v = int(s, b)
        return float(v)

    case(viv, f, "int", [s, b])
section("int of a number")
for x in [0.0, -0.0, 2.7, -2.7, -0.5, 1e300, 123456.789]:
    case("int(%s)" % num(x), lambda x=x: float(int(x)))

# ---- split, rsplit, splitlines, join ---------------------------------------------------

section("split")
TEXTS = [
    "", " ", "a", "a b  c", "  a b c  ", "a,b,,c,", ",a,b", "a\tb\nc\x0bd\x0ce\rf", "x　y z w v u", "x​y", "a  b", "aaa", "a||b||",
    "héllo wörld  ñ", "one  two\tthree\n", "\u0085x\u0085",
]
SEPS = [",", "a", "aa", "||", " ", "b", "ö", "x"]
for t in TEXTS:
    case("split(%s)" % lit(t), lambda t=t: t.split())
    case("rsplit(%s)" % lit(t), lambda t=t: t.rsplit())
    for sep in SEPS:
        case("split(%s, %s)" % (lit(t), lit(sep)), lambda t=t, sep=sep: t.split(sep))
        case("rsplit(%s, %s)" % (lit(t), lit(sep)), lambda t=t, sep=sep: t.rsplit(sep))
        for m in [0, 1, 2, -1]:
            case("split(%s, %s, %d)" % (lit(t), lit(sep), m), lambda t=t, sep=sep, m=m: t.split(sep, m))
            case("rsplit(%s, %s, %d)" % (lit(t), lit(sep), m), lambda t=t, sep=sep, m=m: t.rsplit(sep, m))
    case("splitlines(%s)" % lit(t), lambda t=t: t.splitlines())
errors.append(("split", ["a b", ""], "empty separator"))
section("splitlines")
for t in ["a\nb", "a\r\nb", "a\rb", "a\n\nb\n", "\n", "\r\n\r\n", "a\x1cb\x1dc\x1ed\x85e f g\x0bh\x0ci", "x\n\ry", "no end", ""]:
    case("splitlines(%s)" % lit(t), lambda t=t: t.splitlines())
section("join")
for sep in ["", ", ", "-", "é"]:
    for items in [[], ["a"], ["a", "b", "c"], ["", ""], ["héllo", "wörld"]]:
        viv = "join(%s, [%s])" % (lit(sep), ", ".join(lit(x) for x in items)) if items else "join(%s, [])" % lit(sep)
        case(viv, lambda sep=sep, items=items: sep.join(items))

# ---- strip -----------------------------------------------------------------------

section("strip")
for t in ["", "  a  ", "\t\na b\r\n", "　x ", "xxaxx", "abcba", "  ", "é é", "\x1cx\x1f"]:
    for f in ["strip", "lstrip", "rstrip"]:
        case("%s(%s)" % (f, lit(t)), lambda t=t, f=f: getattr(t, f)())
        for chars in ["x", "ab", "", " é", "\t\n"]:
            case("%s(%s, %s)" % (f, lit(t), lit(chars)), lambda t=t, f=f, chars=chars: getattr(t, f)(chars))

# ---- searching -------------------------------------------------------------------

section("find")
HAY = ["hello world", "héllo wörld héllo", "", "aaaa", "abcabcabc", "ñañaña"]
NEEDLES = ["", "l", "lo", "o w", "ö", "héllo", "aa", "abc", "ña", "zz"]
BOUNDS = [None, 0, 1, 3, -1, -3, 20, -20]
for h in HAY:
    for nd in NEEDLES:
        for f in ["find", "rfind", "count", "index", "rindex"]:
            case("%s(%s, %s)" % (f, lit(h), lit(nd)), lambda h=h, nd=nd, f=f: getattr(h, f)(nd), f if f in ("index", "rindex") else None, [h, nd])
        for a in BOUNDS[1:]:
            for e in BOUNDS:
                for f in ["find", "rfind", "count"]:
                    if e is None:
                        case("%s(%s, %s, %s)" % (f, lit(h), lit(nd), num(a)), lambda h=h, nd=nd, f=f, a=a: getattr(h, f)(nd, a))
                    else:
                        case("%s(%s, %s, %s, %s)" % (f, lit(h), lit(nd), num(a), num(e)), lambda h=h, nd=nd, f=f, a=a, e=e: getattr(h, f)(nd, a, e))
        case("startswith(%s, %s)" % (lit(h), lit(nd)), lambda h=h, nd=nd: h.startswith(nd))
        case("endswith(%s, %s)" % (lit(h), lit(nd)), lambda h=h, nd=nd: h.endswith(nd))
        case("%s in %s" % (lit(nd), lit(h)), lambda h=h, nd=nd: nd in h)
        case("%s not in %s" % (lit(nd), lit(h)), lambda h=h, nd=nd: nd not in h)
section("replace")
for h in HAY:
    for old in ["", "l", "lo", "ö", "a", "aa", "ña"]:
        for new in ["", "-", "XYZ", "é"]:
            case("replace(%s, %s, %s)" % (lit(h), lit(old), lit(new)), lambda h=h, old=old, new=new: h.replace(old, new))
            for c in [0, 1, 2, -1]:
                case("replace(%s, %s, %s, %d)" % (lit(h), lit(old), lit(new), c), lambda h=h, old=old, new=new, c=c: h.replace(old, new, c))

# ---- case and predicates -----------------------------------------------------------

section("upper and lower")
mapped = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and (chr(c).upper() != chr(c) or chr(c).lower() != chr(c))]
for k in range(0, len(mapped), 48):
    t = "".join(chr(c) for c in mapped[k:k + 48])
    case("upper(%s)" % lit(t), lambda t=t: t.upper())
    case("lower(%s)" % lit(t), lambda t=t: t.lower())
for t in ["ΑΣ", "ΑΣΑ", "Σ", "ΑΣ'", "ΑΣ.Α", "Α'Σ", "ΑΣ́", "ΑΣ Α", "aΣ1", "ΣΑ", "ΣΣ", "ΑΣΣ", "straße", "İstanbul", "ﬁne", "ǅ", "ŉ", "HELLO wörld"]:
    case("upper(%s)" % lit(t), lambda t=t: t.upper())
    case("lower(%s)" % lit(t), lambda t=t: t.lower())
section("predicates")
PRED_TEXTS = ["", "abc", "ABC", "Abc", "a1", "123", "١٢٣", "½", "²", "Ⅻ", " \t\n", "　", "ǅ", "ǅa", "aǅ", "ß", "ÉCOLE", "école", "a b", "漢字", "٣", "x​",
              "\x00", "~", "­", "é", "\U0001F600", "ǈ", "ⅰ", "ǲ", "𝟗"]
for t in PRED_TEXTS:
    for f in ["isspace", "isalpha", "isalnum", "isdigit", "isdecimal", "isnumeric", "isupper", "islower", "isascii", "isprintable"]:
        case("%s(%s)" % (f, lit(t)), lambda t=t, f=f: getattr(t, f)())
# the classes of single characters: a sample of every plane, and whole blocks
section("predicates of single characters")
props = ["isspace", "isalpha", "isdigit", "isdecimal", "isnumeric", "isupper", "islower", "isprintable"]
rnd = random.Random(7)
sample = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
picks = sorted(set(rnd.sample(sample, 1200)) | set(range(0, 0x250)) | set(range(0x2000, 0x2070)) | set(range(0x3000, 0x3010)))
for c in picks:
    t = chr(c)
    lines_viv.append("c = %s" % lit(t))
    lines_viv.append('print(join(" ", [%s]))' % ", ".join("str(%s(c))" % f for f in props))
    lines_out.append(" ".join("true" if getattr(t, f)() else "false" for f in props))

# ---- repr -------------------------------------------------------------------------

section("repr")
for t in ["", "a", "it's", 'say "hi"', "both ' and \"", "back\\slash", "\t\n\r", "\x00\x07\x1f\x7f", "\x80\x9f\xa0\xad", "é", "​ 　", "\U0001F600",
          "\U000e0001", "퟿￿", "\U0010ffff", "abc\x85def"]:
    case("repr(%s)" % lit(t), lambda t=t: repr(t))
    case("repr(%s)" % lit(t), lambda t=t: repr(t))
case('repr(["a", "it\'s", "é"])', lambda: repr(["a", "it's", "é"]))

# ---- indexing and slicing ------------------------------------------------------------

section("index and slice")
for s in ["héllo wörld", "hello", "", "ñ"]:
    n = len(s)
    for i in range(-n - 2, n + 2):
        case("%s[%s]" % (lit(s), num(i)), lambda s=s, i=i: s[i], "at", [s, i])
    opts = [None, 0, 1, 3, -1, -3, n, n + 5, -n - 5]
    for a in opts:
        for b in opts:
            for st in [None, 1, 2, -1, -2, 3]:
                code = "%s[%s:%s%s]" % (lit(s), "" if a is None else num(a), "" if b is None else num(b), "" if st is None else ":" + num(st))
                case(code, lambda s=s, a=a, b=b, st=st: s[slice(a, b, st)])
section("lists")
W = ["a", "bb", "héllo", "", "z"]
wl = "[%s]" % ", ".join(lit(x) for x in W)
for i in range(-7, 7):
    case("%s[%s]" % (wl, num(i)), lambda i=i: W[i], "list_index", [i])
for a in [None, 1, -2, 9]:
    for b in [None, 2, -1]:
        for st in [None, 1, -1, 2]:
            code = "%s[%s:%s%s]" % (wl, "" if a is None else num(a), "" if b is None else num(b), "" if st is None else ":" + num(st))
            case(code, lambda a=a, b=b, st=st: W[slice(a, b, st)])
case("len(%s)" % wl, lambda: len(W))
case("sorted(%s)" % wl, lambda: sorted(W))
case('sorted(["b", "a", "é", "e", "Z", "", "aa"])', lambda: sorted(["b", "a", "é", "e", "Z", "", "aa"]))
case("%s + [%s]" % (wl, lit("q")), lambda: W + ["q"])
case('%s in %s' % (lit("bb"), wl), lambda: "bb" in W)
case('%s in %s' % (lit("b"), wl), lambda: "b" in W)
case('index(%s, %s)' % (wl, lit("héllo")), lambda: W.index("héllo"))
case('count(["a", "b", "a"], "a")', lambda: ["a", "b", "a"].count("a"))

# ---- building, padding, chr and ord -----------------------------------------------------

section("building")
for s in ["", "ab", "é"]:
    for n in [-1, 0, 1, 3]:
        case("%s * %d" % (lit(s), n), lambda s=s, n=n: s * n)
        case("%d * %s" % (n, lit(s)), lambda s=s, n=n: n * s)
for a in ["", "a", "é", "abc"]:
    for b2 in ["", "b", "é"]:
        case("%s + %s" % (lit(a), lit(b2)), lambda a=a, b2=b2: a + b2)
        for op in ["<", "<=", ">", ">=", "==", "!="]:
            case("%s %s %s" % (lit(a), op, lit(b2)), lambda a=a, b2=b2, op=op: eval("a %s b2" % op))
section("pad")
for s in ["", "a", "ab", "abc", "é", "-42", "+7", "42"]:
    for w in [-1, 0, 1, 2, 5, 6, 7]:
        for f in ["ljust", "rjust", "center"]:
            case("%s(%s, %d)" % (f, lit(s), w), lambda s=s, w=w, f=f: getattr(s, f)(w))
            case("%s(%s, %d, %s)" % (f, lit(s), w, lit("*")), lambda s=s, w=w, f=f: getattr(s, f)(w, "*"))
            case("%s(%s, %d, %s)" % (f, lit(s), w, lit("é")), lambda s=s, w=w, f=f: getattr(s, f)(w, "é"))
        case("zfill(%s, %d)" % (lit(s), w), lambda s=s, w=w: s.zfill(w))
errors.append(("ljust", ["a", 3, "ab"], "The fill character must be exactly one character long"))
section("chr and ord")
for c in [0, 9, 65, 127, 128, 233, 0x3A3, 0xFFFF, 0x10000, 0x1F600, 0x10FFFF]:
    case("chr(%d)" % c, lambda c=c: chr(c))
    case("ord(%s)" % lit(chr(c)), lambda c=c: ord(chr(c)))
for c in [-1, 0x110000]:
    case("chr(%d)" % c, lambda c=c: chr(c), "chr", [c])
for s in ["", "ab"]:
    case("ord(%s)" % lit(s), lambda s=s: ord(s), "ord", [s])

# ---- text of values ------------------------------------------------------------------------

section("str, repr and fmt")
for x in [0.0, -0.0, 1.0, 0.1, 0.1 + 0.2, 1e16, 1e-5, 123456789.0, float("inf"), float("nan"), -2.5]:
    case("str(%s)" % (num(x) if math.isfinite(x) else ("1e400" if x > 0 else "0.0 / 0.0" if x != x else "-1e400")), lambda x=x: vstr(x))
for s in ["", "ab", "é", "héllo"]:
    for spec in ["", "5", "<5", ">5", "^5", "*^6", "é>4", ".1", "8.2", ">8.3s", "s"]:
        case("fmt(%s, %s)" % (lit(s), lit(spec)), lambda s=s, spec=spec: format(s, spec))
section("f-strings")
for x in [1.5, 0.1 + 0.2, -3.0]:
    for spec in [".3f", ".2e", "10.4f", "g", "d"] if True else []:
        if spec == "d":
            case('f"{%s:%s}"' % ("%d" % int(x), spec), lambda x=x: f"{int(x):d}")
        else:
            case('f"x = {%s:%s}!"' % (num(x), spec), lambda x=x, spec=spec: f"x = {x:{spec}}!")
    case('f"{%s}"' % num(x), lambda x=x: vstr(x))
    case('f"{%s!r}"' % num(x), lambda x=x: f"{x!r}")
for s in ["ab", "it's", "é"]:
    case('f"[{%s}] [{%s!r}] [{%s:>5}] [{%s!r:^9}]"' % (lit(s), lit(s), lit(s), lit(s)), lambda s=s: f"[{s}] [{s!r}] [{s:>5}] [{s!r:^9}]")
case('f"{{literal}} {1.0 + 2.0} {"a" + "b"}"', lambda: f"{{literal}} {vstr(1.0 + 2.0)} {'a' + 'b'}")
case('f"{join(", ", ["x", "y"])}"', lambda: ", ".join(["x", "y"]))
case('len("héllo")', lambda: len("héllo"))
case('len("")', lambda: 0)
case('len("\\u{1F600}\\u{301}")', lambda: len("\U0001F600́"))


def rust_str(s):
    out = ['"']
    for c in s:
        o = ord(c)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif 0x20 <= o < 0x7F:
            out.append(c)
        else:
            out.append("\\u{%x}" % o)
    out.append('"')
    return "".join(out)


def rust_arg(a):
    if isinstance(a, str):
        return "A::S(%s)" % rust_str(a)
    return "A::N(%r)" % float(a)


def main():
    viv = "# GENERATED by tools/strings/cases.py from CPython %s: one printed line per case.\n" % sys.version.split()[0]
    viv += "\n".join(lines_viv) + "\n"
    out = "\n".join(lines_out) + "\n"
    rs = "// GENERATED by tools/strings/cases.py from CPython %s: calls that stop,\n" % sys.version.split()[0]
    rs += "// with CPython's message. Do not edit: rerun the script.\n\n"
    rs += "pub enum A {\n    S(&'static str),\n    N(f64),\n}\n\n"
    rs += "pub static ERRORS: &[(&str, &[A], &str)] = &[\n"
    for kind, args, msg in errors:
        rs += "    (%s, &[%s], %s),\n" % (rust_str(kind), ", ".join(rust_arg(a) for a in args), rust_str(msg))
    rs += "];\n"
    if "--check" in sys.argv:
        same = open(VIV, encoding="utf-8").read() == viv and open(OUT, encoding="utf-8").read() == out and open(RS, encoding="utf-8").read() == rs
        sys.exit(0 if same else 1)
    for path, text in [(VIV, viv), (OUT, out), (RS, rs)]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    print("%d cases, %d that stop" % (len(lines_out), len(errors)))


main()

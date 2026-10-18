# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's strings (strs.py, rules/strings.toml): Python's str, and lists of str, translated to
Viv strings exactly, or refused.

    python3 -m pytest -q tools/py2viv/tests/test_py_str.py

The differential tests run CPython and the translation in the Viv interpreter, C and wasm
(diff.py) and require the printed text to be CPython's byte for byte (and every number bit for
bit); a run that raises must stop with Python's exception line. They need a built compiler
(VIV=<path> or target/release/viv) and, for wasm, viv.wasm and node.
"""
import os
import shutil
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import py2viv  # noqa: E402
import diff  # noqa: E402
import strs  # noqa: E402


def tr(src):
    return py2viv.translate(textwrap.dedent(src), "t.py")


def messages(src):
    return [m.splitlines()[0] for m in tr(src)["messages"]]


def have_viv():
    return os.path.exists(diff.default_viv())


class ThreeEngines(unittest.TestCase):
    def run_diff(self, src, name, files=None, raises=None, raw=True):
        """Translate src, run it in CPython and the three engines, and require CPython's text,
        byte for byte (files: {name: text} written beside the program first; raises: the
        exception line Python's run ends with)."""
        if not have_viv():
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        engines = ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())
        src = textwrap.dedent(src)
        res = py2viv.translate(src, name + ".py")
        self.assertEqual(res["messages"], [], "refused:\n" + "\n".join(res["messages"]))
        with tempfile.TemporaryDirectory() as d:
            for fn, text in (files or {}).items():
                with open(os.path.join(d, fn), "w", newline="") as f:
                    f.write(text)
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write(src)
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines)
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])
        if raw:  # (a NumPy array is printed as a list by Viv and read back by diff.py)
            self.assertTrue(rep["raw_identical"], diff.format_report(rep))
        if raises is not None:
            self.assertEqual((rep.get("stop") or {}).get("python"), raises)
        else:
            self.assertIsNone(rep.get("stop"), diff.format_report(rep))
        return rep

    # ------------------------------------------------ values
    def test_literals_and_values(self):
        self.run_diff('''
            a = "plain"
            b = 'single \\'quoted\\' and "double"'
            c = "tab\\there\\nnewline, back\\\\slash, nul-free \\x07 bell, del \\x7f"
            d = "héllo wörld, ß, İ, ΑΣ, 𝔘𝔫𝔦 (non-BMP), e\\u0301 (combining), \\u2028 line sep, \\u00a0 nbsp"
            e = ""
            print(a)
            print(b)
            print(c)
            print(d)
            print(e)
            print(repr(a), repr(b), repr(c), repr(d), repr(e))
            print(len(a), len(b), len(c), len(d), len(e))
            s = a
            s = s + "!"
            print(s, a)
        ''', "literals")

    def test_defs_records_tuples(self):
        self.run_diff('''
            from dataclasses import dataclass

            @dataclass
            class Atom:
                sym: str
                z: int
                x: float = 0.0
                tag: str = "none"

            def label(a):
                return a.sym + str(a.z)

            def split_name(full):
                parts = full.split()
                return parts[0], parts[-1], len(parts)

            def shout(s, times=2, end="!"):
                return (s.upper() + end) * times

            o = Atom("O", 8, 0.117)
            h = Atom(sym="H", z=1, tag="light")
            print(o)
            print(h, label(h), h.sym)
            first, last, n = split_name("  Ada  King Lovelace ")
            print(first, last, n)
            print(shout("hi"), shout("yo", 3), shout("x", end="?"))
            d = {"name": "O", "z": 8, "note": "it's"}
            print(d, d["name"], d["note"] + "!", len(d["name"]))
            names = []
            names.append(o.sym.lower())
            names.append(h.sym.lower() + h.tag)
            print(names)
        ''', "defs")

    # ------------------------------------------------ operators
    def test_operators(self):
        self.run_diff('''
            s = "héllo"
            t = "wörld"
            print(s + " " + t, s * 3, 3 * s, s * 0, s * -2, "" * 5)
            print(s == "héllo", s != t, s < t, "é" > "z", "Z" < "a", "" < "a", "ab" < "abc", "b" >= "abc")
            print("ll" in s, "x" in s, "" in s, "ö" not in t, "é" in "é")
            print(s[0], s[1], s[-1], s[-5], s[4])
            print(s[1:3], s[:2], s[2:], s[:], s[::-1], s[::2], s[1::2], s[-3:], s[:-1])
            print(s[10:], s[-100:2], s[3:1], s[4:0:-1], s[::-2], s[-1:-4:-1], s[100:-100:-1])
            k = 2
            print(s[k], s[k:k + 2], s[-k:], s[:k * 2])
            if s:
                print("nonempty")
            if not "":
                print("empty is false")
            u = s
            u += "!"
            u *= 2
            print(u, s)
            k = 3

            def mode(m):
                if m == "double":
                    return 2
                return 1
            print(k == "3", k != "3", s == None, mode("double"), mode("x"))
        ''', "operators")

    # ------------------------------------------------ the text of values
    def test_str_repr_of_numbers_and_bools(self):
        self.run_diff('''
            vals = [0.0, -0.0, 0.1 + 0.2, 1e16, 1e-7, 1.5e300, 5e-324, 123456789.0, 2.0 ** 52, 1e22, 0.1]
            for v in vals:
                print(str(v) + "|" + repr(v) + "|" + f"{v}|{v!r}|{v:.3e}|{v:g}|{v:12.4f}|{v:>14}|{v:*<14}|{v:^13.2f}")
            inf = float("inf")
            nan = float("nan")
            print(str(inf) + " " + str(-inf) + " " + str(nan) + " " + f"{inf:.2f} {nan:>6}")
            ints = [0, -7, 42, 2 ** 52, -(2 ** 52)]
            for n in ints:
                print(str(n) + "|" + repr(n) + "|" + f"{n}|{n:d}|{n:5d}|{n:>8}|{n:_<6}|{n:.2f}|{n:e}")
            for b in [True, False]:
                print(str(b) + " " + repr(b) + " " + f"{b} {b!r}")
            print(str([1.5, -0.0, 2.0]) + str([1, 2, -3]) + repr(["a", "b'c", "d\\"e"]) + str([[1.0, 2.5], [3.0, 4.0]]))
            print(format(2.5, ".3f") + format(42, ">6") + format("ab", "*^7") + format(0.1) + format(7))
        ''', "text")

    def test_fstrings(self):
        self.run_diff('''
            name = "wörld"
            n = 7
            x = 2.5
            t = f"hello {name}, {n} items at {x:.2f} each: {n * x:.1f} total {{braces}} {name!r} {name:>10}|{name:.2}|{name:*^9}|"
            print(t)
            print(f"{n=} {x=} {name=} {x + 1=:.3f}")
            print(f"{'const':>8}|{'q'!r}|{''}")
            empty = f""
            print(len(empty), f"{empty}|")
            w = f"{name}" + f"-{n}"
            print(w)
        ''', "fstrings")

    def test_format_and_percent(self):
        self.run_diff('''
            n, x, s = 3, 2.5, "ab"
            print("{} and {}".format(n, s))
            print("{0} {1} {0}".format(s, n))
            print("{:>6.2f}|{:<5}|{:^7}|{name}|{name!r}".format(x, n, s, name=s))
            print("%d items, %.2f each, %s" % (n, x, s))
            print("%5d|%-5d|%8.3f|%-8.3f|%e|%g|%s|%r|%%|%.1s" % (n, n, x, x, x, x, s, s, s))
            print("%d" % 2.7, "%d" % -2.7, "%s" % 1.5, "%s" % n, "%i" % True)
            print("%s" % s)
        ''', "formatting")

    # ------------------------------------------------ join
    def test_join(self):
        self.run_diff('''
            words = ["a", "bb", "ccc"]
            v = [1.5, -0.0, 2.0]
            m = [1, 2, 30]
            print(", ".join(words))
            print("".join(words), "-".join("abc"), "|".join([]), " ".join(w.upper() for w in words))
            print(", ".join(str(x) for x in v))
            print(", ".join(map(str, m)), "+".join(map(str, v)))
            print(", ".join([f"{k}={x:.1f}" for k, x in enumerate(v)]))
            print(":".join(w for w in words if len(w) > 1))
            print("\\n".join(words))
            sep = " / "
            print(sep.join(sorted(words, reverse=True)))
        ''', "join")

    # ------------------------------------------------ methods
    def test_methods(self):
        self.run_diff('''
            s = "  Hello, Wörld!\\t\\n"
            print(repr(s.strip()), repr(s.lstrip()), repr(s.rstrip()), repr("xxhixx".strip("x")), repr("abcba".strip("ab")))
            print(repr(" \\u2003 wide \\u00a0".strip()), repr("\\x1c sep \\x1f".strip()), repr("".strip()))
            print(s.split(), "a,b,,c".split(","), "a,b,c".split(",", 1), "a b c".rsplit(" ", 1), "".split(), "".split(","))
            print("  a  b\\u00a0c\\u3000d ".split(), "x".split("x"), "abc".split("abc"))
            print("a\\nb\\r\\nc\\rd\\u2028e\\n".splitlines(), "".splitlines(), "one".splitlines())
            print("Straße".upper(), "ΑΣ".lower(), "İ".lower(), len("İ".lower()), "ǅ".upper(), "ﬃ".upper())
            print("hello".replace("l", "L"), "hello".replace("l", "L", 1), "aaa".replace("", "-"), "abc".replace("x", "y"))
            h = "héllo héllo"
            print(h.find("llo"), h.rfind("llo"), h.find("z"), h.count("l"), h.count(""), h.index("é"), h.rindex("é"))
            print(h.find("l", 5), h.find("l", 3, 5), h.count("l", -4), h.find("llo", -5, -1), "abc".find(""))
            print(h.startswith("hé"), h.startswith(""), h.endswith("lo"), h.endswith("x"))
            print("abc".isalpha(), "ab1".isalnum(), "٣4".isdigit(), "½".isnumeric(), "½".isdigit(), " \\t".isspace(), "".isspace())
            print("ABC".isupper(), "abc".islower(), "Abc".islower(), "é".isascii(), "\\x07".isprintable(), "".isdigit())
            print(repr("ab".ljust(5)), repr("ab".rjust(5, "*")), repr("ab".center(5)), repr("ab".center(6, "-")), repr("abc".center(2)))
            print("-42".zfill(6), "+7".zfill(4), "abc".zfill(2), "".zfill(3))
            print("test_case".removeprefix("test_"), "abc".removeprefix(""), "file.py".removesuffix(".py"), "abc".removesuffix(""))
            print(s.lower().strip().split(",")[0].upper())
        ''', "methods")

    # ------------------------------------------------ numbers from text
    def test_parsing(self):
        self.run_diff('''
            for t in [" 1_000.5 ", "-0", "-0.0", "1e-7", "1e16", "inf", "-Infinity", "nan", "1e400", "4.9e-324",
                      "0.1", "\\u0663.5", " \\u2003 2.5\\n", "+.5", "5."]:
                x = float(t)
                print(repr(t), x, repr(x))
            for t in [" -42 ", "0", "-0", "+7", "1_000", "٣"]:
                print(repr(t), int(t))
            print(int("ff", 16), int("0x1F", 16), int("0b101", 0), int("777", 8), int("z", 36), int("-0o17", 0))
        ''', "parsing")

    # ------------------------------------------------ lists of strings
    def test_lists_of_strings(self):
        self.run_diff('''
            words = []
            for w in "the quick brown fox jumps".split():
                if len(w) > 3:
                    words.append(w)
            print(words, len(words), words[0], words[-1], words[1:], words[::-1])
            print(sorted(words), sorted(words, reverse=True), sorted("hello"), sorted(["b", "a", "B", "é", "z"]))
            print("quick" in words, "the" in words, "the" not in words, words.index("brown"), words.count("fox"))
            print(words == ["quick", "brown", "jumps"], words != ["x"], words + ["!"])
            caps = [w.upper() for w in words]
            print(caps, list("abc"), list(words))
            n = 0
            for w in words:
                n += len(w)
            print(n, str(words), repr(words))
            letters = ["b", "a"]
            letters.append("c")
            letters.sort()
            print(letters)
            letters.reverse()
            letters.insert(1, "x")
            letters.insert(-10, "first")
            letters.insert(100, "last")
            print(letters)
            last = letters.pop()
            first = letters.pop(0)
            print(first, last, letters)
            letters.remove("x")
            letters.extend(["y", "z"])
            letters.extend(w for w in "pq")
            print(letters)
            letters.clear()
            print(letters, len(letters))
            if not letters:
                print("empty")
        ''', "lists")

    def test_iteration(self):
        self.run_diff('''
            s = "héllo"
            for c in s:
                print(c, ord(c))
            for i, c in enumerate(s):
                print(i, c)
            for c in reversed(s):
                print(c)
            for a, b in zip(s, "abc"):
                print(a + b)
            codes = [ord(c) for c in s]
            print(codes, "".join(chr(k + 1) for k in codes), [c * 2 for c in s if c != "l"])
            vowels = 0
            for c in "a quick brown fox":
                if c in "aeiou":
                    vowels += 1
            print(vowels)
            print(chr(233), chr(65), chr(128512), ord("€"))
        ''', "iteration")

    # ------------------------------------------------ print
    def test_print(self):
        self.run_diff('''
            s = "a\\nb"
            n = 3
            print(s)
            print("x", s, n, sep="")
            print(s, n, s, sep=" | ")
            print(n, s)
            print(["a", "b"], s, repr(s))
            print("line1\\nline2", n)
            print("")
            print()
            t = "tab\\there"
            print(t, f"[{t}]")
            w = "x yy z".split()
            print(*w)
            print(*w, sep=", ")
            print(*"".split())
            cr = "a\\r\\nb\\rc\\x0cd\\u2028e"
            print(cr, len(cr))
        ''', "printing")

    # ------------------------------------------------ files
    def test_files(self):
        self.run_diff('''
            from pathlib import Path

            text = open("data.txt").read()
            print(repr(text))
            for line in open("data.txt"):
                print(repr(line), repr(line.strip()))
            lines = open("data.txt").readlines()
            print(lines)
            with open("data.txt") as f:
                total = 0.0
                for line in f:
                    parts = line.split()
                    if parts:
                        total += float(parts[1])
            print(total)
            with open("out.txt", "w") as f:
                f.write("first line\\n")
                for k in range(3):
                    f.write(f"row {k}\\n")
            print(repr(Path("out.txt").read_text()))
            n = Path("out2.txt").write_text("a\\r\\nb\\rc")
            print(n, repr(open("out2.txt").read()), open("out2.txt").readlines())
            with open("out.txt") as f:
                body = f.read()
            print(len(body), body.splitlines())
            print(repr(Path("empty.txt").read_text()), open("empty.txt").readlines())
        ''', "files", files={"data.txt": "x 1.5\r\ny 2.25\n\nz -0.5", "empty.txt": ""})

    # ------------------------------------------------ holes found on the way (RULES.md, Fixed holes)
    def test_fstring_constant_fields(self):
        # fstring-const: a string constant's conversion and spec were dropped in a print
        self.run_diff('''
            print(f"{'ab'!r} {'ab':>5}|{'x':*^5}|{'q'!s}")
        ''', "fconst")

    def test_rows_that_are_numpy_arrays(self):
        # list-of-arrays: a NumPy array put into a list stays one (printed as NumPy prints it)
        self.run_diff('''
            import numpy as np
            b = []
            b.append(np.array([1.5, 2.0]))
            b.append(np.array([3.0, 4.0]))
            print(b[0], b[1] * 2)
            for r in b:
                print(r + 1)
            c = [np.zeros(2) + k for k in range(2)]
            print(c[1])
        ''', "ndrows", raw=False)

    # ------------------------------------------------ stops
    def test_stops(self):
        cases = [
            ('x = float("abc")\nprint(x)\n', "ValueError: could not convert string to float: 'abc'"),
            ('x = int("1.5")\nprint(x)\n', "ValueError: invalid literal for int() with base 10: '1.5'"),
            ('s = "abc"\nprint(s[3])\n', "IndexError: string index out of range"),
            ('s = ["a"]\nprint(s[-2])\n', "IndexError: list index out of range"),
            ('print("abc".index("z"))\n', "ValueError: substring not found"),
            ('print(ord("ab"))\n', "TypeError: ord() expected a character, but string of length 2 found"),
            ('print("a,b".split(""))\n', "ValueError: empty separator"),
            ('print("abc"[::0])\n', "ValueError: slice step cannot be zero"),
            ('w = ["a", "b"]\nprint(w.index("z"))\n', "ValueError: 'z' is not in list"),
            ('w = ["a", "b"]\nw.remove("z")\nprint(w)\n', "ValueError: list.remove(x): x not in list"),
            ('x = 2.5\nraise ValueError(f"bad value {x:.1f} for {\'k\'}")\n', "ValueError: bad value 2.5 for k"),
            ('m = ""\nraise RuntimeError(m)\n', "RuntimeError"),
            ('k = 3\nassert k > 5, "k is " + str(k)\n', "AssertionError: k is 3"),
            ('print("ab".center(5, "--"))\n', "TypeError: The fill character must be exactly one character long"),
            ('print(chr(1114112))\n', "ValueError: chr() arg not in range(0x110000)"),
        ]
        for k, (src, line) in enumerate(cases):
            with self.subTest(src=src):
                self.run_diff(src, "stop%d" % k, raises=line)


    def test_sorted_set_and_unique(self):
        # sorted(set(x)) of text is Viv's unique(x); sorted of a string, its characters
        self.run_diff("""
            TEXT = "hello world! " * 5 + "héllo"
            vocab = sorted(set(TEXT))
            print(vocab, len(vocab), len(set(TEXT)))
            words = "the cat and the hat".split()
            print(sorted(set(words)), sorted(set(words), reverse=True), len(set(words)))
            print(sorted("banana"), sorted("banana", reverse=True), sorted(set("")))
            print("".join(sorted(set("mississippi"))))
        """, "unique")


class Refusals(unittest.TestCase):
    def refused(self, src, what):
        ms = messages(src)
        self.assertTrue(any(what in m for m in ms), "expected %r in %r" % (what, ms))

    def test_refusals(self):
        self.refused('t = ("a", "b")\nprint(t)\n', "a tuple holding a string")
        self.refused('w = ["a"]\nw[0] = "b"\n', "a store into a list of strings")
        self.refused('s = "ab"\ns[0] = "x"\n', "a store into a string")
        self.refused('print("Ab".casefold())\n', "str.casefold")
        self.refused('print("%x" % 255)\n', "%x")
        self.refused('x = 1.5\nprint(f"{x:+.2f}")\n', "the format spec `+.2f`")
        self.refused('print("a b".split(None, 1))\n', "on whitespace with maxsplit")
        self.refused('print("a" + 1)\n', "`+` of a string and a number")
        self.refused('print("a" * 1.5)\n', "`*` of a string and a number")
        self.refused('w = ["b", "a"]\nw += ["c"]\n', "`+=` of a list of strings")
        self.refused('print(sorted(["bb", "a"], key=len))\n', "sorted of strings with key=")
        self.refused('import numpy as np\ns = str(np.zeros(2))\n', "str of a NumPy array")
        self.refused('print("a" < 1)\n', "comparing a string with a number")
        self.refused('print(input())\n', "`input`")
        self.refused('f = open("x.txt", "a")\n', "`open`")
        self.refused('with open("x.txt", "a") as f:\n    f.write("x")\n', "open in the mode 'a'")
        self.refused('with open("x.txt") as f:\n    a = f.read()\n    b = f.read()\n', "a file read more than once")
        self.refused('print("a", end="")\n', "print's end=''")
        self.refused('d = {}\ns = "k"\nd[s] = 1\nd[2] = 1\n', "a dict whose keys are both numbers and strings")
        self.refused('print(sorted(set([3, 1, 3])))\n', "a set of a rank-1 array")
        self.refused('s = set("abc")\nprint(len(s))\n', "`set`")
        self.refused('print(sorted(set("ba"), key=len))\n', "sorted of strings with key=")
        self.refused('s = "ab"\nprint(s[1.0])\n', "an index that is a float")
        self.refused('print(f"{3!a}")\n', "the conversion `!a`")
        self.refused('import numpy as np\nb = []\nb.append(np.zeros(2))\nprint(b)\n', "printing a list of NumPy arrays")
        self.refused('import numpy as np\nb = []\nb.append(np.zeros(2))\nb.append([1.0, 2.0])\n', "rows of another kind")

    def test_host_unicode(self):
        # under a Python whose Unicode data is not Viv's (14.0.0), a literal beyond ASCII is
        # refused and text read at run time is checked to be ASCII
        saved = strs.HOST_UNICODE
        try:
            strs.HOST_UNICODE = "15.0.0"
            self.assertTrue(any("beyond ASCII" in m for m in messages('s = "é"\nprint(s)\n')))
            self.assertEqual(messages('s = "e"\nprint(s)\n'), [])
            v = tr('print(open("a.txt").read())\n')["viv"]
            self.assertIn("py_ascii_text(read_text(", v)
        finally:
            strs.HOST_UNICODE = saved
        self.assertNotIn("py_ascii_text", tr('print(open("a.txt").read())\n')["viv"])


if __name__ == "__main__":
    unittest.main()

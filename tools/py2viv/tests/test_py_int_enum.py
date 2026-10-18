# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's ints, enums and optionals (ints.py): Python ints annotated `int` as Viv Ints,
enum.Enum classes as Viv enums, Optional[T] as Viv's T?, translated exactly or refused.

    python3 -m pytest -q tools/py2viv/tests/test_py_int_enum.py

The differential tests run CPython and the translation in the Viv interpreter, C and wasm
(diff.py) and require CPython's printed text byte for byte (and every number bit for bit); a run
that raises must stop with Python's exception line. An Int that overflows 64 bits stops the run
in every engine (Python's int does not overflow: a refusal at run time). They need a built
compiler (VIV=<path> or target/release/viv) and, for wasm, viv.wasm and node.
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


def tr(src):
    return py2viv.translate(textwrap.dedent(src), "t.py")


def messages(src):
    return [m.splitlines()[0] for m in tr(src)["messages"]]


def have_viv():
    return os.path.exists(diff.default_viv())


def engines():
    return ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())


class ThreeEngines(unittest.TestCase):
    def run_diff(self, src, name, raises=None):
        """Translate src, run it in CPython and the three engines, and require CPython's text,
        byte for byte (raises: the exception line Python's run ends with)."""
        if not have_viv():
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        src = textwrap.dedent(src)
        res = py2viv.translate(src, name + ".py")
        self.assertEqual(res["messages"], [], "refused:\n" + "\n".join(res["messages"]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write(src)
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines())
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])
        self.assertTrue(rep["raw_identical"], diff.format_report(rep))
        if raises is not None:
            self.assertEqual((rep.get("stop") or {}).get("python"), raises)
        else:
            self.assertIsNone(rep.get("stop"), diff.format_report(rep))
        return rep

    # ------------------------------------------------ enums
    def test_enums(self):
        self.run_diff('''
            import enum
            from enum import Enum


            class Color(Enum):
                """Primary colors."""
                RED = 1
                GREEN = 2
                BLUE = 3


            class Phase(enum.Enum):
                SOLID = "s"
                LIQUID = "l"
                GAS = -1


            def phase_at(t: float) -> Phase:
                if t < 0.0:
                    return Phase.SOLID
                if t < 100.0:
                    return Phase.LIQUID
                return Phase.GAS


            def warm(c: Color) -> bool:
                return c == Color.RED or c is Color.GREEN


            def describe(c):
                if c == Color.RED:
                    return "warm"
                elif c != Color.BLUE:
                    return "mild"
                return "cold"


            def paint(c=Color.GREEN, times: int = 2) -> str:
                return str(c) * times


            c = Color.BLUE
            print(c)
            print(c.name, str(c), f"[{c}]", "is " + str(c), f"{c!s}", repr(c.name))
            for t in (-5.0, 20.0, 150.0):
                p = phase_at(t)
                print(t, p, p.name, p == Phase.GAS, p is not Phase.SOLID)
            print(warm(Color.RED), warm(c), describe(Color.GREEN), describe(c))
            best = Color.RED
            for k in range(3):
                if k == 2:
                    best = Color.GREEN
            print(best, best == Color.GREEN, best != Color.GREEN)
            names = [Color.RED.name, Color.BLUE.name]
            print(names, len(Color.GREEN.name))
            print(paint(), paint(Color.RED, 1), c, sep=" | ")

            if __name__ == "__main__":
                class Mode(Enum):
                    ON = "on"
                    OFF = "off"
                m = Mode.OFF
                print(m, m == Mode.OFF)
        ''', "enums")

    # ------------------------------------------------ ints
    def test_ints(self):
        self.run_diff('''
            def fact(n: int) -> int:
                if n <= 1:
                    return 1
                return n * fact(n - 1)


            def collatz_steps(n: int) -> int:
                steps: int = 0
                while n != 1:
                    if n % 2 == 0:
                        n = n // 2
                    else:
                        n = 3 * n + 1
                    steps += 1
                return steps


            def mean_of(xs: list[int]) -> float:
                total: int = 0
                for x in xs:
                    total += x
                return total / len(xs)


            def gcd(a: int, b: int) -> int:
                while b:
                    a, b = b, a % b
                return abs(a)


            def ident(n: int) -> int:
                return n


            # a Bool for an int parameter: its clone keeps the translation of a Bool (True prints True)
            print(ident(5), ident(True))
            for k in (5, 20):
                print(k, fact(k))
            big: int = fact(20)
            print(big, big // 7, big % 1000, -big // 7, -big % 1000, big // -3, big % -7)
            print(collatz_steps(27), collatz_steps(97))
            print(mean_of([3, 4, 8]), gcd(1071, 462), gcd(-48, 18), gcd(0, 5))
            m: int = 3037000499
            print(m * m, m - 2 * m, -m, abs(-m), min(m, 5), max(m, -m), m ** 2, m ** 0)
            print(m / 7, m * 0.5, m + 0.25, m < 3.5e9, m == 3037000499, m != 3037000499.0)
            print(float(m), int(m), round(m), str(m), len(str(big)), float(big))
            print(f"{m:15d}|{m:>15}|{m:<12}|{m:.3e}|{m:.1f}|{-m}")
            xs = [10, 20, 30, 40]
            i: int = 2
            print(xs[i], xs[i - 1] + xs[-i], i in (1, 2), i not in (5, 6))
            if i:
                print("nonzero")
            j: int = len(xs)
            j -= 1
            j *= -2
            print(j, j // 3, j % 3, j // -3, j % -3)
            s = 0
            for q in range(4):
                s = s + i * q
            print(s)
        ''', "ints")

    def test_int_floor_rules(self):
        # Python's // floors and % takes the divisor's sign, for every sign; Viv's Int follows
        self.run_diff('''
            def fl(a: int, b: int) -> int:
                return a // b


            def md(a: int, b: int) -> int:
                return a % b


            for a in (-7, -6, -1, 0, 1, 6, 7):
                for b in (-3, -2, 2, 3):
                    print(a, b, fl(a, b), md(a, b), a // b == fl(a, b), a % b == md(a, b))
            big: int = 9007199254740993
            print(big // 2, big % 2, big // -10, big % -10, -big // 10, big - 1, big + 1)
        ''', "floor_rules")

    def test_zero_division(self):
        src = '''
            def q(a: int, b: int):
                return a %s b


            print(q(7, 2))
            print(q(1, 0))
        '''
        for op, which in (("//", "floordiv_int"), ("%", "mod_int"), ("/", "div_int")):
            with self.subTest(op=op):
                self.run_diff(src % op, "zero", raises=py2viv.HOST_ERRORS[which])

    # ------------------------------------------------ optionals
    def test_optionals(self):
        self.run_diff('''
            from enum import Enum
            from typing import Optional, Union


            class Level(Enum):
                LOW = 1
                HIGH = 2


            def find(xs: list[int], want: int) -> Optional[int]:
                for k in range(len(xs)):
                    if xs[k] == want:
                        return k
                return None


            def first_neg(xs: list[float]) -> float | None:
                for x in xs:
                    if x < 0.0:
                        return x


            def greet(name: Union[str, None] = None) -> str:
                if name is None:
                    return "hello, nobody"
                else:
                    return "hello, " + name


            def scale(x: Optional[float], k: float) -> float:
                if x is not None:
                    return x * k
                return 0.0


            def classify(x: float) -> Optional[Level]:
                if x < 0.0:
                    return None
                if x < 0.5:
                    return Level.LOW
                return Level.HIGH


            def deepest(n: int) -> Optional[int]:
                if n == 0:
                    return 0
                if n < 0:
                    return None
                return deepest(n - 2)


            xs = [4, 8, 15, 16, 23, 42]
            for w in (15, 99):
                r = find(xs, w)
                if r is None:
                    print(w, "missing")
                else:
                    print(w, "at", r, "next", xs[r + 1])
            neg = first_neg([1.5, -2.25, 3.0])
            if neg is not None:
                print("first negative", neg)
            nn = first_neg([1.0, 2.0])
            if nn is None:
                print("no negative")
            print(greet(), greet("Ada"), scale(2.5, 4.0), scale(None, 4.0))
            best: Optional[int] = None
            for v in [7, 3, 9, 3, 1]:
                if best is None:
                    best = v
                elif v < best:
                    best = v
            if best is not None:
                print("min", best)
            tag: int | None = None
            count: int = 0
            for v in [5, 6, 7]:
                count += v
                if count > 10:
                    tag = count
                    break
            if tag is not None:
                print("crossed at", tag)
            else:
                print("never crossed")
            for x in (-1.0, 0.25, 0.75):
                lv = classify(x)
                if lv is None:
                    print(x, "none")
                elif lv == Level.HIGH:
                    print(x, lv, "high")
                else:
                    print(x, lv)
            for n in (4, 5):
                d = deepest(n)
                if d is not None:
                    print(n, "reaches", d)
                else:
                    print(n, "skips zero")
        ''', "optionals")

    # ------------------------------------------------ an Int past 64 bits: a refusal at run time
    def test_overflow_stops(self):
        if not have_viv():
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        res = tr('''
            def square(n: int) -> int:
                return n * n


            print(square(3037000499))
            print(square(3037000500))
        ''')
        self.assertEqual(res["messages"], [])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.viv")
            with open(path, "w") as f:
                f.write(res["viv"])
            eng = diff.run_engines(path, diff.default_viv(), diff.default_wasm(), d, engines(), 300)
        for name, r in eng.items():
            with self.subTest(engine=name):
                self.assertNotEqual(r["code"], 0)
                self.assertIn("integer overflow: 3037000500 * 3037000500 does not fit in an Int", r["err"])
                if name != "wasm":  # (node's module reports the error line only)
                    self.assertEqual(r["out"].strip(), "9223372030926249001")


class Translation(unittest.TestCase):
    def test_viv_forms(self):
        v = tr('''
            from enum import Enum
            from typing import Optional


            class C(Enum):
                A = 1
                B = 2


            def f(n: int, c: C, x: Optional[int] = None) -> Optional[int]:
                if x is None:
                    return n
                return x


            y: int = 3
            r = f(y, C.A)
        ''')["viv"]
        self.assertIn("enum C { A, B }", v)
        self.assertIn("def f(n: Int, c: C, x: Int?) -> Int? {", v)
        self.assertIn("match x {", v)
        self.assertIn("return Some(n)", v)
        self.assertIn("y: Int = 3", v)
        self.assertIn("r: Int? = f(y, C.A, None)", v)

    def test_unannotated_ints_are_unchanged(self):
        v = tr('''
            def f(n):
                return n * 2 + 1


            k = 3
            print(f(k), k // 2)
        ''')["viv"]
        self.assertNotIn("Int", v)
        self.assertNotIn("py_ifloordiv", v)


class Refusals(unittest.TestCase):
    E = "from enum import Enum\nclass C(Enum):\n    A = 1\n    B = 2\n"
    O = "from typing import Optional\n"

    def refused(self, src, what):
        ms = messages(src)
        self.assertTrue(any(what in m for m in ms), "expected %r in %r" % (what, ms))

    def test_enum_refusals(self):
        E = self.E
        self.refused(E + "print(C.A.value)\n", "the `.value` of an enum member")
        self.refused("from enum import Enum, auto\nclass C(Enum):\n    A = auto()\n", "`auto()`")
        self.refused(E + "print(C(1))\n", "looking up a member of the enum `C` by value")
        self.refused(E + "print(C['A'])\n", "looking up a member of the enum `C` by name")
        self.refused(E + "for c in C:\n    print(c)\n", "the enum class `C` used as a value")
        self.refused(E + "print(len(C))\n", "the enum class `C` used as a value")
        self.refused("from enum import Enum\nclass C(Enum):\n    A = 1\n    def f(self):\n        return 1\n",
                     "a method in an enum")
        self.refused("from enum import IntEnum\nclass C(IntEnum):\n    A = 1\n", "an IntEnum")
        self.refused("import enum\nclass C(enum.Flag):\n    A = 1\n", "a Flag")
        self.refused("from enum import Enum\nclass C(Enum):\n    A = 1\n    B = 1\n", "an alias")
        self.refused("from enum import Enum\nclass C(Enum):\n    A = 1.5\n", "not an int or str constant")
        self.refused("from enum import Enum\nclass C(Enum):\n    _A = 1\n", "a leading underscore")
        self.refused("from enum import Enum\nclass C(Enum):\n    to = 1\n", "a Viv keyword or type name")
        self.refused(E + "print(C.A + 1)\n", "a member of the enum `C` used here")
        self.refused(E + "xs = [C.A, C.B]\n", "a member of the enum `C` used here")
        self.refused(E + "d = {C.A: 1}\n", "a member of the enum `C` used here")
        self.refused(E + "c = C.A\nif c:\n    print(1)\n", "the truth value of a member")
        self.refused(E + "print(repr(C.A))\n", "repr of an enum member")
        self.refused(E + "print(C.A == 1)\n", "comparing a member of the enum `C` with a number")
        self.refused(E + "print(C.A < C.B)\n", "ordering enum members")
        self.refused(E + "print(f'{C.A:>8}')\n", "a format spec on an enum member")

    def test_int_refusals(self):
        self.refused("def f(n: int) -> int:\n    return n\nprint(f(2.5))\n",
                     "`f` called with a float for `n`, which is annotated `int`")
        self.refused("def f(n: int) -> int:\n    return n / 2\nprint(f(3))\n", "a float where `f`'s result is annotated `-> int`")
        self.refused("x: int = 2.5\n", "a float where `x` is annotated `int`")
        self.refused("def f(n: int) -> int:\n    if n > 0:\n        return n\nprint(f(1))\n",
                     "`f`, annotated `-> int`, can reach its end without `return`")
        self.refused("def f(n: int) -> int:\n    return None\nprint(f(1))\n", "returning None from `f`")
        self.refused("def f(n: int) -> int:\n    return n\ng = [f]\n", "used as a value")
        self.refused("i: int = 0\nfor i in range(3):\n    print(i)\n", "`i`, which holds an Int")

    def test_optional_refusals(self):
        O = self.O
        self.refused(O + "def f(x: Optional[int]) -> int:\n    return x + 1\nprint(f(1))\n",
                     "an Optional[int] value used here (it may be None)")
        self.refused(O + "x: Optional[int] = None\nprint(x)\n", "printing an Optional[int]")
        self.refused(O + "x: Optional[int] = None\nb = x is None\n", "an `is None` test of an Optional[int]")
        self.refused(O + "def f(x: Optional[int]) -> int:\n    if x is not None and x > 2:\n        return x\n    return 0\nprint(f(3))\n",
                     "an `is None` test of an Optional[int]")
        # the narrowing continues after the if: the use after it is refused
        self.refused(O + "def f(x: Optional[int]) -> int:\n    if x is None:\n        return 0\n    return x + 1\nprint(f(2))\n",
                     "an Optional[int] value used here")
        self.refused(O + "def f(x: Optional[int]) -> int:\n    if x is not None:\n        x = x + 1\n        return x\n    return 0\nprint(f(2))\n",
                     "`x` read after it is assigned where it was known not to be None")
        self.refused(O + "def f(x: Optional[int]) -> int:\n    if x is not None:\n        for k in range(2):\n            x = k\n    return 0\nprint(f(2))\n",
                     "`x` assigned in a loop where it is known not to be None")
        self.refused(O + "def f(x: Optional[float] = None) -> float:\n    if x is None:\n        return 0.0\n    else:\n        return x\nprint(f(3))\n",
                     "`f` called with an int for `x`, which is annotated `Optional[float]`")
        self.refused(O + "x: Optional[int] = None\nif x:\n    print(1)\n", "the truth value of an Optional[int]")


if __name__ == "__main__":
    unittest.main()

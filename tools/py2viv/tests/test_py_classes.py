# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's classes, dataclasses, named tuples and dicts (classes.py): each translates to a Viv
record exactly, or is refused where Python's reference semantics could show.

    python3 -m pytest -q tools/py2viv/tests/test_py_classes.py

The differential tests run CPython with NumPy and the translation in the Viv interpreter, C
and wasm (diff.py), and compare every printed value bit for bit; they need a built compiler
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


def tr(src):
    return py2viv.translate(textwrap.dedent(src), "t.py")


def messages(src):
    return [m.splitlines()[0] for m in tr(src)["messages"]]


def have_viv():
    return os.path.exists(diff.default_viv())


class ThreeEngines(unittest.TestCase):
    def run_diff(self, src, name):
        if not have_viv():
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        engines = ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())
        res = tr("import numpy as np\n" + textwrap.dedent(src))
        self.assertEqual(res["messages"], [], "refused:\n" + "\n".join(res["messages"]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write("import numpy as np\n" + textwrap.dedent(src))
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines)
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])

    # ------------------------------------------------ classes whose fields are never rebound
    def test_record_classes(self):
        self.run_diff("""
            class Particle:
                \"\"\"A point mass.\"\"\"
                G = 9.81

                def __init__(self, x, v, m=1.0):
                    self.x = x
                    self.v = v
                    self.m = m
                    self.p = m * v

                def energy(self):
                    return 0.5 * self.m * self.v * self.v + self.m * self.G * self.x

                @property
                def speed(self):
                    return abs(self.v)

                @staticmethod
                def unit():
                    return Particle(0.0, 1.0)

                @classmethod
                def at_rest(cls, x):
                    return cls(x, 0.0)

            def total(a, b):
                return a.energy() + b.energy()

            p = Particle(1.5, -2.0)
            q = Particle(0.5, 3.0, m=2.0)
            u = Particle.unit()
            r = Particle.at_rest(-0.0)
            print(p.energy(), q.speed, total(p, q), Particle.G, p.G, u.v, r.x, r.p)
            print(f"{p.x:.3f} {q.m} {p.speed:g}")
            best = p if p.energy() > q.energy() else q
            print(best.x)
        """, "t_records")

    def test_nested_records_and_array_fields(self):
        self.run_diff("""
            class Vec:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def dot(self, o):
                    return self.x * o.x + self.y * o.y

            class Mesh:
                def __init__(self, n):
                    self.n = n
                    self.xs = np.linspace(0.0, 1.0, n)
                    self.u = np.zeros(n)
                    self.origin = Vec(0.5, -1.0)

                def fill(self, k):
                    for i in range(self.n):
                        self.u[i] = k * self.xs[i]

            m = Mesh(5)
            m.fill(2.0)
            alias = m
            alias.u[0] = -7.0
            w = m.u
            w[1] = 3.5
            print(m.u, alias.u, m.origin.dot(Vec(2.0, 1.0)), m.origin.x)
            a = np.array([1.0, 2.0, 3.0])
            class Holder:
                def __init__(self, v):
                    self.v = v
            h = Holder(a)
            a[0] = 10.0
            a += 1.0
            h.v[2] = -1.0
            print(a, h.v)
        """, "t_nested")

    def test_dataclasses_and_named_tuples(self):
        self.run_diff("""
            from dataclasses import dataclass, replace
            from collections import namedtuple
            from typing import NamedTuple

            @dataclass
            class Box:
                lo: float
                hi: float
                n: int = 10
                closed: bool = True

                def width(self):
                    return self.hi - self.lo

            @dataclass(frozen=True)
            class Cfg:
                dt: float
                steps: int

            Point = namedtuple("Point", ["x", "y"])
            Pair = namedtuple("P2", "a b", defaults=[0.5])

            class Seg(NamedTuple):
                a: float
                b: float = 1.0

                def length(self):
                    return self.b - self.a

            b = Box(0.0, 2.5)
            c = Cfg(0.01, 100)
            c2 = replace(c, steps=5)
            p = Point(1.5, 2)
            x, y = p
            s = Seg(0.25)
            print(b, b.width(), c, c2, c2.dt)
            print(p, x, y, p[0], p[-1], len(p), Pair(1.0), s, s.length(), s._replace(b=4.0))
            print(f"{b} and {p!r}")
        """, "t_dataclasses")

    def test_dataclass_defaults_are_record_field_defaults(self):
        # a literal default of a dataclass or named tuple is the record
        # field's default, and the constructor def declares its record as its return type
        src = """
            from dataclasses import dataclass

            @dataclass
            class Box:
                lo: float
                n: int = 10
                scale: float = -0.5
                closed: bool = True

            b = Box(0.0)
            print(b.n, b.scale, b.closed)
        """
        out = tr(src)
        self.assertEqual(out["messages"], [])
        self.assertIn("n: Num = 10", out["viv"])
        self.assertIn("scale: Num = -0.5", out["viv"])
        self.assertIn("closed: Bool = true", out["viv"])
        self.assertIn("-> Box {", out["viv"])
        self.run_diff(src, "t_dataclass_defaults")

    def test_generic_classes(self):
        # type parameters are annotations only
        self.run_diff("""
            from typing import Generic, TypeVar
            T = TypeVar("T")

            class Box(Generic[T]):
                def __init__(self, v: T):
                    self.v = v

                def get(self) -> T:
                    return self.v

            print(Box(2.5).get(), Box(-1.0).v)
        """, "t_generic")

    def test_str_and_repr_through_fstrings(self):
        self.run_diff("""
            class Q:
                def __init__(self, a, n):
                    self.a = a
                    self.n = n

                def __str__(self):
                    return f"Q[a={self.a:.2f}, n={self.n}]"

                def __repr__(self):
                    return f"Q({self.a!r}, {self.n})"

            q = Q(1.0 / 3.0, 4)
            print(q)
            print(repr(q), f"<{q}>", f"{q!r}")
        """, "t_strrepr")

    # ------------------------------------------------ fields rebound after construction
    def test_state_threading(self):
        self.run_diff("""
            class Counter:
                def __init__(self, start):
                    self.n = start
                    self.total = 0.0
                    self.hist = np.zeros(4)

                def add(self, x):
                    self.n += 1
                    self.total = self.total + x
                    self.hist[self.n % 4] += x

                def mean(self):
                    return self.total / self.n

                def bump(self, k):
                    self.add(k)
                    if k > 10.0:
                        return -1
                    return self.n * 2

                def pair(self):
                    self.n -= 1
                    return self.n, self.total

            def run(c, steps):
                for i in range(steps):
                    c.add(0.5 * i)
                return c.mean()

            def make(k):
                c = Counter(k)
                c.add(1.0)
                return c

            c = Counter(1)
            c.add(2.5)
            c.add(-1.0)
            print(c.n, c.total, c.mean(), c.hist)
            m = run(c, 3)
            print(m, c.n, c.total)
            r = c.bump(4.0)
            r2 = c.bump(11.0)
            n, t = c.pair()
            print(r, r2, c.n, n, t)
            c.total = 100.0
            c.hist += 1.0
            print(c.total, c.hist)
            d = make(3)
            for i in range(2):
                if i > 0:
                    d.add(float(i))
            print(d.n, d.total, c.n)
        """, "t_threading")

    def test_array_field_in_place_through_an_alias(self):
        self.run_diff("""
            class Grid:
                def __init__(self, u):
                    self.u = u
                    self.steps = 0

                def relax(self):
                    self.u[1:-1] = 0.5 * (self.u[:-2] + self.u[2:])
                    self.steps += 1

            u0 = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
            g = Grid(u0)
            g.relax()
            g.relax()
            g.u += 0.25
            print(g.u, u0, g.steps)
        """, "t_inplace")

    def test_threading_forms(self):
        self.run_diff("""
            from dataclasses import dataclass

            class W:
                def __init__(self):
                    self.a = 1.0
                    self.b = 0.0
                    self.last = 0.0

                def step(self, h):
                    self.a, self.b = self.a - h * self.b, self.b + h * self.a
                    return self.a * self.b

                def go(self):
                    self.last = self.step(0.25)

            def run(k):
                w = W()
                for i in range(k):
                    w.step(0.1)
                return w.step(0.1)

            @dataclass
            class Grid:
                n: int
                length: float

                def __post_init__(self):
                    if self.n <= 1:
                        raise ValueError("n must be above 1")
                    self.h = self.length / (self.n - 1)

            class Hidden:
                def __init__(self, x):
                    self.__x = x

                def __double(self):
                    return 2 * self.__x

                def get(self):
                    return self.__double()

            w = W()
            w.go()
            w.go()
            g5 = Grid(5, 2.0)
            print(w.a, w.b, w.last, run(3), g5, Grid(3, 1.0).h, Hidden(1.5).get())
            g = Grid(1, 2.0)
        """, "t_forms")

    # ------------------------------------------------ dicts with literal keys
    def test_dicts(self):
        self.run_diff("""
            params = {"alpha": 1.5, "n": 3, "on": True, "grid": np.array([0.0, 0.5])}
            units = {1: 2.0, 2: 4.5}
            nested = {"a": {"x": 1, "y": -2.0}, "b": 0.25}

            def energy(p, s):
                return p["alpha"] * p["n"] * s

            print(params["alpha"] * 2, len(params), "n" in params, "z" in params, "z" not in params)
            print(units, units[2], nested, nested["a"]["y"], energy(params, 2.0))
            print(params.get("alpha"), params.get("beta", -1.0), units.get(1, 0.0))
            for k, v in units.items():
                print(k, v)
            for k in nested:
                print(k)
            for v in nested["a"].values():
                print(v)
            cfg = dict(dt=0.1, steps=4)
            cfg["dt"] = cfg["dt"] / 2
            cfg["steps"] += 1
            units[1] = -3.0
            params["grid"][1] = 7.0
            print(cfg, units, params["grid"])
            c2 = cfg.copy()
            c2["dt"] = 9.0
            print(cfg["dt"], c2["dt"])
        """, "t_dicts")

    # ------------------------------------------------ holes fixed on the way
    def test_defaults_are_evaluated_once_where_the_def_is(self):
        self.run_diff("""
            x = 1.0
            def f(a, b=x):
                return a + b
            def g():
                x = 5.0
                return f(1.0)
            def h(a, acc=np.zeros(2)):
                acc[0] += a
                return acc[0]
            x = 2.0
            print(g(), f(0.5), h(1.0), h(1.0), h(2.5, np.ones(2)))
        """, "t_defaults")


class Refusals(unittest.TestCase):
    def refused_with(self, src, want):
        m = messages(src)
        self.assertTrue(m, "translated; expected: " + want)
        self.assertTrue(any(want in x for x in m), "\n".join(m) + "\nexpected: " + want)

    def test_aliasing_an_object_whose_fields_are_rebound(self):
        pre = "class C:\n    def __init__(self, x):\n        self.x = x\n    def inc(self):\n        self.x += 1.0\n        return self.x\n"
        cases = {
            "p = C(1.0)\nq = p\nq.inc()\nprint(p.x)\n": "a second name for an object of class `C`",
            "p = C(1.0)\nq = p if p.x > 0 else C(2.0)\n": "used here, whose fields are rebound",
            "def f(p):\n    return p\np = C(1.0)\nq = f(p)\n": "returning an object of class `C` `p` that the caller also holds",
            "p = C(1.0)\ny = p.inc() + 1.0\n": "inside an expression",
            "p = C(1.0)\nprint(p.inc())\n": "inside an expression",
            "p = C(1.0)\ndef g():\n    p.inc()\n    return 0\n": "a change to the module's object `p`",
            "def two(a, b):\n    a.inc()\n    b.inc()\n    return 0\np = C(1.0)\ntwo(p, p)\n": "the same object passed twice",
            "p = C(1.0)\nC(0.0).inc()\n": "with an expression there",
            "class D:\n    def __init__(self, c):\n        self.c = c\np = C(1.0)\nd = D(p)\n": "a second name for an object of class `C`",
            "p = C(1.0)\nL = [p, p]\n": "a list of objects",
            "def show():\n    return p.x\np = C(1.0)\np.inc()\n": "which a def reads as a module variable",
            "p = C(1.0)\nprint(p.__class__)\n": "the attribute `__class__`",
        }
        for body, want in cases.items():
            with self.subTest(body=body):
                self.refused_with(pre + body, want)

    def test_classes_that_do_not_translate(self):
        cases = {
            "class A:\n    pass\nclass B(A):\n    pass\n": "a class that inherits from `A`",
            "class A:\n    def __init__(self, x):\n        if x > 0:\n            self.y = x\n": "first set inside a block",
            "class A:\n    def __init__(self, x):\n        self.y = self.z\n        self.z = x\n": "read before it is set",
            "class A:\n    def __init__(self):\n        self.x = 1.0\n    def m(self):\n        self.w = 2.0\na = A()\na.m()\n":
                "a new attribute `w`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\n        self.m()\n    def m(self):\n        return 0\n":
                "`self` used inside `__init__` other than to set or read an attribute",
            "class A:\n    def __init__(self):\n        self.next = None\n": "`None` as a value",
            "class A:\n    def __init__(self):\n        self.x = 1.0\na = A()\nprint(a)\n": "printing an object of class `A`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\na = A()\nb = A()\nprint(a == b)\n": "comparing an object of class `A`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\n    def __add__(self, o):\n        return A()\na = A() + A()\n":
                "an operator on an object of class `A`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\n    def __getattr__(self, k):\n        return 0\n": "`__getattr__`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\na = A()\nprint(a.y)\n": "the attribute `y`, which an object of class `A` does not have",
            "class A:\n    def __init__(self):\n        self.x = 1.0\na = A()\nprint(a is a)\n": "`is`",
            "class A:\n    def __init__(self):\n        self.x = 1.0\na = A()\nprint(isinstance(a, A))\n": "isinstance",
            "def f():\n    class A:\n        pass\n    return 0\n": "a class inside a function",
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass A:\n    x: float\na = A(1.0)\na.x = 2.0\n":
                "assigning a field of a frozen dataclass",
            "from dataclasses import dataclass, field\n@dataclass\nclass A:\n    x: list = field(default_factory=list)\n":
                "a dataclass field(...)",
            "from collections import namedtuple\nP = namedtuple('P', 'x y')\np = P(1.0, 2.0)\np.x = 3.0\n": "assigning a field of a named tuple",
            "class A:\n    def __init__(self):\n        self.__x = 1.0\na = A()\nprint(a.__x)\n": "the private attribute `__x` outside the class `A`",
        }
        for body, want in cases.items():
            with self.subTest(body=body):
                self.refused_with(body, want)

    def test_objects_where_python_raises(self):
        pre = "import numpy as np\nclass C:\n    def __init__(self):\n        self.x = 1.0\np = C()\na = np.zeros(3)\n"
        cases = {"r = abs(p)\n": "`abs` of an object of class `C`", "r = int(p)\n": "`int` of an object of class `C`",
                 "r = np.sqrt(p)\n": "`np.sqrt` of an object of class `C`", "r = min(p, p)\n": "builtin min of an object",
                 "r = a[p]\n": "an index that is an object of class `C`", "r = p in (p, p)\n": "`in` with an object",
                 "r = p + 1\n": "an operator on an object of class `C`", "r = len(p)\n": "len of an object of class `C`",
                 "if p:\n    r = 1\n": "the truth value of an object of class `C`"}
        for body, want in cases.items():
            with self.subTest(body=body):
                self.refused_with(pre + body, want)

    def test_dicts_that_do_not_translate(self):
        cases = {
            "d = {'a': 1.0}\nd['b'] = 2.0\n": "adding the key 'b'",
            "d = {'a': 1.0}\nk = 'a'\nprint(d[k])\n": "a dict read with a key computed at run time",
            "d = {1: [1.0], 2: [2.0]}\ni = 1\nprint(d[i])\n": "a dict value that is",
            "d = {'a': 1.0}\nprint(d['b'])\n": "the key 'b', which a dict with the keys 'a' does not have",
            "d = {'a': 1.0, 'a': 2.0}\n": "a repeated key",
            "d = {'a': 1.0}\nfor k in d:\n    if k == 'a':\n        break\n": "`break` in a loop over a dict",
            "d = {'a': 1.0}\nx = list(d.values())\n": "outside a for loop header",
            "d = {'a': 1.0}\ne = d\ne['a'] = 2.0\n": "a second name for a dict",
            "d = {i: 'x' for i in range(3)}\nd[5] = 2.0\n": "a dict whose values are both numbers and strings",
        }
        for body, want in cases.items():
            with self.subTest(body=body):
                self.refused_with(body, want)

    def test_keyword_arguments_out_of_order_with_calls(self):
        src = "def g(x):\n    print(x)\n    return x\ndef f(a, b):\n    return a - b\nprint(f(b=g(1.0), a=g(2.0)))\n"
        self.refused_with(src, "keyword arguments in another order")
        # in the parameters' order, or without calls, they translate
        self.assertEqual(messages("def f(a, b):\n    return a - b\nprint(f(b=1.0, a=2.0))\n"), [])

    def test_a_name_that_is_an_int_and_a_bool(self):
        self.refused_with("x = 1\nif x > 0:\n    x = True\nprint(x)\n", "`x` holds a bool here and a number elsewhere")
        v = tr("x = 3\nprint(x)\nx = True\nprint(x)\n")["viv"]
        self.assertIn("x_2 = true", v)


if __name__ == "__main__":
    unittest.main()

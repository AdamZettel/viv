# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's np.random: NumPy's legacy functions after np.random.seed, and a Generator from
np.random.default_rng(seed), draw for draw through lib/nprandom; and the refusals that keep
them exact (no seed, the legacy normals, a Generator used as a value).

    python3 -m pytest -q tools/py2viv/tests/test_np_random.py

The differential tests run CPython with NumPy against the Viv interpreter, Viv compiled to C
and Viv compiled to wasm (diff.py), and require every printed value to agree bit for bit. They
need a built compiler (VIV=<path> or target/release/viv); the wasm engine joins them when
viv.wasm and node are present.
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
    def run_diff(self, src, name, verdict="PASS (bitwise)"):
        if not have_viv():
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        engines = ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write("import numpy as np\n" + textwrap.dedent(src))
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines)
        self.assertEqual(rep["verdict"], verdict, diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])
        return rep

    def test_legacy_seeds(self):
        # an int seed (Knuth's recurrence), the largest one, an array seed (init_by_array), a
        # key longer than the state, reseeding in a loop
        self.run_diff("""
            for s in [0, 1, 42, 2**32 - 1]:
                np.random.seed(s)
                print(np.random.rand(), np.random.rand(3))
            np.random.seed([1, 2, 3])
            print(np.random.random_sample(2))
            np.random.seed(np.arange(700) * 7919)
            print(np.random.random(2))
            np.random.seed([2**32 - 1, 0])
            print(np.random.ranf(), np.random.sample(1))
        """, "legacy_seeds")

    def test_legacy_distributions_interleaved(self):
        # every translated function, each draw leaving the state where NumPy leaves it
        self.run_diff("""
            np.random.seed(2024)
            print(np.random.rand(2, 3), np.random.uniform(-1.0, 3.0), np.random.uniform(5.0, 1.0, 4))
            print(np.random.randint(6), np.random.randint(-3, 3, size=5), np.random.randint(0, 2**32))
            print(np.random.randint(-5, 2**40, size=3), np.random.randint(7, 8, size=2), np.random.randint(0, 5, size=(2, 2)))
            a = np.arange(10.0)
            np.random.shuffle(a)
            L = [1, 2, 3, 4, 5, 6]
            np.random.shuffle(L)
            M = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
            np.random.shuffle(M)
            print(a, L, M)
            print(np.random.permutation(7), np.random.permutation([1.5, -2.0, 3.0]), np.random.permutation(M))
            print(np.random.choice(10), np.random.choice(10, 4), np.random.choice([0.5, 1.5, 2.5], 5))
            print(np.random.choice(6, 6, replace=False), np.random.choice(4, 5, p=[0.1, 0.2, 0.3, 0.4]))
            print(np.random.choice(5, 3, replace=False, p=[0.4, 0.3, 0.2, 0.1, 0.0]), np.random.rand(0))
        """, "legacy_mix")

    def test_legacy_long_runs(self):
        # more words than one twist of the state (624), in a def the module seeds first
        self.run_diff("""
            def noise(n):
                return np.random.rand(n)

            np.random.seed(7)
            x = noise(1500)
            print(x[0], x[623], x[624], x[1499], np.sum(x))
            k = np.random.randint(0, 100, size=2000)
            print(k[-5:], np.sum(k))
        """, "legacy_long")

    def test_generator_seeds(self):
        # SeedSequence of ints up to 2**64 + 5 and of lists of them, an int expression
        self.run_diff("""
            for s in [0, 1, 42, 2**32 - 1, 9007199254740991]:
                g = np.random.default_rng(s)
                print(g.random(), g.integers(0, 2**40))
            print(np.random.default_rng(2**64 + 5).random(2), np.random.default_rng(2**128 - 1).random())
            print(np.random.default_rng([1, 2, 3]).random(), np.random.default_rng([2**40, 5]).random())
            n = 5
            print(np.random.default_rng(n * 3).random(2), np.random.default_rng(np.array([3, 2**32])).random())
        """, "generator_seeds")

    def test_generator_methods_interleaved(self):
        # integers below 2**32 keep half of a 64-bit draw for the next one: a random() between
        # them does not take it, and the second integers() does
        self.run_diff("""
            from numpy.random import default_rng
            rng = default_rng(99)
            print(rng.integers(0, 10), rng.random(), rng.integers(0, 10), rng.integers(0, 10, size=3))
            print(rng.integers(5), rng.integers(1, 6, size=4, endpoint=True), rng.integers(0, 2**40, size=2))
            print(rng.integers(0, 2**32), rng.integers(-7, 2**32 + 5), rng.integers(3, 4))
            print(rng.uniform(), rng.uniform(-2.0, 2.0, size=3), rng.random((2, 2)))
            print(rng.standard_normal(5), rng.normal(10.0, 0.5), rng.normal(size=(2, 2)))
            b = np.arange(8)
            rng.shuffle(b)
            print(b, rng.permutation(6), rng.permutation(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
            print(rng.choice(10), rng.choice(10, 3), rng.choice([2.5, 3.5, 4.5], 4), rng.choice(8, 8, replace=False))
            big = rng.choice(20000, 500, replace=False)
            print(rng.choice(50, 5, replace=False, shuffle=False), big[:5], np.sum(big))
            print(rng.choice(4, 6, p=[0.1, 0.2, 0.3, 0.4]), rng.choice(4, 3, replace=False, p=[0.1, 0.2, 0.3, 0.4]))
        """, "generator_mix")

    def test_generator_through_defs(self):
        # a Generator passed to a def and returned: the state is shared, as Python shares the object
        self.run_diff("""
            def make(seed):
                return np.random.default_rng(seed)

            def walk(rng, steps):
                x = 0.0
                for k in range(steps):
                    x += rng.standard_normal()
                return x

            g = make(3)
            h = g
            print(walk(g, 50), h.random(), walk(h, 10))
        """, "generator_defs")

    def test_normal_tail_stops(self):
        # default_rng(0)'s draw 303 is in the ziggurat's tail, where NumPy calls log1p: the run
        # stops there with a py2viv line, in every engine, after printing the draws before it
        if not have_viv():
            self.skipTest("no viv binary")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tail.py")
            with open(path, "w") as f:
                f.write("import numpy as np\nrng = np.random.default_rng(0)\nx = rng.standard_normal(300)\nprint(x[-1])\n"
                        "print(rng.standard_normal(5))\n")
            rep = diff.differential(path, os.path.join(d, "out"))
        self.assertTrue(rep["engines_agree"], diff.format_report(rep))
        self.assertTrue((rep.get("stop") or {}).get("viv", "").startswith("py2viv: a normal draw reached the ziggurat's tail"),
                        diff.format_report(rep))
        line4 = [v for v in rep["values"] if v["line"] == 4]
        self.assertEqual(line4[0]["verdict"], "bitwise")


class Refusals(unittest.TestCase):
    def test_unseeded_is_refused(self):
        m = messages("import numpy as np\nx = np.random.rand(3)\ng = np.random.default_rng()\nh = np.random.default_rng(None)\n")
        self.assertIn("np.random.rand without np.random.seed in this file", m[0])
        self.assertIn("np.random.default_rng() without a seed", m[1])
        self.assertIn("np.random.default_rng() without a seed", m[2])
        m = messages("import numpy as np\nnp.random.seed()\n")
        self.assertIn("np.random.seed() without a seed", m[0])

    def test_draw_before_the_seed_stops(self):
        res = tr("import numpy as np\ndef f():\n    return np.random.rand()\nx = f()\nnp.random.seed(1)\nprint(x)\n")
        self.assertEqual(res["refusals"], [])
        self.assertIn("py2viv: a draw from np.random before np.random.seed", res["viv"])

    def test_legacy_normals_are_refused(self):
        m = messages("import numpy as np\nnp.random.seed(0)\na = np.random.randn(3)\nb = np.random.normal(0.0, 1.0)\n"
                     "c = np.random.standard_normal()\n")
        self.assertEqual(len(m), 3)
        for line, name in zip(m, ("randn", "normal", "standard_normal")):
            self.assertIn("`np.random.%s`" % name, line)

    def test_other_distributions_and_objects_are_refused(self):
        m = messages("import numpy as np\nnp.random.seed(0)\na = np.random.poisson(3.0)\nr = np.random.RandomState(0)\n"
                     "g = np.random.default_rng(1)\nb = g.exponential(2.0)\nc = g.bit_generator\n")
        self.assertIn("np.random.poisson", m[0])
        self.assertIn("`np.random.RandomState`", m[1])
        self.assertIn("Generator.exponential", m[2])
        self.assertIn("the attribute `.bit_generator` of a Generator", m[3])

    def test_generator_as_a_value_is_refused(self):
        m = messages("import numpy as np\nr = np.random.default_rng(3)\nz = r + 1\nq = [r, r]\nprint(r)\n")
        self.assertIn("a Generator (np.random.default_rng) used as a value here", m[0])
        self.assertIn("a Generator (np.random.default_rng) used as a value here", m[1])
        self.assertIn("printing a Generator", m[2])

    def test_draws_are_evaluated_once_and_in_order(self):
        m = messages("import numpy as np\nnp.random.seed(1)\nc = 1\nx = np.random.rand() + (np.random.rand() if c > 0 else 0.0)\n"
                     "y = np.square(np.random.rand(3))\nz = np.random.rand(5)[1:]\nw = np.random.rand(5)[-1]\n")
        self.assertIn("after a call of `np.random.rand`", m[0])
        self.assertIn("np.square of an expression that draws random numbers", m[1])
        self.assertIn("a slice of an expression that draws random numbers", m[2])
        # a negative index reads its array once in Viv, as in NumPy: nothing to refuse
        self.assertEqual(len(m), 3, m)

    def test_shuffle_of_a_view_is_refused(self):
        m = messages("import numpy as np\nnp.random.seed(1)\na = np.zeros(4)\nnp.random.shuffle(a[1:])\nb = a[1:]\n"
                     "np.random.shuffle(b)\nprint(a)\n")
        self.assertIn("np.random.shuffle of an expression", m[0])
        self.assertIn("which shuffles it in place", m[1])

    def test_narrow_dtypes_are_refused(self):
        m = messages("import numpy as np\nnp.random.seed(1)\na = np.random.randint(0, 5, size=3, dtype=np.int32)\n"
                     "g = np.random.default_rng(1)\nb = g.random(3, dtype=np.float32)\n")
        self.assertIn("np.random.randint with dtype=np.int32", m[0])
        self.assertIn("Generator.random with dtype=np.float32", m[1])

    def test_names_of_the_library_are_reserved(self):
        res = tr("import numpy as np\nnpr_mt_new = 3\nNPR_N = 4\nprint(npr_mt_new + NPR_N)\n")
        self.assertIn("u_npr_mt_new", res["viv"])
        self.assertIn("u_NPR_N", res["viv"])


if __name__ == "__main__":
    unittest.main()

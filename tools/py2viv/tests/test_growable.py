# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's growable lists (gap R17): `lst = []` and `lst.append(x)` become Viv's
`lst = []` and `lst = append(lst, x)` when nothing can alias the list, and are refused
otherwise (standard library only; the run needs a built compiler).

    python3 -m unittest discover -s tools/py2viv/tests -p 'test_growable.py'
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, PKG)
import py2viv  # noqa: E402


def viv_bin():
    for p in (os.environ.get("VIV"), os.path.join(ROOT, "target", "release", "viv")):
        if p and os.path.exists(p):
            return p
    return None


def tr(src):
    return py2viv.translate(textwrap.dedent(src), "t.py")


SCAN = """
    def evens(n):
        out = []
        for i in range(n):
            if i % 2 == 0:
                out.append(i)
        return out


    def mean_half(n):
        r = []
        for i in range(n):
            r.append(i / 2)
        total = 0.0
        for v in r:
            total += v
        return total / len(r)


    e = evens(9)
    print(e, len(e))
    print(mean_half(5))
    kept = []
    for i in range(20):
        if i * i % 7 == 2:
            kept.append(i * 1.5)
    print(kept, kept[0], sum(kept))
    """


class Growable(unittest.TestCase):
    def test_append_rebinds(self):
        res = tr(SCAN)
        self.assertEqual(res["messages"], [])
        v = res["viv"]
        self.assertIn("out = []", v)
        self.assertIn("out = append(out, i)", v)
        self.assertIn("r = append(r, i / 2)", v)
        self.assertIn("kept = append(kept, i * 1.5)", v)

    def test_runs_as_python_prints(self):
        viv = viv_bin()
        if viv is None:
            self.skipTest("no viv binary (cargo build --release -p viv, or VIV=<path>)")
        res = tr(SCAN)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.viv")
            with open(p, "w") as f:
                f.write(res["viv"])
            out = subprocess.run([viv, "run", "-q", p], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, "[0, 2, 4, 6, 8] 5\n1.0\n[4.5, 6.0, 15.0, 16.5, 25.5, 27.0] 4.5 94.5\n")

    def refusal(self, src, words):
        res = tr(src)
        whats = [r.what for r in res["refusals"]]
        self.assertTrue(any(words in w for w in whats), whats)

    def test_an_alias_is_refused(self):
        # Python's alias sees the append; Viv's would not
        self.refusal("def f():\n    a = []\n    b = a\n    a.append(1)\n    return b\n", "an empty list grown and aliased")

    def test_an_argument_is_refused(self):
        self.refusal("def g(x):\n    return x\n\n\ndef f():\n    a = []\n    a.append(1)\n    return g(a)\n", "and passed to `g`")

    def test_other_methods_are_refused(self):
        # pop is followed as a statement or the whole value of an assignment (tests/test_py_funcs.py);
        # inside another expression it is not
        self.refusal("def f():\n    a = []\n    a.append(1)\n    print(a.pop() + 1)\n    return a\n", "and changed or read by `.pop`")

    def test_appending_while_looping_over_it_is_refused(self):
        self.refusal("def f():\n    a = []\n    a.append(1)\n    for v in a:\n        if v < 3:\n            a.append(v + 1)\n    return a\n",
                     "while a loop runs over it")

    def test_ints_and_floats_are_refused(self):
        self.refusal("def f():\n    a = []\n    a.append(1)\n    a.append(2.5)\n    return a\n", "a list of ints and floats")

    def test_a_list_bound_elsewhere_is_not_grown(self):
        self.refusal("import numpy as np\na = np.zeros(2)\na.append(1.0)\n", "list.append")

    def test_a_literal_start_is_grown(self):
        src = "def f():\n    a = [0]\n    for i in range(3):\n        a.append(i + 1)\n    return a\n\n\nb = [0.5, 1.5]\nb.append(2.5)\nprint(f(), b, len(b))\n"
        res = tr(src)
        self.assertEqual(res["messages"], [])
        self.assertIn("a = [0]", res["viv"])
        self.assertIn("b = append(b, 2.5)", res["viv"])
        viv = viv_bin()
        if viv is None:
            self.skipTest("no viv binary")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.viv")
            with open(p, "w") as f:
                f.write(res["viv"])
            out = subprocess.run([viv, "run", "-q", p], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, "[0, 1, 2, 3] [0.5, 1.5, 2.5] 3\n")

    def test_a_literal_start_mixing_types_is_refused(self):
        self.refusal("def f():\n    a = [1, 2.5]\n    a.append(1.0)\n    return a\n", "starts with float and int")
        self.refusal("def f():\n    a = [1]\n    a.append(2.5)\n    return a\n", "a list of ints and floats")

    def test_enumerate_in_a_for_header_reads_it(self):
        res = tr("def f():\n    a = [1.0]\n    a.append(2.0)\n    t = 0.0\n    for k, v in enumerate(a):\n        t += k * v\n    return t\n")
        self.assertEqual(res["messages"], [])
        self.refusal("def f():\n    a = [1.0]\n    for k, v in enumerate(a):\n        a.append(v)\n    return a\n", "while a loop runs over it")

    def test_a_literal_start_aliased_is_refused(self):
        self.refusal("def f():\n    a = [1.0]\n    b = a\n    a.append(2.0)\n    return b\n", "a list grown and aliased")


if __name__ == "__main__":
    unittest.main()

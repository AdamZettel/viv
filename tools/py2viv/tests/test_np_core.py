# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's NumPy core: slices, reductions
along an axis in NumPy's order, reshaping and joining; and the refusals that keep them exact
(a slice is a view in NumPy and a copy in Viv; a transpose is summed in memory order).

    python3 -m pytest -q tools/py2viv/tests/test_np_core.py

The differential tests need a built compiler (VIV=<path> or target/release/viv); the wasm
engine joins them when viv.wasm and node are present.
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
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write("import numpy as np\n" + textwrap.dedent(src))
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines)
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])

    def test_slices(self):
        self.run_diff("""
            a = np.array([1.5, -2.0, 3.25, 4.0, -0.5, 6.0])
            print(a[1:4], a[-2:], a[:-1], a[::2], a[4:1], a[10:], a[-100:2])
            k = -3
            print(a[k:], a[:k], a[k:k + 2], np.sum(a[1:]) + a[0])
            A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            j = -1
            print(A[:, 1], A[1:, -1], A[0][1:], A[1:-1, 1:-1], A[:, j], A[::2, ::2])
            L = [3, 1, 4, 1, 5]
            M = L[1:4]
            M[0] = 7
            print(L, M)
        """, "slices")

    def test_slice_stores_and_in_place_updates(self):
        self.run_diff("""
            a = np.linspace(0.0, 1.0, 7)
            u = np.zeros(7)
            u[1:-1] = a[2:] - a[:-2]
            u[0:2] = 7.0
            u[3:] += 0.5
            C = np.zeros((3, 3))
            C[1] = a[:3]
            C[:, 0] = a[4:]
            print(u, C)
            x = np.zeros(3)
            y = x
            y += 1.0
            def bump(v):
                v *= 2.0
                return 0
            k = bump(x)
            print(x)
        """, "stores")

    def test_reductions_along_an_axis(self):
        self.run_diff("""
            A = np.array([[1.0, 2.5, -3.0], [-3.0, 0.25, 1e16], [0.1, 0.2, 0.3]])
            print(np.sum(A, axis=0), np.sum(A, axis=1), A.sum(axis=-1, keepdims=True), np.sum(A, keepdims=True))
            print(np.mean(A, axis=0), np.var(A, axis=1), np.std(A, 0), A.var(axis=0, keepdims=True))
            print(np.max(A, axis=1), A.min(axis=0), np.argmax(A, axis=1), A.argmin(axis=0))
            print(np.prod(A, axis=1), np.sum(A.T, axis=0), np.sum(A.T), np.cumsum(A, axis=1))
            M = np.array([[1, 2, 3], [4, 5, 6]])
            print(np.sum(M, axis=0), M.sum(1), np.max(M, axis=0), np.cumsum(M, axis=0), np.cumsum(M))
            T = np.zeros((2, 3, 4))
            for i in range(2):
                for j in range(3):
                    for k in range(4):
                        T[i, j, k] = 0.1 * (i + 1) - 0.37 * j + 1e-3 * k * k
            print(np.sum(T, axis=0), np.sum(T, axis=1), np.mean(T, axis=2), np.var(T, axis=-2))
        """, "axis")

    def test_numpys_order_of_summation(self):
        # a sum longer than NumPy's buffer (8192) is added block by block; a sum along the
        # first axis adds the rows in order; eight -0.0 sum to 0.0
        self.run_diff("""
            n = 20000
            v = np.zeros(n)
            for i in range(n):
                v[i] = ((i * 7919) % 1000) * 1e-3 + 1e12 * (i % 3)
            print(np.sum(v), np.mean(v))
            W = np.zeros((3, 9000))
            for i in range(3):
                for j in range(9000):
                    W[i, j] = 1.0 / (1.0 + i + 3.0 * j)
            print(np.sum(W, axis=1), np.sum(W), np.sum(W[:, 1:-1]))
            z = np.zeros(8) * -1.0
            print(np.sum(z), np.sum(W[:, :40], axis=0)[:3])
        """, "order")

    def test_reshape_and_joins(self):
        self.run_diff("""
            a = np.arange(12.0)
            B = a.reshape(3, 4)
            C = np.reshape(a, (2, -1))
            D = B.reshape(-1)
            print(B, C, D, np.ravel(C), C.flatten())
            u = np.array([1.0, 2.0])
            w = np.array([3.0, 4.0])
            print(np.concatenate((u, w)), np.concatenate((B, B), axis=1), np.concatenate((B, B), axis=None))
            print(np.stack((u, w)), np.stack((u, w), axis=1), np.vstack((u, w)), np.hstack((u, w)), np.column_stack((u, w)))
            X, Y = np.meshgrid(np.array([1.0, 2.0, 3.0]), u)
            P, Q = np.meshgrid(u, w, indexing='ij')
            print(X, Y, P, Q)
        """, "joins")


class Refusals(unittest.TestCase):
    def test_writes_that_would_show_a_view(self):
        cases = {
            "b = a[1:3]\nb[0] = 9.0\n": "a write through `b`, which is a slice of another array",
            "b = a[1:3]\na[1] = 9.0\nprint(b)\n": "a slice of `a`, which this code also writes into",
            "b = a[1:3]\nb += 1.0\n": "bound to `b`, which this code writes through",
            "def w(v):\n    v[0] = 7.0\n    return 0\n\nk = w(a[1:])\n": "a slice in a statement that calls `w`",
            "a[1:].sort()\n": "`.sort()` of a slice",
            "for x in a[1:]:\n    a[0] = x\n": "a slice of `a`, which this code also writes into",
            "def mid(v):\n    return v[1:-1]\n\nm = mid(a)\na[1] = 0.0\n": "a view of `a` (`mid` may return one)",
            "B = a.reshape(2, 2)\nB[0, 0] = 1.0\n": "a write through `B`",
            "r = a[::-1]\nr[0] = 5.0\n": "a write through `r`, which is a slice of another array",
            "r = a[::-1]\na[0] = 5.0\nprint(r)\n": "a slice of `a`, which this code also writes into",
            "r = a.reshape(2, 2)[::-1]\n": "a slice with the step `-1`",
            "r = a[1:3][0]\n": "a subscript of a slice",
        }
        for body, want in cases.items():
            m = messages("import numpy as np\na = np.array([1.0, 2.0, 3.0, 4.0])\n" + body)
            self.assertTrue(m, body)
            self.assertIn(want, m[0], body)

    def test_reads_of_a_view_translate(self):
        src = "import numpy as np\na = np.array([1.0, 2.0, 3.0, 4.0])\nb = a[1:3]\nc = b * 2.0\n" \
              "def rd(v):\n    return v.sum()\n\nprint(c, rd(a[1:]), np.max(a[::2]))\n"
        self.assertEqual(messages(src), [])

    def test_a_transpose_is_summed_in_memory_order(self):
        v = tr("import numpy as np\nA = np.ones((2, 3))\ns = np.sum(A.T)\n")["viv"]
        self.assertIn("py_npsum(transpose(transpose(A)))", v)
        m = messages("import numpy as np\nA = np.ones((2, 3))\ns = np.sum(A.T * 2.0)\n")
        self.assertIn("a transpose's memory order", m[0])
        m = messages("import numpy as np\ndef f(X):\n    return np.sum(X)\n\nA = np.ones((2, 3))\ns = f(A.T)\n")
        self.assertIn("a transpose's memory order", m[0])

    def test_axis_must_be_a_literal_in_range(self):
        pre = "import numpy as np\nA = np.ones((2, 3))\nk = 0\n"
        self.assertIn("an axis that is not an int literal", messages(pre + "s = np.sum(A, axis=k)\n")[0])
        self.assertIn("NumPy raises AxisError", tr(pre + "s = np.sum(A, axis=2)\n")["messages"][0])
        self.assertIn("np.sum with `dtype=`", messages(pre + "s = np.sum(A, dtype=float)\n")[0])

    def test_rank_three_literals_are_refused(self):
        m = messages("import numpy as np\nT = np.array([[[1.0]], [[2.0]]])\n")
        self.assertIn("a rank-3 list literal", m[0])


if __name__ == "__main__":
    unittest.main()

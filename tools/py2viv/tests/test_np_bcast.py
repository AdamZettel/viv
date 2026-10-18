# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's broadcasting, masks and integer indices (the NumPy survey's build steps 4
and 5): operators of two arrays of different shapes, `None`/np.newaxis in a subscript,
keepdims results broadcast back, a Bool mask or an integer array as an index (read, store,
`+=`), np.nonzero, np.where with one argument, np.argwhere, np.count_nonzero,
np.flatnonzero; and the refusals and run-time stops that keep them exact.

    python3 -m pytest -q tools/py2viv/tests/test_np_bcast.py

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
    def run_diff(self, src, name, stops=None):
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
        if stops is not None:
            self.assertTrue(rep["stop"]["python"].startswith(stops), rep["stop"])
            self.assertEqual(rep["stop"]["viv"], stops)
        return rep

    def test_broadcasting(self):
        self.run_diff("""
            A = np.array([[1.5, -2.0, 3.25], [4.0, -0.5, 6.0]])
            mean_row = A.mean(axis=0)
            c = np.array([10.0, -0.0])
            print(A - mean_row, A * c[:, None], A + c[:, np.newaxis], A / c.reshape(-1, 1))
            print(A / A.sum(axis=1, keepdims=True), A - A.max(axis=0, keepdims=True))
            x = np.array([1.0, 2.0, 3.0])
            y = np.array([0.5, 1.5])
            D = x[:, None] - y[None, :]
            print(D, D * D, x[None] + x[:, None] > 3.0, x[:, None] == x)
            T = np.zeros((2, 1, 3))
            for i in range(2):
                for k in range(3):
                    T[i, 0, k] = 0.1 * i - 0.37 * k
            print(T + np.array([[1.0], [2.0], [3.0], [4.0]]))
            M = np.array([[1, 2, 3], [4, 5, 6]])
            print(M + np.array([10, 20, 30]), M * np.array([[2], [3]]), M < np.array([2, 5, 9]))
            z = np.array([1.0, np.nan, -np.inf])
            print(z[:, None] * np.array([0.0, -0.0]), np.maximum(A, x), np.where(A > 0, A, c[:, None]))
            print(np.fmin(np.array([[np.nan], [1.0]]), np.array([0.5, np.nan])), np.allclose(A, A[0]))
            print(np.arctan2(np.array([[1.0], [-1.0]]), x), np.hypot(np.array([[3.0], [5.0]]), np.array([4.0, 12.0])))
            print(np.add(A, x), np.multiply(A, c[:, None]), np.subtract(3, 5), np.divide(x, 0.0), np.divide(-1.0, 0.0))
            u = np.array([0.5, -1.0, 2.0, 0.0, 3.5])
            for _ in range(3):
                u = u - 0.25 * (u - np.roll(u, 1))
                u[1:-1] += 0.1 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            print(u)
        """, "bcast")

    def test_in_place_updates_and_slice_stores_broadcast(self):
        self.run_diff("""
            A = np.array([[1.5, -2.0, 3.25], [4.0, -0.5, 6.0]])
            B = A
            A -= A.mean(axis=0)
            print(A, B)
            C = np.zeros((2, 3))
            C += np.array([1.0, 2.0, 3.0])
            C *= np.array([[2.0], [-1.0]])
            C[0] = np.array([7.0])
            C[:, 1:] = np.array([[0.5], [0.25]])
            C[1, :] = 9.0
            print(C)
        """, "inplace")

    def test_broadcasting_stops_where_numpy_raises(self):
        self.run_diff("""
            a = np.array([1.0, 2.0, 3.0])
            print(a * 2.0)
            b = a + np.array([1.0, 2.0])
            print(b)
        """, "bad_shapes", stops="ValueError: operands could not be broadcast together with shapes")
        self.run_diff("""
            a = np.array([[1.0, 2.0, 3.0]])
            a += np.ones((2, 3))
            print(a)
        """, "bad_inplace", stops="ValueError: non-broadcastable output operand with shape")

    def test_masks(self):
        self.run_diff("""
            a = np.array([1.5, -2.0, np.nan, -0.0, 4.0])
            print(a[a > 0], a[a != a], a[a > 100.0], a[[True, False, True, False, False]])
            A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            print(A[A > 2.5], A[np.array([False, True])], A[:, np.array([True, False, True])])
            b = a[a > 0]
            b[0] = 99.0
            print(a, b)
            a[a < 0] = 0.0
            a[np.isnan(a)] = -1.0
            print(a)
            A[A > 2.5] += 10.0
            A[:, np.array([True, False, False])] = np.array([[7.0], [8.0]])
            print(A, np.count_nonzero(A > 7.0), np.flatnonzero(a), np.argwhere(A > 7.5))
        """, "masks")

    def test_integer_indices(self):
        self.run_diff("""
            a = np.array([1.5, -2.0, 3.25, 4.0])
            idx = np.array([3, 0, 0, -1])
            print(a[idx], a[[1, 2]], a[np.array([[0, 1], [2, 3]])])
            A = np.arange(12.0).reshape(3, 4)
            print(A[[2, 0]], A[:, [3, 3, 0]], A[[0, 2], [1, 3]], A[1, [0, 2]], A[[0, 1], -1], A[np.array([[0], [2]]), np.array([1, 3])])
            z = np.zeros(5)
            z[[0, 0, 4]] = np.array([1.0, 2.0, 3.0])
            z[[1, 1, 1]] += 1.0
            z[np.array([2, -2])] = 7.5
            print(z)
            B = np.zeros((3, 3))
            B[[0, 1, 2], [2, 1, 0]] = np.array([1.0, 2.0, 3.0])
            B[:, [0, 0]] = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
            B[[1, 1], :] -= 1.0
            print(B)
            M = np.array([[0, 3], [4, 0], [0, 5]])
            i, j = np.nonzero(M)
            print(i, j, np.nonzero(M)[1], np.where(M > 3)[0], M[np.nonzero(M)[0], 1], np.argwhere(M))
            v = np.array([5, 1, 4])
            v[v > 2] -= 2
            print(v, v[np.array([2, 1])] * 3)
        """, "fancy")

    def test_indexing_stops_where_numpy_raises(self):
        self.run_diff("""
            a = np.array([1.5, -2.0, 3.25])
            print(a[np.array([0, 2])])
            print(a[np.array([1, 3])])
        """, "out_of_bounds", stops="IndexError: index")
        self.run_diff("""
            a = np.array([1.5, -2.0, 3.25])
            print(a[np.array([True, False])])
        """, "mask_length", stops="IndexError: boolean index did not match indexed array along dimension")
        self.run_diff("""
            a = np.array([1.5, -2.0, 3.25])
            a[a > 0] = np.array([1.0, 2.0, 3.0])
            print(a)
        """, "mask_store", stops="ValueError: NumPy boolean array indexing assignment cannot assign")

    def test_grad_through_broadcasting(self):
        # a differentiated def whose arrays broadcast: Viv's reverse mode follows the helpers
        src = textwrap.dedent("""
            # py2viv: rtol=1e-13 atol=1e-16
            import numpy as np
            from viv import value_and_grad

            def loss(W, b, X):
                H = np.tanh(X @ W + b)
                d = H - b
                return np.sum(d * d) / d.size

            X = np.array([[0.5, -1.0], [1.5, 0.25], [-0.5, 2.0]])
            W = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
            b = np.array([0.05, -0.1, 0.2])
            l, g = value_and_grad(loss)(W, b, X)
            print(l, g)
        """)
        if not have_viv():
            self.skipTest("no viv binary")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.py")
            with open(path, "w") as f:
                f.write(src)
            rep = diff.differential(path, os.path.join(d, "out"),
                                    engines=("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ()))
        self.assertTrue(rep["verdict"].startswith("PASS"), diff.format_report(rep))


class Translations(unittest.TestCase):
    def test_an_array_with_itself_needs_no_expansion(self):
        v = tr("import numpy as np\nx = np.ones(3)\ny = x * x\nz = (x + 1.0) - (x + 1.0)\n")["viv"]
        self.assertIn("y = x * x", v)
        self.assertIn("z = x + 1.0 - (x + 1.0)", v)

    def test_shapes_equal_by_their_form_keep_vivs_operator(self):
        v = tr("import numpy as np\nu = np.ones(5)\nw = u[2:] - 2.0 * u[1:-1] + u[:-2]\n"
               "r = u - 0.5 * (u - np.roll(u, 1))\nu += np.sin(u)\n")["viv"]
        self.assertIn("w = u[min(2, len(u)):] - 2.0 * u[py_lo(1, len(u)):py_hi(1, -1, len(u))] + u[:py_lo(-2, len(u))]", v)
        self.assertIn("r = u - 0.5 * (u - roll(u, 1))", v)
        self.assertIn("u += sin(u)", v)

    def test_two_arrays_go_through_a_broadcasting_helper(self):
        v = tr("import numpy as np\nx = np.ones(3)\ny = np.ones((2, 3))\nz = y - x\n")["viv"]
        self.assertIn("z = py_bsub(y, x)", v)
        v = tr("import numpy as np\nx = np.arange(3)\nz = x[:, None] + x\n")["viv"]
        self.assertIn("py_int_arr(py_badd(py_newax(x, [1]), x))", v)

    def test_a_mask_read_is_a_copy(self):
        m = messages("import numpy as np\na = np.ones(4)\nb = a[a > 0]\nb[0] = 2.0\nprint(a)\n")
        self.assertEqual(m, [])
        m = messages("import numpy as np\na = np.ones(4)\ni = np.array([0, 1])\nb = a[i]\na[0] = 5.0\nprint(b)\n")
        self.assertEqual(m, [])

    def test_none_makes_a_view(self):
        m = messages("import numpy as np\na = np.ones(3)\nb = a[:, None]\nb[0, 0] = 2.0\n")
        self.assertIn("a write through `b`", m[0])
        m = messages("import numpy as np\na = np.ones(3)\nb = a[None]\na[0] = 2.0\nprint(b)\n")
        self.assertIn("a slice of `a`, which this code also writes into", m[0])
        m = messages("import numpy as np\na = np.ones(3)\na[None] = 2.0\n")
        self.assertIn("`None`, np.newaxis or `...` in a store target", m[0])


class Refusals(unittest.TestCase):
    def check(self, body, want):
        m = messages("import numpy as np\na = np.array([1.0, 2.0, 3.0, 4.0])\nA = np.ones((2, 3))\n"
                     "z = np.zeros(2, dtype=complex)\nL = [1.0, 2.0]\n" + body)
        self.assertTrue(m, body)
        self.assertIn(want, m[0], body)

    def test_refused(self):
        cases = {
            "r = z / z\n": "`/` of two arrays, one of them complex",
            "r = z < z\n": "comparing a complex array",
            "r = L < L\n": "of a Python list or a tuple",
            "r = L < 2.0\n": "of a Python list or a tuple",
            "r = a[np.array([0.0, 1.0])]\n": "an index array of float numbers",
            "r = A[np.array([True, False]), 0]\n": "a Bool mask beside another index",
            "r = A[1:, np.array([0, 1])]\n": "a range beside a mask or integer indices",
            "r = L[np.array([0, 1])]\n": "held in a Python list",
            "k = np.array([1, 2])\nk += 0.5\n": "whose dtype cannot hold the result",
            "k = np.array([1, 2])\nk /= 2\n": "whose dtype cannot hold the result",
            "a[0] = np.array([1.0])\n": "an array stored into one element",
            "a += A\n": "`+=` of a rank-1 array with a rank-2 result",
            "r = np.nonzero(a)\n": "np.nonzero outside",
            "r = np.nonzero(a)[1]\n": "np.nonzero(...)[1] of a rank-1 array",
            "def f(t):\n    return t\n\na[f(np.array([0]))] += 1.0\n": "through indices that call `f`",
            "s = a[1:]\na[a > 2] = 0.0\nprint(s)\n": "a slice of `a`, which this code also writes into",
        }
        for body, want in cases.items():
            self.check(body, want)


if __name__ == "__main__":
    unittest.main()

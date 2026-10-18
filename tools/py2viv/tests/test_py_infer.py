# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's whole-program typing of defs (infer.py): parameters typed by every call site, defs
specialized per argument types (a clone per distinct tuple), return types through recursion,
Python lists and NumPy arrays kept apart, list parameters a def may change when every call passes
a new list, the columns of rows mixing ints and floats, annotations and `#pythran export` lines
as evidence, and unknown types that come from a refusal reported as cascades; and the refusals
that keep all of it exact.

    python3 -m pytest -q tools/py2viv/tests/test_py_infer.py

The differential tests run each program in CPython and its translation in the Viv interpreter,
in C and in wasm (diff.py), and need a built compiler (VIV=<path> or target/release/viv); the wasm
engine joins them when viv.wasm and node are present. A test passes only when every printed value
agrees bit for bit and the printed text is CPython's byte for byte. In every program here the
types of some parameter are knowable only through its call sites, a def's return, or a
container's items.
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
        res = tr(src)
        self.assertEqual(res["messages"], [], "\n".join(res["messages"]))
        engines = ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write(textwrap.dedent(src))
            rep = diff.differential(path, os.path.join(d, "out"), engines=engines)
        self.assertTrue(rep["engines_agree"], diff.format_report(rep))
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        bad = [r for r in rep["lines"] if r["verdict"] != "identical"]
        self.assertEqual(bad, [], diff.format_report(rep))
        return res

    # ------------------------------------------------ specialization by argument types
    def test_ints_and_floats_each_get_their_own_def(self):
        res = self.run_diff("""
            def scale(x, k):
                return x * k

            def show(x):
                print("x =", x)
                return x

            print(scale(3, 4), scale(2.5, 2), scale(-7, 3), scale(0.5, 0.25))
            print(show(2) + show(3), show(0.5))
            print(scale(2 ** 20, 2 ** 20), scale(1e300, 10))
        """, "ints_floats")
        self.assertIn("def scale__t2(", res["viv"])

    def test_a_number_and_an_array_for_one_parameter(self):
        self.run_diff("""
            import numpy as np

            def half(x, n):
                if n == 0:
                    return x
                return half(x / 2, n - 1)

            def twice(x):
                return x + x

            print(half(8.0, 2), half(np.array([4.0, 2.0]), 1))
            print(twice(2), twice(0.5), twice(np.array([1, 2])), twice(np.array([1.5, -0.0])))
        """, "num_arr")

    def test_a_list_and_an_array_stay_apart(self):
        self.run_diff("""
            import numpy as np

            def norm2(v):
                s = 0.0
                for t in v:
                    s += t * t
                return s

            def first_copy(v):
                b = v[:]
                b[0] = 100.0
                return b[0] + v[0]

            print(norm2([1.0, 2.0]), norm2(np.array([3.0, 4.0])), norm2((0.5, 0.25)))
            L = [1.0, 2.0]
            print(first_copy(L), L)
            show = [3, 4]
            print(show, np.array(show))
        """, "list_arr")

    def test_recursion_chains_and_tuple_returns_per_clone(self):
        self.run_diff("""
            import numpy as np

            def power(b, e):
                if e == 0:
                    return b / b
                return b * power(b, e - 1)

            def is_even(n):
                if n == 0:
                    return True
                return is_odd(n - 1)

            def is_odd(n):
                if n == 0:
                    return False
                return is_even(n - 1)

            def show(x):
                print("show", x)
                return x

            def twice(x):
                return show(x) + show(x)

            def stats(v):
                s = v[0] - v[0]
                for t in v:
                    s += t
                return s, len(v)

            def fill(a, k=2):
                for i in range(len(a)):
                    a[i] = a[i] * k
                return 0

            print(power(3.0, 4), power(np.array([1.5, 2.0]), 3))
            print(is_even(4), is_odd(3), is_even(7))
            print(twice(2), twice(0.5), twice(np.array([1, 2])))
            a, b = stats([1, 2, 3])
            c, d = stats([0.5, 0.25])
            e, f = stats(np.array([1.5, 2.5]))
            print(a, b, c, d, e, f)
            L = [1, 2, 3]
            k = fill(L)
            A = np.array([1.5, 2.5])
            k = fill(A, 3)
            print(L, A)
        """, "recursion")

    # ------------------------------------------------ lists a def changes
    def test_merge_sort_pops_lists_every_call_passes_new(self):
        self.run_diff("""
            def merge(left, right):
                result = []
                while left and right:
                    if right[0] < left[0]:
                        result.append(right.pop(0))
                    else:
                        result.append(left.pop(0))
                result.extend(left)
                result.extend(right)
                return result

            def merge_sort(c):
                if len(c) <= 1:
                    return c
                mid = len(c) // 2
                return merge(merge_sort(c[:mid]), merge_sort(c[mid:]))

            print(merge_sort([0, 5, 3, 2, 2, -45, -2, -5, 103, 9, 1, 7, 11]))
            print(merge_sort([2.5, -0.0, 1.25]), merge_sort([7]))
        """, "merge_sort")

    def test_a_new_list_from_a_def_may_be_changed_by_another(self):
        self.run_diff("""
            def grow(v):
                v.append(1.0)
                return v

            def fresh():
                return [0.5]

            def keep(a):
                return a

            print(grow(fresh()), grow(keep([2.0, 3.0])), grow(sorted([4.0, 3.0])), grow([x * 2.0 for x in range(3)]))
        """, "fresh")

    # ------------------------------------------------ items that keep their types
    def test_int_columns_of_rows_that_mix_ints_and_floats(self):
        self.run_diff("""
            def times(b, e):
                r = b - b
                for _ in range(e):
                    r = r + b
                return r

            cases = [[3, 4], [11, 3], [1.5, 2]]
            for k in range(3):
                print(times(2.5, cases[k][1]), cases[k][1] // 2)
            for row in cases:
                print(row[1] + 1, row[1] * 3)
        """, "columns")

    # ------------------------------------------------ evidence from annotations
    def test_annotations_type_a_def_never_called(self):
        m = messages("""
            def knap(w: int, wt: list[int], n: int) -> int:
                best = 0
                for i in range(n):
                    for j in range(w + 1):
                        if wt[i] <= j:
                            best = best + 1
                return best
        """)
        self.assertEqual(m, [])
        v = tr("""
            #pythran export log_likelihood(float64[], float64, float64)
            import numpy
            def log_likelihood(data, mean, sigma):
                s = (data - mean) ** 2 / (2 * (sigma ** 2))
                return numpy.log(numpy.exp(-s)).sum()
        """)
        self.assertEqual(v["messages"], [])
        self.assertIn("def log_likelihood(data: Arr, mean, sigma)", v["viv"])

    def test_bool_items_beside_numbers_stay_bools(self):
        # the hole: a Bool in a tuple, a list or rows beside numbers printed as 1 or 1.0 (bool-items)
        self.run_diff("""
            t = (True, 1.5)
            print(t[0], t[1])
            L = [1, 2.5, True]
            print(L[2], L[0] + L[1])
            M = [[True, 2.5], [False, 4.0]]
            print(M[0][0], M[1][1])
            for a, b in M:
                print(a, b)
        """, "bool_items")

    def test_a_print_whose_arguments_print(self):
        # diff.py records each print's own values (a hole in the runner fixed 2026-09-29)
        self.run_diff("""
            def loud(x):
                print("in", x)
                return x * 2

            print(loud(1), loud(loud(2.5)), "end")
        """, "nested_print")


class Refusals(unittest.TestCase):
    def refused(self, src, want):
        m = messages(src)
        self.assertTrue(m, "expected a refusal")
        self.assertTrue(any(want in x for x in m), m)

    def test_a_list_and_an_array_that_the_def_changes(self):
        self.refused("import numpy as np\ndef grow(v):\n    v.append(0.5)\n    return v\n"
                     "x = [1.0]\ny = grow(x)\nz = grow(np.array([1.0]))\nprint(x, y, z)\n", ".append")
        # the hole: v * 2 repeats a list and doubles an array (list-array-join)
        self.refused("import numpy as np\ndef double(v):\n    return v * 2\n"
                     "print(double(np.array([1.0, 2.0])), double([1.0, 2.0]))\n", "`*` of a list by a number")

    def test_an_int_or_a_float_decided_at_run_time(self):
        self.refused("def show(x):\n    print(x)\n    return 0\nvals = [1, 2.5]\nfor i in range(2):\n    k = show(vals[i])\n",
                     "printing a number that is an int on some paths and a float on others")
        self.refused("def power(b, e):\n    if e == 0:\n        return 1\n    return b * power(b, e - 1)\nprint(power(1.5, 3))\n",
                     "printing a number that is an int on some paths and a float on others")

    def test_an_annotation_the_call_contradicts(self):
        self.refused("def f(n: int):\n    return n + 1\nprint(f(2.5))\n", "which is annotated `int`")
        self.refused("import numpy as np\ndef f(v: list[float]):\n    return len(v)\nprint(f(np.zeros(3)))\n",
                     "which is annotated `list[float]`")
        # a float annotation takes an int (PEP 484's numeric tower)
        self.assertEqual(messages("def f(x: float):\n    return x * 2\nprint(f(3))\n"), [])

    def test_a_list_parameter_the_caller_still_holds(self):
        self.refused("def f(v):\n    v.append(1.0)\n    return v\nx = [1.0]\ny = f(x)\nprint(x, y)\n", "list.append")
        self.refused("def f(v):\n    v.append(1.0)\n    return v\ndef g(a):\n    return a\nx = [2.0]\nprint(f(g(x)), x)\n", "list.append")
        self.refused("G = [0.25]\ndef h():\n    return G\ndef f(v):\n    v.append(1.0)\n    return v\nprint(f(h()), G)\n", "list.append")
        self.refused("def f(v=[]):\n    v.append(1.0)\n    return len(v)\nprint(f(), f())\n", "an empty list")

    def test_list_items_keep_their_type(self):
        # the hole: an int stored into a list of floats printed 3.0, a float into a list of
        # ints was written as an int (list-item-type)
        self.refused("out = [0.0] * 2\nout[0] = 3\nprint(out)\n", "storing an int into a list of floats")
        self.refused("L = [1, 2]\nL[0] = 2.5\nprint(L)\n", "storing a float into a list of ints")
        self.refused("L = [1, 2]\nL[0] += 0.5\nprint(L)\n", "storing a float into a list of ints")
        # the hole: the ints of rows mixing ints and floats printed as floats (int-float-rows)
        self.refused("M = [[1, 2.5], [3, 4.0]]\nprint(M)\n", "printing a list of ints and floats")

    def test_bool_items_that_cannot_be_read_back(self):
        self.refused("L = [1, 2.5, True]\nprint(L)\n", "printing a list of Bools beside numbers")
        self.refused("t = (True, 1.5)\na, b = t\nprint(a)\n", "unpacking a Bool")

    def test_arrays_of_two_dtypes_do_not_share_a_name(self):
        # the hole: a name that held an int array and a float array printed both as floats
        # (dtype-join)
        self.refused("import numpy as np\nx = np.array([1, 2])\nfor i in range(2):\n    print(x)\n    x = x / 2\n",
                     "holds")

    def test_too_many_signatures(self):
        calls = "\n".join("print(f(%s))" % ", ".join("1" if k >> j & 1 else "1.5" for j in range(4)) for k in range(16))
        self.refused("def f(a, b, c, d):\n    return a + b + c + d\n" + calls + "\n", "called with more than 8 different tuples")

    def test_unknown_values_from_a_refusal_are_cascades(self):
        res = tr("import matplotlib.pyplot as plt\nimport numpy as np\nfig, ax = plt.subplots()\nax.set_xlabel('x')\n"
                 "def draw(a, x):\n    a.plot(x)\n    return 0\nk = draw(ax, np.zeros(3))\nprint(k)\n")
        casc = {r.what: r.cascade for r in res["refusals"]}
        self.assertTrue(casc["the method `.set_xlabel` of a value of unknown type (it comes from a construct refused elsewhere)"])
        self.assertTrue(casc["the method `.plot` of a value of unknown type (it comes from a construct refused elsewhere)"])


if __name__ == "__main__":
    unittest.main()

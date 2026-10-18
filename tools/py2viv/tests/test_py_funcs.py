# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's Python functions and lists (funcs.py): comprehensions and generator expressions,
lambdas, nested defs, calls through function parameters, key functions, list methods, lists of
rows and of tuples, `in` and `==` of lists, reversed slices, text in a print; and the refusals
and run-time stops that keep them exact.

    python3 -m pytest -q tools/py2viv/tests/test_py_funcs.py

The differential tests run each program in CPython and its translation in the Viv interpreter,
in C and in wasm (diff.py), and need a built compiler (VIV=<path> or target/release/viv); the wasm
engine joins them when viv.wasm and node are present. A test passes only when every printed value
agrees bit for bit and the printed text is CPython's byte for byte.
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
    def run_diff(self, src, name, stop=None, text=True):
        """Translate and run src; every value bit for bit, the text byte for byte (text=True),
        the three engines alike. stop: the line both runs end with (Python's exception, or the
        start of a `py2viv:` run-time refusal)."""
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
        if stop is not None and stop.startswith("py2viv:"):
            self.assertTrue((rep["stop"]["viv"] or "").startswith(stop), diff.format_report(rep))
            return rep
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        if text:
            bad = [r for r in rep["lines"] if r["verdict"] != "identical"]
            self.assertEqual(bad, [], diff.format_report(rep))
        if stop is not None:
            self.assertTrue(rep["stop"]["python"].startswith(stop), rep["stop"])
        return rep

    # ------------------------------------------------ comprehensions
    def test_comprehensions(self):
        self.run_diff("""
            import numpy as np
            x = 5
            L = [x * 2 for x in range(3)]
            print(x, L)
            v = np.array([1.5, -2.5, 3.0, 0.25])
            w = [2.0, 4.0, 1.0]
            print([t * 2 for t in v if t > 0], [i + 1 for i in range(10) if i % 3 == 0 if i > 0])
            print([a * b for a, b in zip(v, w)], [i * t for i, t in enumerate(w)], [i * t for i, t in enumerate(w, 10)])
            print([[i * j for j in range(3)] for i in range(2)], [[a, b] for a, b in zip(w, w)])
            print([i * 10 + j for i in range(3) for j in range(i)], [t for t in reversed(w)], [k for k in range(9, 0, -4)])
            print([t > 1.5 for t in w], [i for i in range(0)], [(i, t) for i, t in enumerate(w)][1][1])
            M = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            print([r[0] + r[1] for r in M], [a - b for a, b in M], [[c * 2 for c in r] for r in M if r[0] > 2])
            print(list(range(2, 9, 3)), list(zip([1, 2], [3.5, 4.5]))[1][0], list(enumerate(w))[2][0])
        """, "comprehensions")

    def test_generators(self):
        self.run_diff("""
            w = [2.0, 4.0, 1.0]
            def loud(t):
                print("item", t)
                return t > 1.5
            print(sum(t * t for t in w), sum(i for i in range(5) if i != 2), sum(i * j for i in range(3) for j in range(i)))
            # (a print inside another print's arguments confuses diff.py's recording of values)
            y1 = any(loud(t) for t in w)
            y2 = all(loud(t) for t in w)
            print(y1, y2)
            print(any(i * j > 3 for i in range(4) for j in range(4)), all(i + j < 9 for i in range(4) for j in range(4)))
            print(max(t for t in w), min(abs(t - 2.5) for t in w), max(range(4), key=lambda k: w[k % 3]))
            print(min(w, key=lambda t: (t - 2.2) ** 2), max((t for t in w if t > 9), default=0.5))
            print(next(i for i in range(10) if i * i > 20), next((t for t in w if t > 10), -1.0))
            print(max([3, 1, 3], key=lambda t: -t), min([[1.0, 5.0], [0.5, 9.0]], key=lambda r: r[1]))
        """, "generators")

    def test_empty_max_and_next_stop_as_python_raises(self):
        self.run_diff("""
            w = [2.0, 4.0]
            print(max(t for t in w if t > 3))
            print(max(t for t in w if t > 5))
        """, "max_empty", stop="ValueError: max()")
        self.run_diff("""
            w = [2.0, 4.0]
            print(next(t for t in w if t > 3))
            print(next(t for t in w if t > 5))
        """, "next_empty", stop="StopIteration")

    def test_sorted_is_stable_with_key_and_reverse(self):
        self.run_diff("""
            w = [2.5, -1.0, 2.5, 0.5, -1.0, 3.0]
            idx = list(range(len(w)))
            print(sorted(idx, key=lambda k: w[k]), sorted(idx, key=lambda k: w[k], reverse=True))
            print(sorted(w, reverse=True), sorted([3, 1, 2, 1]), sorted(i % 3 for i in range(7)))
            pairs = [(2, 0.5), (1, 0.25), (2, 0.125), (1, 0.75)]
            s = sorted(pairs, key=lambda p: (p[0], -p[1]))
            print([p[1] for p in s], [p[0] for p in sorted(pairs, key=lambda p: p[0], reverse=True)])
            rows = [[2, 1], [1, 9], [2, 0], [1, 3]]
            print(sorted(rows), sorted(rows, key=lambda r: r[1]))
            zs = [0.0, -0.0, 1.0, -0.0]
            print(sorted(zs, key=abs), sorted(zs, reverse=True))
            v = [0.5, -1.5, 2.5, -0.25]
            v.sort(key=lambda t: abs(t))
            u = v
            v.sort(reverse=True)
            print(v, u)
        """, "sorted")

    def test_sorting_by_a_nan_stops(self):
        self.run_diff("""
            w = [2.5, float("nan"), 1.0]
            print(sorted(w, key=lambda t: t))
        """, "sort_nan", stop="py2viv:")

    # ------------------------------------------------ lambdas and nested defs
    def test_lambdas(self):
        self.run_diff("""
            import math
            sq = lambda x: x * x
            K = 3.0
            scale = lambda x: K * x

            def integrate(f, a, b, n):
                h = (b - a) / n
                s = 0.5 * (f(a) + f(b))
                for i in range(1, n):
                    s += f(a + i * h)
                return s * h

            def twice(g, x):
                return integrate(g, 0.0, x, 4) + g(x)

            def cube(x):
                return x * x * x

            print(sq(3.0), scale(2.0), integrate(sq, 0.0, 1.0, 10), integrate(cube, 0.0, 2.0, 8))
            print(integrate(lambda t: math.sin(t) * K, 0.0, 1.0, 16), twice(cube, 1.5), twice(sq, 2.0))
            K = 4.0
            print(scale(2.0), (lambda a, b=2: a * b)(5))
            print(list(map(lambda t: t + 1, [1, 2, 3])), list(filter(lambda t: t % 2 == 1, range(9))), sum(map(sq, [0.5, 1.5])))
            print(list(map(lambda a, b: a * b, [1.5, 2.0], [4.0, 0.5, 9.0])), list(filter(None, [0, 3, 0, 4])))
            for v in map(lambda t: t * 0.5, [1.0, 3.0]):
                print(v)
        """, "lambdas")

    def test_nested_defs(self):
        self.run_diff("""
            def outer(n, c):
                total = 0.0
                parent = [i for i in range(n)]

                def find(i):
                    if i != parent[i]:
                        parent[i] = find(parent[i])
                    return parent[i]

                def shifted(x):
                    return x + c + total

                def both(x):
                    return shifted(x) + find(n - 1)

                for k in range(1, n):
                    parent[k] = k - 1
                total = 10.0
                r = find(n - 1)
                return r + shifted(1.5) + both(0.25), parent

            r, p = outer(6, 0.25)
            print(r, p)

            def search(hay, item):
                def rec(lo, hi):
                    if hi < lo:
                        return -1
                    mid = lo + (hi - lo) // 2
                    if hay[mid] == item:
                        return mid
                    if item < hay[mid]:
                        return rec(lo, mid - 1)
                    return rec(mid + 1, hi)
                return rec(0, len(hay) - 1)

            print([search([0, 5, 7, 10, 15], q) for q in [0, 7, 15, 6, -1]])
        """, "nested")

    def test_a_function_parameter_that_writes_is_seen_by_the_view_check(self):
        m = messages("""
            import numpy as np
            def w(v):
                v[1] = 9.0
                return 0
            def apply(f, x):
                return f(x)
            a = np.array([1.0, 2.0, 3.0])
            b = a[1:]
            k = apply(w, a)
            print(b)
        """)
        self.assertTrue(m and "a slice of `a`" in m[0], m)

    # ------------------------------------------------ lists
    def test_list_methods(self):
        self.run_diff("""
            stack = [1, 2, 3]
            stack.append(4)
            top = stack.pop()
            stack.pop(0)
            stack.insert(1, 9)
            stack.insert(-10, 7)
            stack.insert(100, 8)
            stack.extend([5, 6])
            stack.extend(range(2))
            stack.remove(9)
            print(top, stack, len(stack), 5 in stack, 11 not in stack, stack.index(5), stack.count(0))
            stack.reverse()
            print(stack)
            stack.sort(reverse=True)
            print(stack, [1, 2] == [1, 2], [1, 2] != [1, 2.0], [1, 2] == [1, 2, 3])
            rows = []
            for i in range(3):
                rows.append([i * 1.0, i * 2.0])
            rows.insert(0, [9.0, 9.5])
            last = rows.pop()
            print(rows, last, len(rows))
            pairs = []
            for i in range(4):
                pairs.append((i, i * 0.5))
            a, b = pairs.pop()
            for i, w in pairs:
                print(i, w, a, b)
            todo = [5]
            seen = []
            while todo:
                x = todo.pop()
                seen.append(x)
                if x > 1:
                    todo.append(x // 2)
                    todo.append(x - 1)
            print(seen)
            q = [3.5, 1.5]
            q.clear()
            q.append(2.5)
            print(q, 2.5 in q)
        """, "list_methods")

    def test_list_errors_stop_as_python_raises(self):
        self.run_diff("""
            s = [1.5]
            s.pop()
            print(s)
            s.pop()
        """, "pop_empty", stop="IndexError: pop from empty list")
        self.run_diff("""
            s = [1, 2, 3]
            s.remove(2)
            print(s)
            s.remove(7)
        """, "remove", stop="ValueError: list.remove(x): x not in list")
        self.run_diff("""
            s = [1, 2, 3]
            print(s.index(3))
            print(s.index(7))
        """, "index", stop="ValueError: ")
        self.run_diff("""
            s = [1.0, float("nan")]
            print(1.0 in s)
            print(s.count(float("nan")))
        """, "nan_in", stop="py2viv:")

    def test_ragged_rows_stop(self):
        self.run_diff("""
            rows = []
            rows.append([1.0, 2.0])
            print(rows)
            rows.append([1.0])
        """, "ragged", stop="py2viv:")

    # ------------------------------------------------ operators, slices, printing
    def test_and_or_whose_right_side_needs_statements(self):
        self.run_diff("""
            def is_safe(board, row, column):
                n = len(board)
                return (
                    all(board[i][j] != 1 for i, j in zip(range(row), [column] * row))
                    and all(board[i][j] != 1 for i, j in zip(range(row - 1, -1, -1), range(column - 1, -1, -1)))
                    and all(board[i][j] != 1 for i, j in zip(range(row - 1, -1, -1), range(column + 1, n)))
                )

            def f(x):
                print("f", x)
                return x > 1.0

            board = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
            print([1 if is_safe(board, 2, c) else 0 for c in range(4)])
            xs = [0.5, 2.0, 3.0]
            if xs[0] > 1.0 or any(f(t) for t in xs):
                print("yes")
            ok = xs[0] > 0.0 and all(f(t) for t in xs)
            print(ok)
        """, "boolop")

    def test_reversed_slices(self):
        self.run_diff("""
            import numpy as np
            v = [1, 2, 3, 4, 5, 6, 7]
            a = np.array([0.5, -1.5, 2.25, 8.0, -0.0])
            print(v[::-1], v[5:1:-2], v[-2::-3], v[:2:-1], v[10:-10:-1], v[1:5:-1])
            print(a[::-1], np.sum(a[::-1]), sum(a[::-2]), np.mean(a[3::-1]))
        """, "rslice")

    def test_text_in_print(self):
        self.run_diff("""
            x = 0.1 + 0.2
            n = 7
            v = [1.5, 2.0]
            print("x = " + str(x) + ", n = " + str(n), str(v), repr(x) + f" and {n:3d}", "ok " + str(True))
            print(str(-0.0) + "|" + str(1e300 * 10) + "|" + str(2 ** 40))
        """, "text")

    def test_nested_unpacking(self):
        self.run_diff("""
            import numpy as np
            xs = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]])
            (ax, ay), (bx, by) = xs[0, :2], xs[1, :2]
            p, (q, r) = 2, (1, 2.5)
            a, b = 1.0, 2.0
            (a, b), c = (b, a), a
            print(ax, ay, bx, by, p, q, r, a, b, c)
        """, "nested_unpack")

    def test_min_max_of_an_int_and_a_float(self):
        self.run_diff("""
            m = max(3, 2.5) + 0.5
            print(m, max(2, 2.5) * 1.0, min(1, 1.0) + 0.25)
        """, "mixed_ok")


class Refusals(unittest.TestCase):
    def refused(self, src, want):
        m = messages(src)
        self.assertTrue(m, "expected a refusal")
        self.assertTrue(any(want in x for x in m), m)

    def test_a_number_that_is_an_int_or_a_float(self):
        self.refused("def h(c):\n    x = 1\n    if c > 0:\n        x = 1.1\n    return x ** 3\nprint(h(1))\n",
                     "of a number that is an int on some paths and a float on others")
        self.refused("print(max(3, 2.5))\n", "printing a number that is an int on some paths and a float on others")

    def test_a_default_python_computed_at_the_def(self):
        # evaluated once where the def runs, into a module variable every call passes (classes.py)
        v = tr("A = 2.0\ndef f(x, a=A):\n    return x * a\nA = 3.0\nprint(f(1.0))\n")
        self.assertEqual(v["messages"], [])
        self.assertIn("py_dflt_f_a = A", v["viv"])
        self.assertIn("f(1.0, py_dflt_f_a)", v["viv"])
        self.assertEqual(messages("import math\nTOL = 1e-3\ndef f(x, a=TOL, b=math.pi, c=-2):\n    return x * a + b + c\n"
                                  "print(f(1.0))\n"), [])

    def test_nested_defs_that_cannot_be_lifted(self):
        self.refused("def f(n):\n    k = 0\n    def g():\n        nonlocal k\n        k += 1\n    g()\n    return k\nprint(f(1))\n",
                     "(`nonlocal`)")
        self.refused("def f(n):\n    def g(x):\n        return x + n\n    return h(g)\ndef h(q):\n    return q(1.0)\nprint(f(1.0))\n",
                     "used as a value")
        self.refused("def f(n):\n    r = g(1.0)\n    def g(x):\n        return x + n\n    return r\nprint(f(1.0))\n",
                     "read before its `def` runs")
        self.refused("def f(n):\n    def g(x):\n        return x + m\n    r = g(1.0)\n    m = 2.0\n    return r\nprint(f(1.0))\n",
                     "which is not assigned before this call")

    def test_a_lambda_that_reads_a_local_passed_to_a_def(self):
        self.refused("def h(q):\n    return q(1.0)\ndef f(k):\n    return h(lambda x: x + k)\nprint(f(2.0))\n",
                     "a lambda that reads `k`")

    def test_rows_other_names_hold(self):
        self.refused("import numpy as np\nA = np.zeros((2, 2))\nL = []\nfor r in A:\n    L.append(r)\nA[0, 0] = 1.0\nprint(L)\n",
                     "a row of `A`, which this code also writes into")
        self.refused("L = []\nM = [[1.0, 2.0], [3.0, 4.0]]\nfor r in M:\n    L.append(r)\nM[0][0] = 5.0\nprint(L)\n",
                     "a row")

    def test_printing_tuples_and_mixed_lists(self):
        self.refused("t = (1, 2.5)\nprint(t)\n", "printing a tuple")
        self.refused("L = [1, 2.5]\nprint(L)\n", "printing a list of ints and floats")
        self.refused("t = (1, 2)\nt[0] = 3\n", "a store into a tuple")

    def test_list_aliases(self):
        self.refused("a = [1, 2]\nb = a\na.pop()\nprint(b)\n", "list.pop")
        self.refused("def f(v):\n    v.pop()\n    return 0\nx = [1.0]\nk = f(x)\n", "list.pop")
        # appending to a parameter changes the caller's list in Python (a hole fixed 2026-09-28)
        self.refused("def f(v):\n    v.append(2.0)\n    return 0\nx = [1.0]\nk = f(x)\nprint(x)\n", "list.append")
        self.refused("L = [1.0]\ndef f():\n    L.append(2.0)\n    return 0\nk = f()\nprint(L)\n", "list.append")

    def test_break_in_a_loop_over_a_generator_of_several_clauses(self):
        self.refused("for t in (i * j for i in range(3) for j in range(3)):\n    if t > 2:\n        break\n",
                     "`break` in a loop over a generator with several `for` clauses")

    def test_sum_of_floats_under_312_in_a_numpy_program(self):
        if py2viv.HOST_PY < (3, 12):
            self.skipTest("the builtin sum of floats is compensated from Python 3.12 on")
        self.refused("import numpy as np\nv = [0.1, 0.2]\nprint(sum(v))\n", "the builtin sum of a list of floats")


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Unit tests for py2viv (standard library only).

    python3 -m unittest discover -s tools/py2viv/tests
    python3 -m pytest tools/py2viv/tests

The tests that run Viv need a built compiler (VIV=<path>, or target/release/viv in this
checkout); the ones that also compare with CPython need NumPy. Each is skipped, with the
reason, when its tool is missing.
"""
import os
import re
import shutil
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

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def viv_bin():
    for p in (os.environ.get("VIV"), os.path.join(ROOT, "target", "release", "viv")):
        if p and os.path.exists(p):
            return p
    return None


def tr(src, **kw):
    return py2viv.translate(textwrap.dedent(src), "t.py", **kw)


def ok(test, src, **kw):
    res = tr(src, **kw)
    test.assertEqual(res["messages"], [], "unexpected refusals")
    return res["viv"]


def refused(test, src, **kw):
    res = tr(src, **kw)
    test.assertTrue(res["refusals"], "expected a refusal")
    return res


class Operators(unittest.TestCase):
    def test_python_operators(self):
        v = ok(self, """
            a = 7
            b = -a % 3
            c = -a // 2
            d = 2.0 ** 0.5
            e = -d ** 2
        """)
        # CPython's floored // and % (py_fdm), an int's negation without a -0, float_pow
        self.assertIn("b = py_mod_i(0 - a, 3)", v)
        self.assertIn("c = py_floordiv_i(0 - a, 2)", v)
        self.assertIn("d = py_fpow(2.0, 0.5)", v)
        self.assertIn("e = -py_fpow(d, 2)", v)

    def test_power_left_operand_is_parenthesized(self):
        # an array's power goes through py_apow (Viv types `a ^ y` of an array a def returned as
        # a number), and an array's negation is -1.0 * a (the same bits, signs of zero included)
        v = ok(self, "import numpy as np\nx = np.zeros(3)\ny = (-x) ** 2\n")
        self.assertIn("y = py_apow(-1.0 * x, 2)", v)

    def test_and_or_precedence(self):
        # Viv's `and` and `or` share a level; Python's `and` binds tighter
        v = ok(self, "a = True\nb = False\nc = True\nd = a or b and c\n")
        self.assertIn("d = a or (b and c)", v)

    def test_not_of_comparison(self):
        v = ok(self, "x = 1.0\ny = not x > 2.0\n")
        self.assertIn("y = not (x > 2.0)", v)

    def test_matmul_of_two_vectors_is_a_dot_product(self):
        v = ok(self, """
            import numpy as np
            u = np.zeros(3)
            w = np.ones(3)
            s = u @ w
            A = np.zeros((3, 3))
            y = A @ u
        """)
        self.assertIn("s = sum(u * w)", v)
        self.assertIn("y = A @ u", v)

    def test_chained_comparison(self):
        v = ok(self, "x = 0.5\ninside = 0.0 < x < 1.0\n")
        self.assertIn("inside = 0.0 < x and x < 1.0", v)

    def test_in_literal_tuple(self):
        v = ok(self, "k = 3\nhit = k in (59, 60)\n")
        self.assertIn("hit = (k == 59 or k == 60)", v)


class Loops(unittest.TestCase):
    def test_range_forms(self):
        v = ok(self, """
            s = 0
            for i in range(10):
                s += i
            for i in range(2, 10):
                s += i
            for i in range(10, 0, -3):
                s += i
            for i in reversed(range(4)):
                s += i
        """)
        self.assertIn("for i to 10 {", v)
        self.assertIn("for i from 2 to 10 {", v)
        self.assertIn("for i from 10 to 0 by -3 {", v)
        self.assertIn("for i to 4 reversed {", v)

    def test_loop_variable_rebound_in_body_uses_a_hidden_counter(self):
        v = ok(self, """
            for i in range(3):
                i = i * 2
                print(i)
        """)
        self.assertRegex(v, r"for py_k\d+ to 3 \{\n    i = py_k\d+")

    def test_enumerate_and_zip(self):
        v = ok(self, """
            import numpy as np
            a = np.zeros(4)
            b = np.ones(4)
            s = 0.0
            for i, x in enumerate(a):
                s += i * x
            for x, y in zip(a, b):
                s += x * y
        """)
        self.assertIn("for i to len(a) {", v)
        self.assertRegex(v, r"for py_i\d+ to min\(len\(a\), len\(b\)\) \{")

    def test_loop_over_rows_with_strings_is_unrolled(self):
        v = ok(self, """
            import numpy as np
            for name, A in (("first", np.eye(2)), ("second", 2.0 * np.eye(2))):
                print(name, A[0, 0])
        """)
        self.assertEqual(len(re.findall(r"^A = ", v, re.M)), 2)
        res = tr("""
import numpy as np
for name, A in (("first", np.eye(2)), ("second", 2.0 * np.eye(2))):
    print(name, A[0, 0])
""")
        self.assertEqual([s["parts"][0]["text"] for s in res["sites"]], ["first", "second"])

    def test_loop_over_numeric_rows(self):
        v = ok(self, """
            s = 0.0
            for h, n in ((0.05, 1), (1.0, 2)):
                s += h * n
        """)
        self.assertRegex(v, r"py_v\d+ = \[0.05, 1.0\]")

    def test_simple_comprehension_is_viv_s_comprehension(self):
        v = ok(self, """
            import numpy as np
            xs = np.linspace(0.0, 1.0, 5)
            ys = [0.5 * x * x for x in xs]
        """)
        self.assertRegex(v, r"ys = \[0.5 \* (py_x\d+) \* \1 for \1 in xs\]")

    def test_a_comprehension_with_more_than_loops_stays_a_loop(self):
        # a list of rows checks their lengths as it grows: statements a
        # comprehension cannot hold
        v = ok(self, "rows = [[i, i + 1] for i in range(3)]\n")
        self.assertRegex(v, r"py_l\d+ = \[\]")

    def test_generator_sum_becomes_an_accumulator(self):
        v = ok(self, "n = 5\ns = sum(i * i for i in range(n))\n")
        # the generator's `i` is its own (Python 3 scoping): a fresh Viv name, not the program's `i`
        self.assertRegex(v, r"py_sum\d+ = 0\nfor (py_i\d+) to n \{\n    py_sum\d+ = py_int\(py_sum\d+ \+ py_int\(\1 \* \1\)\)\n\}")


class Assignment(unittest.TestCase):
    def test_parallel_assignment_uses_temporaries_when_needed(self):
        v = ok(self, "a = 0\nb = 1\na, b = b, a + b\n")
        # Viv's own unpacking computes every right side first
        self.assertIn("a, b = b, py_int(a + b)", v)

    def test_parallel_assignment_without_overlap_is_direct(self):
        v = ok(self, "x, y = 1.0, 2.0\n")
        self.assertIn("x = 1.0\ny = 2.0", v)

    def test_chained_index_is_one_index(self):
        v = ok(self, """
            import numpy as np
            A = np.zeros((2, 2))
            A[1][0] = 3.0
            x = A[1][0]
        """)
        self.assertIn("A[1, 0] = 3.0", v)
        self.assertIn("x = A[1, 0]", v)

    def test_negative_literal_index(self):
        v = ok(self, "import numpy as np\na = np.zeros(4)\nx = a[-1]\n")
        self.assertIn("x = a[-1]", v)

    def test_name_reused_for_another_type_gets_a_new_name(self):
        v = ok(self, """
            import numpy as np
            y = 1.0
            print(y)
            y = np.array([2.0, 0.0])
            print(y)
        """)
        self.assertIn("y_2 = [2.0, 0.0]", v)

    def test_name_reused_inside_a_branch_is_refused(self):
        res = refused(self, """
            import numpy as np
            y = 1.0
            if y > 0:
                y = np.zeros(2)
        """)
        self.assertIn("holds", res["refusals"][0].what)

    def test_keywords_and_builtins_are_renamed(self):
        v = ok(self, "to = 1.0\nlabel = 2.0\npy_x = 3.0\nz = to + label + py_x\n")
        self.assertIn("z = to_ + label_ + u_py_x", v)


class NumpyMapping(unittest.TestCase):
    def test_sort_is_a_copy(self):
        v = ok(self, "import numpy as np\na = np.array([2.0, 1.0])\nb = np.sort(a)\n")
        self.assertIn("b = py_sort(a)", v)
        self.assertIn("    b = a * 1.0\n    sort(b)\n    return b\n}", v)
        self.assertIn("sorting a NaN", v)  # where the two sorts differ, the run stops

    def test_solve_allocates_its_output(self):
        v = ok(self, "import numpy as np\nA = np.eye(2)\nb = np.ones(2)\nx = np.linalg.solve(A, b)\n")
        self.assertIn("x = py_solve(A, b)", v)
        self.assertIn("lin_solve(A, b, x)", v)

    def test_cholesky_and_eigvalsh(self):
        v = ok(self, "import numpy as np\nA = np.eye(2)\nL = np.linalg.cholesky(A)\nw = np.linalg.eigvalsh(A)\n")
        self.assertIn("cholesky(A, L)", v)
        self.assertIn("spectrum_sym(H, w)", v)

    def test_linear_algebra_is_listed_as_inexact(self):
        res = tr("import numpy as np\nA = np.eye(2)\nw = np.linalg.eigvalsh(A)\nx = A @ w\n")
        whats = [x["what"] for x in res["inexact"]]
        self.assertTrue(any("eigvalsh" in w for w in whats) and any(w.startswith("@") for w in whats), whats)

    def test_fft_copies_into_a_complex_array(self):
        v = ok(self, "import numpy as np\nx = np.zeros(8)\nX = np.fft.fft(x)\ny = np.fft.ifft(X)\n")
        self.assertIn("X = py_fft_r(x)", v)
        self.assertIn("y = py_ifft_c(X)", v)
        self.assertIn("X = czeros(len(x))", v)

    def test_constructors(self):
        v = ok(self, """
            import numpy as np
            a = np.zeros((2, 3))
            b = np.ones(4)
            c = np.full(3, 7.0)
            d = np.arange(5)
            e = np.eye(3, k=1)
            f = np.zeros_like(a)
        """)
        self.assertIn("a = zeros(2, 3)", v)
        self.assertIn("b = zeros(4) + 1.0", v)
        self.assertIn("c = py_full(3, 7.0)", v)  # fill keeps a -0.0
        self.assertIn("d = py_arange(0, 5, 1)", v)
        self.assertIn("e = eye(3, 3, 1)", v)
        self.assertIn("f = zeros(len(a), shape(a)[1])", v)

    def test_methods(self):
        v = ok(self, """
            import numpy as np
            a = np.ones((2, 2))
            s = a.sum()
            c = a.copy()
            t = a.T
            m = a.mean()
            n = a.shape[0]
            k = a.size
        """)
        for want in ("s = py_npsum(a)", "c = py_copy(a)", "t = transpose(a)", "m = py_mean(a)", "n = len(a)", "k = numel(a)"):
            self.assertIn(want, v)

    def test_python_exact_round_and_int(self):
        v = ok(self, "x = 2.5\na = round(x)\nb = int(-x)\n")
        self.assertIn("a = py_round(x)", v)
        self.assertIn("b = py_trunc(-x)", v)

    def test_math_constants(self):
        v = ok(self, "import math\nimport numpy as np\na = math.pi\nb = np.e\nc = math.inf\n")
        self.assertIn("a = 3.141592653589793", v)
        self.assertIn("b = 2.718281828459045", v)
        self.assertIn("c = (1.0 / 0.0)", v)


class Tests(unittest.TestCase):
    def test_test_functions_become_test_blocks_at_the_end(self):
        v = ok(self, """
            def f(x):
                return 2.0 * x

            def test_double():
                assert f(2.0) == 4.0
                assert abs(f(0.1) - 0.2) < 1e-15

            print(f(1.0))
        """)
        body = v.split("###")[-1]
        self.assertLess(body.index('print(fmt(f(1.0)))'), body.index('test "test_double" {'))
        # the body is a def, so the first failed assert ends it, as in pytest
        self.assertIn("if not (f(2.0) == 4.0) {\n        assert_true(false)\n        return 0\n    }", v)
        self.assertIn("if not (abs(f(0.1) - 0.2) < 1e-15) {", v)
        self.assertIn('test "test_double" {\n    py_test_test_double()\n}', v)

    def test_isclose_keeps_its_tolerance(self):
        v = ok(self, """
            import math
            def test_c():
                assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-12)
        """)
        self.assertIn("if not (py_isclose(0.1 + 0.2, 0.3, 1e-12, 0.0)) {", v)

    def test_allclose(self):
        v = ok(self, """
            import numpy as np
            def test_a():
                a = np.ones(3)
                assert np.allclose(a, a, atol=1e-12)
        """)
        self.assertIn("if not (py_ballclose(a, a, 1e-05, 1e-12)) {", v)


class Printing(unittest.TestCase):
    def test_print_is_one_fmt_line(self):
        res = tr('import numpy as np\nx = 0.1\nn = 3\nA = np.eye(2)\nprint("x is", x, n, A)\n')
        v = res["viv"]
        self.assertIn('print("x is", fmt(x), fmt(n + 0.0, "d"), fmt(A))', v)
        parts = res["sites"][0]["parts"]
        self.assertEqual(parts[0], {"kind": "str", "text": "x is"})
        self.assertEqual(parts[2]["type"], {"kind": "num", "py": "float"})

    def test_plain_mode(self):
        v = ok(self, "x = 0.1\nprint(x)\n", print_mode="plain")
        self.assertIn("show x", v)


class Refusals(unittest.TestCase):
    def test_comprehension_refusal_format(self):
        # the P0 acceptance: file:line:col, the line, a caret, a hint
        res = refused(self, """
            import numpy as np
            def positive(xs):
                return {x for x in xs if x > 0}
        """)
        msg = res["messages"][0]
        self.assertTrue(msg.startswith("t.py:4:12: not numeric Python: a set comprehension"))
        lines = msg.splitlines()
        self.assertEqual(lines[1].strip(), "return {x for x in xs if x > 0}")
        self.assertEqual(lines[2].index("^"), lines[1].index("{"))
        self.assertIn("key by integer index", lines[3])

    def expect(self, src, what, gap=None, **kw):
        res = refused(self, src, **kw)
        whats = [r.what for r in res["refusals"]]
        self.assertTrue(any(what in w for w in whats), "%r not in %r" % (what, whats))
        if gap:
            self.assertTrue(any(r.gap == gap for r in res["refusals"]))

    def test_class(self):
        # a class translates as a record (tests/test_py_classes.py); inheritance does not
        self.expect("class B:\n    pass\nclass A(B):\n    pass\n", "a class that inherits from `B`")

    def test_dict(self):
        # a dict literal with constant keys translates as a record; one with a key computed at
        # run time is a map (maps.py), of one kind of key
        self.expect("k = 'a'\nd = {k: 2}\nd[3] = 1\n", "a dict whose keys are both numbers and strings")
        # an empty dict is a map; a constant string key is one of its string keys
        self.expect("d = {}\nd['a'] = 1\nd[2] = 1\n", "a dict whose keys are both numbers and strings")

    def test_fstring_spec_outside_fmt_table(self):
        for spec in ("+.3f", ",", ".1%", ".3", "010.2f", "E", "x", "=10"):
            self.expect('x = 1.0\nprint(f"{x:%s}")\n' % spec, "the format spec", gap="R4")

    def test_fstring_on_a_float_with_d(self):
        self.expect('x = 1.0\nprint(f"{x:d}")\n', "the format spec")

    def test_print_end(self):
        self.expect('print("a", end="")\n', "end=")

    def test_slice_written_through(self):
        self.expect("import numpy as np\na = np.zeros(4)\nb = a[1:3]\nb += 1.0\n", "bound to `b`, which this code writes through")

    def test_slice_backward(self):
        # a vector's reversed slice is a copy walked backward (funcs.py); a matrix's is refused
        self.expect("import numpy as np\na = np.zeros((4, 2))\nb = a[::-1]\n", "a slice with the step `-1`")

    def test_tuple_used_as_a_value(self):
        self.expect("def f(x):\n    return x, 2.0 * x\ny = f(1.0) + 1\n", "used as one value", gap="R5")

    def test_tuple_arity_differs(self):
        self.expect("def f(x):\n    if x > 0:\n        return x, x\n    return x\n", "returns a number here and 2 values elsewhere")

    def test_starred_and_nested_targets(self):
        self.expect("a = [1.0, 2.0, 3.0]\nb, *c = a\n", "a starred unpacking target")
        # a nested target of a tuple of values is translated (funcs.py); of one call's values it is not
        self.expect("def f(x):\n    return x, x\n(a, b), c = f(1.0)\n", "a nested unpacking target")

    def test_big_integer(self):
        self.expect("n = 9007199254740993\n", "above 2^53")

    def test_bit_operations_on_floats(self):
        self.expect("a = 5.0\nb = a & 1\n", "a bit operation on a float")

    def test_bit_operations_translate_exactly(self):
        v = ok(self, "a = 5\nb = a & 1\nc = a >> 1\nd = ~a\n")
        self.assertIn("b = py_band(a, 1, 0)", v)
        self.assertIn("c = py_shr(a, 1)", v)

    def test_raise_stops(self):
        v = ok(self, "def f(x):\n    if x < 0:\n        raise ValueError('no')\n    return x\n")
        self.assertIn('stop("ValueError: no")', v)

    def test_raise_in_a_test(self):
        self.expect("def test_a():\n    raise ValueError('x')\n", "`raise` inside a test")

    def test_eigh_vectors(self):
        self.expect("import numpy as np\nA = np.eye(2)\nw, V = np.linalg.eigh(A)\n", "eigenvectors")

    def test_and_or_as_a_value(self):
        self.expect("a = 0\nb = 5\nc = a or b\n", "`and`/`or` of a number used as a value")

    def test_row_of_a_matrix_that_is_written(self):
        self.expect("import numpy as np\nA = np.zeros((2, 2))\nr = A[0]\nA[0, 1] = 5.0\nprint(r)\n", "which this code also writes into")

    def test_int_dtype_of_floats(self):
        self.expect("import numpy as np\na = np.array([1.5, 2.0], dtype=int)\n", "dtype=int")

    def test_printing_an_int_or_a_float(self):
        self.expect("x = 0\nfor i in range(3):\n    x = x + 0.5\nprint(x)\n", "an int on some paths and a float on others")

    def test_try(self):
        self.expect("try:\n    x = 1.0\nexcept ValueError:\n    x = 2.0\n", "`try`")

    def test_np_random(self):
        self.expect("import numpy as np\nx = np.random.rand(3)\n", "np.random")

    def test_scipy(self):
        self.expect("import scipy\n", "`import scipy`")

    def test_use_of_refused_import_names_the_module(self):
        self.expect("import mpmath\nx = mpmath.mpf(1)\n", "a use of `mpmath`")

    def test_list_append(self):
        self.expect("import numpy as np\na = np.zeros(2)\na.append(1.0)\n", "list.append", gap="R17")

    def test_view_write(self):
        self.expect("import numpy as np\nA = np.zeros((2, 2))\nr = A[0]\nr[1] = 5.0\n", "row of another array")

    def test_int_array_truncation(self):
        self.expect("import numpy as np\na = np.array([1, 2])\na[0] = 0.5\n", "NumPy integer array")

    def test_every_refusal_is_collected(self):
        res = refused(self, "class A(object, int):\n    pass\nd = {1, 2}\nimport scipy\n")
        self.assertEqual(len(res["refusals"]), 3)


def run_python(src):
    """CPython's stdout and, if it raised, the traceback's last line."""
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(src)], capture_output=True, text=True, timeout=120)
    exc = p.stderr.strip().splitlines()[-1] if p.returncode else None
    return p.stdout, exc


def run_viv(test, src):
    """The translation's stdout in `viv run`, and its `error:` message if it stopped."""
    res = tr(src)
    test.assertEqual(res["messages"], [], "unexpected refusals")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.viv")
        with open(p, "w") as f:
            f.write(res["viv"])
        r = subprocess.run([viv_bin(), "run", "--quiet", p], capture_output=True, text=True, timeout=300)
    err = None
    for line in r.stderr.splitlines():
        if line.startswith("error: "):
            err = line[len("error: "):]
            break
    out = "\n".join(l for l in r.stdout.splitlines() if not l.startswith("__")) + ("\n" if r.stdout.strip() else "")
    return out, err


@unittest.skipUnless(viv_bin(), "needs a built viv (VIV=<path>)")
class RunsInViv(unittest.TestCase):
    """Python-exact semantics: the translation prints CPython's text, byte for byte, and stops
    where CPython raises, with the same exception line."""

    def same(self, src):
        want, wexc = run_python(src)
        got, gexc = run_viv(self, src)
        self.assertEqual(got, want)
        self.assertEqual(gexc, wexc)

    def test_python_exact_arithmetic(self):
        exprs = ["-7 % 3", "7.5 % -2", "-7 // 2", "-7.5 // 2", "round(2.5)", "round(-2.5)", "round(0.5)",
                 "round(1.5)", "round(-0.5)", "round(2.675)", "int(-2.5)", "int(2.9)", "2 ** 10", "abs(-3)",
                 "max(1, 5, 3)", "min(2.5, -1.0)", "0 * -1", "-0.0", "0.0 % -5.0", "-0.0 // 3.0", "5.0 % -5.0",
                 "1e300 * 10", "0.1 + 0.2", "1 / 3", "7 // -2", "-7 % -2", "2 ** 52 + 1", "-(3 - 3)",
                 "round(-0.4)", "int(-0.5)", "max(-0.0, 0.0)", "min(0.0, -0.0)", "float('nan') > 1"]
        self.same("".join("print(%s)\n" % e for e in exprs))

    def test_fstrings_byte_for_byte(self):
        self.same('''
            import math
            x = 1.0 / 3.0
            n = 42
            big = 12345.678
            print(f"x = {x}, n = {n}, repr {x!r}, {x=}")
            print(f"{x:.3f} {x:.6g} {x:.3e} {x:g} {x:e} {x:f} {big:10.4f}|{n:5d}|{x:8}|{n:8}")
            print(f"{-0.0} {math.inf} {-math.inf} {math.nan} {1e16} {1e-5} {123456789.0} {0.0001}")
            print("sep", x, n, sep="--")
            print()
            print([1.0, 2.5], [1, 2])
        ''')

    def test_tuples(self):
        self.same('''
            def divmod2(a, b):
                q = a // b
                return q, a - q * b

            def fwd(a):
                return divmod2(a, 3)

            q, r = divmod2(17, 5)
            print(q, r)
            x, y = 1.5, -2
            x, y = y, x
            print(x, y)
            v = [3.0, 1.0, 2.0]
            v[0], v[2] = v[2], v[0]
            print(v)
            a, b = fwd(-10)
            print(a, b)
            lo, hi = [-1.0, 4.0]
            print(hi - lo)
            q2, r2 = divmod(-7, 2)
            print(q2, r2)
        ''')

    def test_shape_unpacking(self):
        if not HAVE_NUMPY:
            self.skipTest("needs NumPy")
        self.same('''
            import numpy as np
            X = np.zeros((2, 3, 4))
            B, H, W = X.shape
            print(B, H, W)
        ''')

    def test_raise_stops_with_the_exception_line(self):
        self.same('''
            def f(x):
                if x < 0:
                    raise ValueError("x must be positive")
                return x
            print(f(2.0))
            print(f(-1.0))
            print("never")
        ''')

    def test_assert_stops(self):
        self.same("x = 3\nprint(x)\nassert x < 2, 'too big'\nprint('never')\n")

    def test_zero_division(self):
        for e in ("1 / 0", "1.0 / 0.0", "1 // 0", "1.0 // 0", "5 % 0", "5.0 % 0.0"):
            self.same("a = 0\nprint('before')\nprint(%s)\n" % e.replace("0.0", "(a * 1.0)").replace(" 0", " a"))

    def test_math_domain(self):
        for e in ("math.sqrt(-1.0)", "math.log(0.0)", "math.log(-2.0)", "math.acos(2.0)", "math.exp(1000.0)",
                  "math.sin(math.inf)", "int(math.nan)", "round(math.inf)", "math.floor(math.nan)", "0.0 ** -1.0"):
            self.same("import math\nprint('before')\nprint(%s)\n" % e)

    def test_integers_past_2_53_stop(self):
        src = "x = 2 ** 52\nprint(x)\ny = x * 2\nprint(y)\n"
        got, err = run_viv(self, src)
        self.assertEqual(got, "4503599627370496\n")
        self.assertIn("2^53", err)

    def test_bit_operations_match_python(self):
        vals = [0, 1, 2, 3, 5, 255, 2 ** 26 - 1, 2 ** 26, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 40 + 12345,
                2 ** 52 - 1, -1, -2, -3, -255, -(2 ** 31), -(2 ** 31) - 1, -(2 ** 32), -(2 ** 52) + 7]
        lines = ["vals = %r" % vals, "for a in vals:",
                 "    for b in vals:", "        print(a & b, a | b, a ^ b)",
                 "    print(~a, a >> 0, a >> 1, a >> 31, a >> 32, a >> 60, a >> 200)",
                 "    for k in (0, 1, 3):", "        if -(2 ** 49) < a < 2 ** 49:", "            print(a << k)"]
        self.same("\n".join(lines) + "\n")

    def test_shift_past_2_53_and_negative_shift(self):
        got, err = run_viv(self, "a = 2 ** 50\nprint(a << 3)\n")
        self.assertIn("2^53", err)
        self.same("a = 5\nk = -1\nprint(a)\nprint(a << k)\n")

    def test_min_max_follow_python(self):
        self.same('''
            nan = float("nan")
            print(max([1.0, nan, 2.0]), max([nan, 1.0]), min([-0.0, 0.0]), max([0.0, -0.0]))
            print(max(-0.0, 0.0), max(0.0, -0.0), min(nan, 1.0), min(1.0, nan))
        ''')

    def test_builtin_sum_order(self):
        # 3.12 compensates a sum of floats; 3.11 adds left to right: the translation follows the host
        self.same("xs = [1e16, 1.0, -1e16, 1e-3, 3.0, -2.5e15, 7.0]\nprint(sum(xs))\nprint(sum(x * 1.0 for x in xs))\n")


@unittest.skipUnless(viv_bin() and HAVE_NUMPY, "needs viv and NumPy")
class NumPyOrder(unittest.TestCase):
    """NumPy's pairwise sum and its mean, var, std, linspace, arange, clip, maximum: bit for bit."""

    def test_reductions(self):
        src = '''
            import numpy as np
            for n in (1, 7, 8, 9, 15, 16, 127, 128, 129, 200, 255, 256, 257, 1000, 3001):
                x = np.zeros(n)
                for i in range(n):
                    x[i] = ((i * 7919) % 1009 - 504.5) * 1.0e-3 + 1.0 / (i + 1)
                print(np.sum(x), x.sum(), np.mean(x), np.var(x), np.std(x), sum(x), np.max(x), np.min(x))
                print(np.cumsum(x).tolist()[-1], np.prod(x))
            A = np.zeros((13, 11))
            for i in range(13):
                for j in range(11):
                    A[i, j] = 1.0 / (i + 2 * j + 1)
            print(np.sum(A), np.trace(A), np.prod(A))
            print(np.linspace(-1.3, 2.7, 17).tolist())
            print(np.arange(0.1, 2.0, 0.3).tolist())
            y = np.array([-2.0, -0.0, 0.5, 3.0, np.nan])
            print(np.clip(y, 0.0, 1.0).tolist(), np.maximum(y, 0.0).tolist(), np.minimum(y, -0.0).tolist())
        '''
        RunsInViv.same(self, src)


@unittest.skipUnless(viv_bin() and HAVE_NUMPY and shutil.which("cc"), "needs viv, NumPy and a C compiler")
class Differential(unittest.TestCase):
    def test_adapters_example(self):
        import diff
        with tempfile.TemporaryDirectory() as d:
            rep = diff.differential(os.path.join(PKG, "examples", "adapters.py"), d,
                                    engines=("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ()))
        self.assertTrue(rep["verdict"].startswith("PASS"), diff.format_report(rep))
        self.assertTrue(rep["engines_agree"])

    def differential(self, src, name):
        import diff
        engines = ("interp", "c") + (("wasm",) if diff.default_wasm() and shutil.which("node") else ())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name + ".py")
            with open(path, "w") as f:
                f.write(textwrap.dedent(src))
            return diff.differential(path, os.path.join(d, "out"), engines=engines), diff

    def test_bit_operations_three_engines(self):
        vals = [0, 1, 7, 2 ** 26 - 1, 2 ** 26, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 52 - 1,
                -1, -2, -(2 ** 31), -(2 ** 31) - 1, -(2 ** 32), -(2 ** 52) + 7]
        src = "vals = %r\nfor a in vals:\n    for b in vals:\n        print(a & b, a | b, a ^ b)\n" \
              "    print(~a, a >> 1, a >> 33, a >> 100)\n" % vals
        rep, diff = self.differential(src, "bits")
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertTrue(rep["raw_identical"])

    def test_a_stop_in_three_engines(self):
        src = """
            import numpy as np
            def f(x):
                if x > 2.0:
                    raise ValueError("too large")
                return 1.0 / (x - 1.0)
            for x in (0.0, 0.5, 2.0):
                print(f(x), f"{x:.2f}", x == 0.0)
            print(np.sum(np.linspace(0.0, 1.0, 11)))
            print(f(1.0))
        """
        rep, diff = self.differential(src, "stops")
        self.assertEqual(rep["verdict"], "PASS (bitwise)", diff.format_report(rep))
        self.assertEqual(rep["stop"]["python"], "ZeroDivisionError: float division by zero")
        self.assertTrue(rep["raw_identical"])


if __name__ == "__main__":
    unittest.main()


class VivModules(unittest.TestCase):
    """`import m` and `from m import f` of a Viv module of the tree map to the same Viv lines;
    a call of one of its defs takes the type its signature declares."""

    def test_a_lib_module_is_found_as_the_compiler_finds_it(self):
        t = py2viv.Translator("x = 1\n", os.path.join(ROOT, "t.py"))
        self.assertIn("rmsnorm", t._viv_module("nn"))
        self.assertIsNone(t._viv_module("no_such_module_here"))

    def test_import_line_is_written_and_a_declared_def_is_called(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "helpmod.viv"), "w") as f:
            f.write("def twice(x) -> Num {\n    return 2.0 * x\n}\ndef raw(x) {\n    return x\n}\n")
        res = py2viv.translate("from helpmod import twice\nimport helpmod as h\nprint(twice(2.0))\nprint(h.twice(3.0))\n", os.path.join(d, "p.py"))
        self.assertEqual(res["messages"], [])
        self.assertIn("from helpmod import twice\n", res["viv"])
        self.assertIn("import helpmod as h\n", res["viv"])
        self.assertIn("twice(2.0)", res["viv"])
        self.assertIn("h.twice(3.0)", res["viv"])
        bad = py2viv.translate("from helpmod import raw\nprint(raw(2.0))\n", os.path.join(d, "p.py"))
        self.assertTrue(bad["refusals"])
        shutil.rmtree(d)

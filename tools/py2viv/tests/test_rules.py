# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Tests of the rule engine (rule_engine.py and rules/*.toml). Standard library only.

    python3 -m pytest tools/py2viv/tests/test_rules.py
"""
import os
import re
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import py2viv  # noqa: E402
import rule_engine  # noqa: E402


def tr(src, **kw):
    return py2viv.translate(textwrap.dedent(src), "t.py", print_mode="plain", **kw)


def viv(test, src):
    res = tr(src)
    test.assertEqual(res["messages"], [], "unexpected refusals")
    return res["viv"]


def messages(src):
    return tr(src)["messages"]


PRE = "import numpy as np\nimport math\nv = np.array([1.0, 2.0, 3.0])\nM = np.eye(3)\nf = 2.5\n"
# the line PRE's next statement is on
L = 6


class Binding(unittest.TestCase):
    def test_positional_and_keyword_bind_the_same_parameter(self):
        self.assertIn("a = py_full(3, 2.0)", viv(self, PRE + "a = np.full(3, 2.0)\n"))
        self.assertIn("a = py_full(3, 2.0)", viv(self, PRE + "a = np.full(3, fill_value=2.0)\n"))

    def test_a_default_fills_an_omitted_parameter(self):
        self.assertIn("a = py_linspace(0, 1, 50)", viv(self, PRE + "a = np.linspace(0, 1)\n"))
        self.assertIn("a = eye(3, 3, 0)", viv(self, PRE + "a = np.eye(3)\n"))
        self.assertIn("a = eye(3, 3, 1)", viv(self, PRE + "a = np.eye(3, k=1)\n"))

    def test_binding_refuses_an_argument_past_the_parameters(self):
        m = messages(PRE + "a = np.linspace(0, 1, 5, 6)\n")
        self.assertEqual(len(m), 1)
        self.assertIn("t.py:%d:26: not numeric Python: np.linspace with a fourth positional argument" % L, m[0])
        m = messages(PRE + "a = np.diff(v, 2)\nb = abs(f, 2)\nc = np.sqrt(v, v)\n")
        self.assertIn("np.diff of a matrix, or of order n", m[0])
        self.assertIn("abs with a second positional argument", m[1])
        self.assertIn("np.sqrt with a second positional argument", m[2])

    def test_binding_refuses_a_keyword_the_function_does_not_take(self):
        m = messages(PRE + "a = np.sqrt(v, where=v)\nb = np.cumsum(v, dtype=float)\nc = np.sqrt(x=v)\n")
        self.assertIn("np.sqrt with `where=`", m[0])
        self.assertIn("np.sqrt without its argument `x`", m[2])
        self.assertIn("np.cumsum with `dtype=`", m[1])

    def test_a_missing_argument_is_a_located_refusal(self):
        m = messages(PRE + "a = np.linspace()\nb = np.full(3)\n")
        self.assertIn("t.py:%d:5: not numeric Python: np.linspace without its argument `start`" % L, m[0])
        self.assertIn("np.full without its argument `fill_value`", m[1])

    def test_checked_binding_collects_unknown_keywords(self):
        v = viv(self, PRE + "a = math.isclose(f, 2.0, rel_tol=1e-3)\n")
        self.assertIn("a = py_isclose(f, 2.0, 0.001, 0.0)", v)

    def test_a_positional_dtype_binds_the_dtype(self):
        # np.zeros(3, int): NumPy's second parameter is the dtype
        res = tr(PRE + "a = np.zeros(3, int)\nprint(a)\n")
        self.assertEqual(res["messages"], [])
        self.assertIn("a = zeros(3)", res["viv"])
        self.assertIn("fmt(a, \"d\")", py2viv.translate(textwrap.dedent(PRE + "a = np.zeros(3, int)\nprint(a)\n"), "t.py")["viv"])

    def test_an_unpacking_rule_rewrites_into_python(self):
        # q, r = divmod(a, b) is translated as q, r = a // b, a % b
        v = viv(self, PRE + "n = 7\nq, r = divmod(n, 3)\n")
        self.assertIn("q = py_floordiv_i(n, 3)\nr = py_mod_i(n, 3)", v)
        m = messages(PRE + "n = 7\nq, r, s = divmod(n, 3)\n")
        self.assertIn("unpacking divmod into 3 names", m[0])

    def test_an_unpacking_rule_refuses(self):
        m = messages(PRE + "w, V = np.linalg.eigh(M)\n")
        self.assertIn("t.py:%d:8: not numeric Python: np.linalg.eigh's eigenvectors" % L, m[0])

    def test_a_rule_can_hoist_a_temporary(self):
        v = viv(self, PRE + "a = math.sinh(f)\n")
        self.assertIn("py_x1 = f\na = py_mhyp(sinh(py_x1), py_x1)", v)

    def test_a_rewrite_into_python_translates_the_new_expression(self):
        # pow(a, b) is a ** b: the operator's translation (py_fpow for floats)
        self.assertIn("a = py_fpow(f, 0.5)", viv(self, PRE + "a = pow(f, 0.5)\n"))


class Templates(unittest.TestCase):
    def test_argument_is_parenthesized_by_precedence(self):
        v = viv(self, PRE + "a = np.square(v + 1.0)\nb = np.square(v)\n")
        self.assertIn("a = (v + 1.0) * (v + 1.0)", v)
        self.assertIn("b = v * v", v)

    def test_result_precedence_is_the_rules(self):
        # np.ones is `zeros(n) + 1.0` at precedence 60: a product parenthesizes it
        self.assertIn("a = (zeros(3) + 1.0) * 2.0", viv(self, PRE + "a = np.ones(3) * 2.0\n"))

    def test_expression_holes_and_braces(self):
        eng = py2viv.RULES
        ctx = rule_engine.Ctx(eng, eng.rules[0], None, None, "np.linalg.solve")
        self.assertEqual(rule_engine.render("{short}{{x}}{len(name)}", ctx), "solve{x}15")

    def test_helpers_are_emitted_with_their_dependencies(self):
        v = viv(self, PRE + "a = np.prod(v)\n")
        self.assertIn("def py_prod(a: Arr)", v)
        self.assertIn("def py_flat(a: Arr)", v)
        self.assertIn("def py_iota(n)", v)

    def test_a_family_member_translates(self):
        v = viv(self, PRE + "a = np.sqrt(v)\nb = math.sqrt(f)\nc = np.floor(f)\n")
        self.assertIn("a = sqrt(v)", v)
        self.assertIn("b = py_msqrt(f)", v)
        self.assertIn("c = floor(f)", v)

    def test_inexact_operations_reach_the_manifest(self):
        res = tr(PRE + "a = np.exp(v)\nb = np.sqrt(v)\nc = np.linalg.solve(M, v)\n")
        whats = [x["what"] for x in res["inexact"]]
        self.assertEqual(whats, ["exp (Viv's libm is not the platform's)",
                                 "np.linalg.solve (Viv's own LU, Cholesky, Jacobi or FFT, not LAPACK or pocketfft)"])


class Refusals(unittest.TestCase):
    def test_a_refuse_rule_gives_its_message_hint_and_gap(self):
        m = messages(PRE + "a = np.tile(v, 2)\n")
        self.assertEqual(len(m), 1)
        self.assertIn("not numeric Python: np.tile (lifted by R10)", m[0])
        self.assertIn("Allocate the result with np.zeros and fill it in a loop", m[0])

    def test_a_case_refuses_on_a_type(self):
        m = messages(PRE + "a = np.linalg.norm(v, 2)\nb = len(f)\n")
        self.assertIn("np.linalg.norm with ord= or axis=", m[0])
        self.assertIn("len of a number", m[1])

    def test_a_keywords_step_refuses_at_the_keyword(self):
        m = messages(PRE + "a = np.eye(3, order='C')\n")
        self.assertEqual(len(m), 1)
        self.assertIn("t.py:%d:21: not numeric Python: np.eye(..., order=)" % L, m[0])
        self.assertIn("\n                        ^^^\n", m[0])

    def test_a_dtype_is_refused_before_the_argument_is_translated(self):
        m = messages(PRE + "a = np.array({1: 2}, np.float32)\n")
        self.assertEqual(len(m), 1)
        self.assertIn("dtype=np.float32", m[0])

    def test_star_patterns_come_after_exact_names(self):
        m = messages(PRE + "a = np.random.poisson(3.0)\n")
        self.assertIn("not numeric Python: np.random.poisson\n", m[0])
        # a translating rule names every function it maps: another module path in the middle
        # reaches no rule (np.linalg.sqrt does not exist; np.emath.sqrt is complex-valued)
        m = messages(PRE + "a = np.linalg.sqrt(v)\nb = np.emath.sqrt(f)\nc = math.abs(f)\n")
        self.assertIn("`np.linalg.sqrt`", m[0])
        self.assertIn("Not in the NumPy mapping yet", m[0])
        self.assertIn("`np.emath.sqrt`", m[1])
        self.assertIn("`math.abs`", m[2])

    def test_unknown_function_is_refused_as_before(self):
        m = messages(PRE + "a = np.foo(v)\nb = foo(v)\nc = math.foo(f)\n")
        self.assertEqual(m[0].splitlines()[0], "t.py:%d:5: not numeric Python: `np.foo`" % L)
        self.assertEqual(m[0].splitlines()[-1], "  Not in the NumPy mapping yet (tools/py2viv/SUBSET.md lists what is).")
        self.assertEqual(m[1].splitlines()[0], "t.py:%d:5: not numeric Python: a call of `foo`" % (L + 1))
        self.assertEqual(m[1].splitlines()[-1], "  Not a def in this file and not in the numeric subset.")
        self.assertIn("`math.foo`", m[2])

    def test_unknown_method_and_attribute_are_refused_as_before(self):
        m = messages(PRE + "a = v.item()\nb = v.flat\nc = v.sum(1)\n")
        self.assertIn("the method `.item`", m[0])
        self.assertIn("Viv has no methods; use the NumPy function form (np.item(a)).", m[0])
        self.assertIn("the attribute `.flat`", m[1])
        self.assertIn("sum with axis=1 of a rank-1 array", m[2])

    def test_a_declining_rule_passes_to_the_next(self):
        # math.log with two arguments is its own rule; with one, the elementwise family's;
        # with three, neither, and the unknown-function refusal
        v = viv(self, PRE + "a = math.log(f, 2.0)\nb = math.log(f)\n")
        self.assertIn("a = py_div_f(py_mlog(f), py_mlog(2.0))", v)
        self.assertIn("b = py_mlog(f)", v)
        self.assertIn("math.log with a second positional argument", messages(PRE + "c = math.log(f, 2.0, 3.0)\n")[0])
        # np.log's second argument is `out`, not a base
        self.assertIn("np.log with a second positional argument", messages(PRE + "c = np.log(v, v)\n")[0])


class Files(unittest.TestCase):
    MINIMAL = {
        "helpers.toml": "",
        "modules.toml": "",
        "numpy.toml": "",
        "strings.toml": "",
    }

    def engine(self, builtins):
        d = tempfile.mkdtemp()
        for name, text in dict(self.MINIMAL, **{"builtins.toml": builtins}).items():
            with open(os.path.join(d, name), "w") as f:
                f.write(textwrap.dedent(text))
        return rule_engine.Engine(d, types=py2viv.RULES.globals)

    def test_a_valid_file_loads(self):
        e = self.engine('''
            [[rule]]
            match = "twice(x)"
            exactness = "exact"
            cases = [{ viv = "2.0 * {x:71}", type = "x" }]
        ''')
        self.assertEqual(e.summary(), {"exact": 1})

    def test_malformed_rules_are_refused_at_load(self):
        bad = [
            ('match = "f(x)"\nexactness = "exact"\ncases = [{ viv = "{x}", type = "x" }]\ncolour = 1', "unknown field"),
            ('match = "f(x)"\nexactness = "roughly"\ncases = [{ viv = "{x}", type = "x" }]', "exactness must be"),
            ('match = "f(x)"\nexactness = "tolerance"\ncases = [{ viv = "{x}", type = "x" }]', "states rtol and atol"),
            ('match = "f(x)"\nexactness = "refuse"\ncases = [{ viv = "{x}", type = "x" }]', "no translating case"),
            ('match = "f(x)"\nexactness = "exact"\ncases = [{ refuse = "f", hint = "h" }]', "needs a translating case"),
            ('match = ["f(x)", "a.f"]\nexactness = "exact"\ncases = [{ viv = "{x}", type = "x" }]', "different kinds"),
            ('match = "f(x"\nexactness = "exact"\ncases = [{ viv = "{x}", type = "x" }]', "bad pattern"),
            ('match = "f(x)"\nexactness = "refuse"\ncases = [{ refuse = "f", hint = "h" }]', "binds no arguments"),
        ]
        for body, err in bad:
            with self.assertRaises(rule_engine.RuleError) as cm:
                self.engine("[[rule]]\n" + body + "\n")
            self.assertIn(err, str(cm.exception))

    def test_expressions_are_whitelisted(self):
        ctx = rule_engine.Ctx(py2viv.RULES, py2viv.RULES.rules[0], None, None, "f")
        for src in ["__import__('os')", "(lambda: 1)()", "name.__class__", "[y for y in name]", "open('x')"]:
            with self.assertRaises(rule_engine.RuleError, msg=src):
                rule_engine.evaluate(src, ctx)


class TheRuleFiles(unittest.TestCase):
    def test_every_rule_states_its_class_and_evidence(self):
        for r in py2viv.RULES.rules:
            self.assertIn(r.exactness, rule_engine.EXACTNESS, r.id)
            if r.exactness == "refuse":
                self.assertTrue(r.gap or r.lift, "%s: a refuse rule says what would lift it" % r.id)
            else:
                self.assertTrue(r.evidence, "%s: no evidence" % r.id)
            if r.exactness == "tolerance":
                self.assertIsNotNone(r.rtol, r.id)
                self.assertIsNotNone(r.atol, r.id)

    def test_every_helper_a_rule_uses_exists(self):
        # a templated name (py_{short}, {mhelper}) must name a helper for some spelling of the
        # rule; a family case guarded by `mpath == 'x'` only for the members whose mpath is x
        for r in py2viv.RULES.rules:
            for c in r.cases:
                m = re.search(r"mpath == '(\w+)'", c.get("if", ""))
                if m and r.vars.get("mpath") != m.group(1):
                    continue
                for u in c.get("use", []):
                    names = set()
                    for p in r.patterns:
                        h = u.replace("{short}", p.key.split(".")[-1]).replace("{name}", p.key)
                        if isinstance(r.vars.get("names"), dict):
                            # an operator rule: its helper per operator (the pattern's key)
                            h = h.replace("{names[op]}", r.vars["names"].get(p.key, ""))
                        for k, val in r.vars.items():
                            h = h.replace("{%s}" % k, str(val))
                        names.add(h)
                    self.assertTrue(any(h in py2viv.PRELUDE for h in names), "%s uses %s" % (r.id, sorted(names)))

    def test_helper_dependencies_exist(self):
        for name, (deps, _) in py2viv.PRELUDE.items():
            for d in deps.split():
                self.assertIn(d, py2viv.PRELUDE, "%s depends on %s" % (name, d))

    def test_inexact_steps_belong_to_tolerance_rules_or_say_so(self):
        for r in py2viv.RULES.rules:
            steps = [c for c in r.cases if "inexact" in c]
            if r.exactness == "tolerance":
                self.assertTrue(steps, "%s: a tolerance rule records its inexact operation" % r.id)
            for c in steps:
                # a refuse rule may record the operation before it refuses (np.linalg.det does)
                self.assertTrue(r.exactness in ("tolerance", "refuse") or c.get("exactness") == "tolerance",
                                "%s: an inexact step in an exact rule states its own class" % r.id)

    def test_ids_are_unique(self):
        ids = [r.id for r in py2viv.RULES.rules]
        self.assertEqual(len(ids), len(set(ids)))


class FixedHoles(unittest.TestCase):
    """RULES.md, "Fixed holes": each hole the rules first reproduced, and how it is closed."""

    def test_precedence_of_a_pass_through(self):
        v = viv(self, PRE + "x = 2.5\nn = 3\na = np.transpose(v + 1.0) * 2.0\nb = np.real(x + 2.0) * 3.0\n"
                           "c = (v + 1.0).transpose() * 2.0\nd = round(n + 1) * 2\ne = math.floor(n + 1) * 2\n")
        self.assertIn("a = (v + 1.0) * 2.0", v)
        self.assertIn("b = (x + 2.0) * 3.0", v)
        self.assertIn("c = (v + 1.0) * 2.0", v)
        self.assertIn("d = py_int(py_int(n + 1) * 2)", v)
        self.assertIn("e = py_int(py_int(n + 1) * 2)", v)

    def test_np_log_second_argument_is_out(self):
        m = messages(PRE + "a = np.log(v, v)\n")
        self.assertIn("np.log with a second positional argument", m[0])
        self.assertIn("NumPy's second argument is `out`", m[0])

    def test_a_module_path_in_the_middle_is_refused(self):
        m = messages(PRE + "a = np.emath.sqrt(-1.0)\nb = np.linalg.sqrt(v)\nc = np.lib.scimath.log(f)\n")
        self.assertEqual(len(m), 3)
        self.assertIn("`np.emath.sqrt`", m[0])
        self.assertIn("`np.linalg.sqrt`", m[1])

    def test_every_argument_binds_or_is_refused(self):
        m = messages(PRE + "a = np.diff(v, 2)\nb = abs(f, 2)\nc = np.zeros(3, np.float32)\n")
        self.assertIn("np.diff of a matrix, or of order n", m[0])
        self.assertIn("abs with a second positional argument", m[1])
        self.assertIn("dtype=np.float32", m[2])

    def test_vdot_conjugates_and_norm_of_complex(self):
        v = viv(self, PRE + "X = np.zeros(3, dtype=complex)\na = np.vdot(X, X)\nb = np.linalg.norm(X)\nc = np.dot(X, X)\n")
        self.assertIn("a = py_cvdot(X, X)", v)
        self.assertIn("p = conj(x) * y", v)
        self.assertIn("b = py_cnorm(X)", v)
        self.assertIn("c = py_cdot(X, X)", v)

    def test_ones_like_and_full_like_of_complex(self):
        v = viv(self, PRE + "X = np.zeros(3, dtype=complex)\na = np.ones_like(X)\nb = np.full_like(X, 2.0)\n")
        self.assertIn("a = py_cfillin(czeros(len(X)), 1.0)", v)
        self.assertIn("b = py_cfillin(czeros(len(X)), 2.0)", v)

    def test_array_of_an_array_with_a_dtype(self):
        m = messages(PRE + "a = np.array(v, dtype=int)\n")
        self.assertIn("np.array(a, dtype=int) of a rank-1 array of floats", m[0])
        v = viv(self, PRE + "vi = np.array([1, 2])\na = np.array(vi, dtype=float)\nb = np.asarray(v)\n")
        self.assertIn("a = py_copy(vi)", v)
        self.assertIn("b = v", v)

    def test_sign_of_an_array_keeps_nan(self):
        v = viv(self, PRE + "a = np.sign(v)\n")
        self.assertIn("a = py_signs(v)", v)
        self.assertIn("where(a != a, a, (a > 0.0) - (a < 0.0))", v)

    def test_arctan2_of_arrays_is_an_array(self):
        res = py2viv.translate(textwrap.dedent(PRE + "a = np.arctan2(v, v)\nprint(a)\n"), "t.py")
        self.assertEqual(res["messages"], [])
        self.assertIn("a = py_atan2v(v, v)", res["viv"])
        man = py2viv.manifest(res, "t.py", PRE, "exact")
        self.assertEqual(man["sites"][-1]["parts"][0]["type"]["kind"], "arr")

    def test_a_missing_argument_and_a_syntax_error_are_refusals(self):
        m = messages(PRE + "a = np.linspace()\n")
        self.assertIn("np.linspace without its argument `start`", m[0])
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "bad.py")
            with open(src, "w") as f:
                f.write("x = (\n")
            man = os.path.join(d, "bad.json")
            self.assertEqual(py2viv.main([src, "--manifest", man]), 2)
            import json
            self.assertIn("a syntax error", json.load(open(man))["refusals"][0]["what"])

    def test_an_argument_a_template_writes_twice_may_not_call_a_def(self):
        src = PRE + "def g(x):\n    print(x)\n    return x\n\na = np.square(g(v))\nb = np.square(v + 1.0)\n"
        m = messages(src)
        self.assertIn("np.square of an expression that calls `g`", m[0])
        self.assertEqual(len(m), 1)

    def test_a_view_bound_to_a_name_may_not_be_written(self):
        m = messages(PRE + "B = M.T\nB[0, 1] = 5.0\n")
        self.assertIn("a write through `B`, which is a view of another array", m[0])

    def test_int_results_are_typed_as_numpy_types_them(self):
        res = py2viv.translate(textwrap.dedent(PRE + "vi = np.array([1, -2])\na = np.abs(vi)\nprint(a)\n"
                                                     "b = np.full(2, 7)\nprint(b)\nc = np.trunc(3)\nprint(c)\n"), "t.py")
        self.assertEqual(res["messages"], [])
        self.assertIn('print(fmt(a, "d"))', res["viv"])
        self.assertIn('print(fmt(b, "d"))', res["viv"])
        self.assertIn("print(fmt(c))", res["viv"])

    def test_a_negative_size_stops_as_numpy(self):
        v = viv(self, PRE + "n = 3\na = np.zeros(n)\nb = np.zeros(len(v))\nc = np.zeros((2, 3))\n")
        self.assertIn("a = zeros(py_size(n))", v)
        self.assertIn("b = zeros(len(v))", v)
        self.assertIn("c = zeros(2, 3)", v)
        self.assertIn('stop("ValueError: negative dimensions are not allowed")', v)

    def test_builtins_of_a_matrix_are_refused(self):
        m = messages(PRE + "a = sum(M)\nb = max(M)\nc = list(M)\nd = abs([1.0, 2.0])\ne = float(v)\n")
        self.assertIn("the builtin sum of a rank-2 array", m[0])
        self.assertIn("max of a rank-2 array", m[1])
        self.assertIn("list of a rank-2 array", m[2])
        self.assertIn("abs of a list", m[3])
        self.assertIn("float of a rank-1 array", m[4])

    def test_math_of_an_array_is_refused(self):
        m = messages(PRE + "a = math.sqrt(v)\nb = math.fabs(v)\n")
        self.assertIn("math.sqrt of a rank-1 array", m[0])
        self.assertIn("math.fabs of a rank-1 array", m[1])

    def test_list_and_shape_arithmetic_and_members_are_refused(self):
        m = messages(PRE + "L = [1.0, 2.0]\na = L - 0.5\nb = M.shape * 2\nc = L.sum()\nd = -L\n")
        self.assertIn("arithmetic on a Python list", m[0])
        self.assertIn("arithmetic on a shape", m[1])
        self.assertIn("the method `.sum` of a Python list", m[2])
        self.assertIn("negating a list", m[3])

    def test_array_negation_and_power_build_in_viv(self):
        v = viv(self, PRE + "a = np.linspace(0.0, 1.0, 5)\nb = -a\nc = a ** 2\n")
        self.assertIn("b = -1.0 * a", v)
        self.assertIn("c = py_apow(a, 2)", v)

    def test_a_hoisted_expression_after_a_call_with_effects_is_refused(self):
        src = PRE + "def g(x):\n    print(x)\n    return x\n\nc = True\na = g(1.0) + (g(2.0) if c else 3.0)\nb = (g(2.0) if c else 3.0) + g(1.0)\n"
        m = messages(src)
        self.assertEqual(len(m), 1)
        self.assertIn("an expression py2viv evaluates before the rest of its statement, after a call of `g`", m[0])

    def test_np_clip_ignores_a_nan_bound(self):
        v = viv(self, PRE + "a = np.clip(v, -1.0, 1.0)\n")
        self.assertIn("a = py_clipv(v, -1.0, 1.0)", v)


if __name__ == "__main__":
    unittest.main()

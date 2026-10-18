# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""py2viv's dicts with keys computed at run time (maps.py): each translates to a Viv map
exactly, or is refused.

    python3 -m pytest -q tools/py2viv/tests/test_py_maps.py

The differential tests run CPython with NumPy and the translation in the Viv interpreter, C
and wasm (diff.py), and compare every printed value bit for bit, and a run that raises must
stop with Python's exception; they need a built compiler (VIV=<path> or target/release/viv)
and, for wasm, a viv.wasm with the map runtime and node.
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
        return rep

    def test_counting_and_reading(self):
        self.run_diff("""
            d = {}
            for x in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]:
                d[x] = d.get(x, 0) + 1
            print(d)
            print(len(d), 5 in d, 7 in d, 7 not in d)
            total = 0
            for k in d:
                total += k * d[k]
            print(total, sorted(d), sum(d.values()), max(d), min(d.keys()), list(d.values()))
            print(f"counts: {d}")
            if d:
                print("not empty")
            e = {}
            if not e:
                print("empty", e)
        """, "count")

    def test_keys_compare_as_python_compares_them(self):
        # 0.0 and -0.0 are one key and the first stored is kept; 1 and 1.0 are one key;
        # ints up to 2**53 are exact keys
        self.run_diff("""
            d = {}
            d[0.0] = 1.5
            d[-0.0] = 2.5
            print(d)
            e = {}
            e[-0.0] = 1.0
            e[0.0] = 3.0
            print(e, e[0.0], e[-0.0])
            f = {}
            f[1] = 1
            f[1.0] = 2
            print(len(f), f[1], f[1.0])
            g = {}
            g[2 ** 52] = 1
            g[2 ** 52 + 1] = 2
            g[-9007199254740991] = 3
            g[9007199254740991] = 4
            print(g, g[2 ** 52 + 1])
            h = {}
            h[0.1] = 1.0
            h[1e300] = 2.0
            h[float("inf")] = 3.0
            h[-float("inf")] = 4.0
            h[5e-324] = 5.0
            print(h, h[0.1 + 0.0])
            print(float("nan") in h, h.get(float("nan"), -1.0))
        """, "keys")

    def test_a_missing_key_raises_key_error(self):
        self.run_diff("""
            d = {1: 2, 3: 4}
            print(d[3])
            print(d[5])
        """, "keyerror")

    def test_a_missing_float_key_raises_key_error(self):
        self.run_diff("""
            d = {}
            d[2.5] = 1.0
            x = d[2.5] + d[0.5]
            print(x)
        """, "keyerror_float")

    def test_del_and_pop_of_a_missing_key(self):
        self.run_diff("""
            d = {1: 2.0}
            print(d.pop(1), d.pop(1, -1.0), d)
            del d[1]
        """, "del_missing")

    def test_removal_and_reinsertion_order(self):
        self.run_diff("""
            d = {}
            for i in range(20):
                d[i * 7 % 20] = i
            for i in range(0, 20, 3):
                del d[i]
            d[0] = 100
            d[7] = -1
            print(d, list(d), len(d))
            print(d.pop(14), d.pop(99, 0), d)
            d.clear()
            d[5] = 5
            print(d)
        """, "order")

    def test_setdefault_update_copy_and_dict(self):
        self.run_diff("""
            a = {1: 1.5}
            print(a.setdefault(1, 9.0), a.setdefault(2, 2.5), a)
            b = {3: 0.5}
            b.update(a)
            print(b)
            c = b.copy()
            c[4] = 4.0
            d = dict(b)
            d[5] = 5.0
            print(b, c, d)
        """, "methods")

    def test_iteration_while_changing_the_size_raises(self):
        self.run_diff("""
            d = {1: 2, 3: 4}
            for k in d:
                if k == 3:
                    d[k + 10] = 0
            print(d)
        """, "grow_in_loop")

    def test_iteration_while_removing_raises(self):
        self.run_diff("""
            d = {1: 2, 3: 4, 5: 6}
            for k, v in d.items():
                if v == 4:
                    d.pop(1)
            print(d)
        """, "shrink_in_loop")

    def test_iteration_that_keeps_the_size_is_python_s(self):
        # values changed in place, a break before the change of size, a loop over list(d)
        self.run_diff("""
            d = {1: 2, 3: 4, 5: 6}
            for k in d:
                d[k] = d[k] * 10
            print(d)
            n = 0
            for k, v in d.items():
                if v > 20:
                    break
                n += 1
            print(n)
            for k in d:
                if k == 3:
                    d[k + 100] = 1
                    break
            print(d)
            for k in list(d):
                if k == 3:
                    del d[k]
            print(d)
            for i, k in enumerate(d):
                print(i, k)
            for k in sorted(d, reverse=True):
                print(k)
        """, "loops")

    def test_comprehensions_and_zip(self):
        self.run_diff("""
            m = {k: v for k, v in zip([1, 2, 3], [0.5, 1.5, 2.5])}
            print(m)
            print({k: v * 2 for k, v in m.items() if k != 2})
            print([k * 10 for k in m], [v for v in m.values()], sum(v for v in m.values()))
            sq = {i: i * i for i in range(5)}
            print(sq, sq[4])
            z = dict(zip([4, 5, 4], [1.0, 2.0, 3.0]))
            print(z)
        """, "comprehensions")

    def test_defaultdict_and_counter(self):
        self.run_diff("""
            from collections import defaultdict, Counter
            import collections
            dd = defaultdict(int)
            for x in [1, 2, 1, 3, 1]:
                dd[x] += 1
            print(dict(dd), dd[7], len(dd))
            fd = collections.defaultdict(float)
            fd[0.5] += 1.5
            print(dict(fd))
            c = Counter([5, 5, 6])
            print(c[5], c[9], len(c), dict(c))
            c[9] += 2
            print(dict(c))
        """, "counters")

    def test_maps_through_defs_and_aliases(self):
        self.run_diff("""
            def tally(d, xs):
                for x in xs:
                    d[x] = d.get(x, 0) + 1

            def squares(n):
                out = {}
                for i in range(n):
                    out[i] = i * i
                return out

            memo = {}

            def fib(n):
                if n < 2:
                    return n
                if n in memo:
                    return memo[n]
                memo[n] = fib(n - 1) + fib(n - 2)
                return memo[n]

            t = {}
            tally(t, [7, 7, 8])
            alias = t
            alias[9] = 1
            print(t, squares(4), fib(60), len(memo))
        """, "defs")

    def test_int_values_computed_as_negative_zero_print_as_python_s(self):
        self.run_diff("""
            d = {}
            x = 0
            d[-x] = -x
            d[1] = 0 * -1
            print(d)
        """, "negzero")

    def test_number_keyed_literal_used_as_a_map(self):
        self.run_diff("""
            d = {1: 10, 2: 20}
            for i in range(3, 6):
                d[i] = i * 10
            k = 2
            print(d[k], d, 4 in d)
        """, "literal_map")

    def test_two_dicts_in_one_name(self):
        # a def called with two dicts, a variable bound to either, a def returning either
        self.run_diff("""
            def total(d):
                s = 0
                for k in d.keys():
                    s += d[k]
                return s

            def build(n):
                if n > 2:
                    m = {i: i for i in range(n)}
                else:
                    m = {}
                    m[7] = 1
                return m

            a = {}
            a[1] = 2
            b = {3: 4}
            b[5] = 6
            print(total(a), total(b), build(4), build(1))
            c = a
            c = b
            print(c, a)
        """, "unify")

    def test_class_holding_a_dict(self):
        self.run_diff("""
            class Graph:
                def __init__(self):
                    self.deg = {}

                def add(self, a, b):
                    self.deg[a] = self.deg.get(a, 0) + 1
                    self.deg[b] = self.deg.get(b, 0) + 1

            g = Graph()
            g.add(1, 2)
            g.add(2, 3)
            print(g.deg)
        """, "class_dict")

    # ---- maps and infer.py's call-site typing together
    def test_a_def_called_with_a_dict_and_with_a_list(self):
        # one clone per kind of argument: `for k in c` iterates the dict's keys in one and the
        # list's items in the other
        self.run_diff("""
            def total(c):
                s = 0
                for k in c:
                    s += k
                return s
            d = {}
            for i in range(4):
                d[i * 2] = i
            print(total(d))
            print(total([1, 2, 3]), total([-4, 5]))
            print(total(d), len(d))
        """, "map_or_list")

    def test_a_map_through_a_cloned_def(self):
        # add is cloned for an int and a float x; both clones take the one map type the two
        # dicts passed for m became, and change the caller's dict
        self.run_diff("""
            def add(m, k, x):
                m[k] = m.get(k, 0.0) + float(x)
                return x * 2
            a = {}
            b = {}
            for i in range(3):
                a[i] = i * 1.0
                b[i + 10] = i / 4
            print(add(a, 1, 2), add(a, 5, 1.5), add(b, 11, 3))
            print(a)
            print(b)
            print(len(a), len(b), 5 in a, 5 in b)
        """, "map_cloned_def")

    def test_a_map_parameter_annotated_dict_int_float(self):
        self.run_diff("""
            def show(m: dict[int, float], k: int) -> float:
                return m.get(k, -1.0) + 0.5
            d = {}
            for i in range(3):
                d[i] = i * 0.25
            print(show(d, 2), show(d, 7))
        """, "map_annotated")

    def test_a_def_returning_a_map_called_from_two_places(self):
        # the two calls' dicts are one map type (the dict the def makes), and two values
        self.run_diff("""
            def hist(xs):
                h = {}
                for x in xs:
                    h[x] = h.get(x, 0) + 1
                return h
            p = hist([1, 2, 2, 3])
            q = hist([5, 5, 5])
            p[9] = 4
            print(p)
            print(q)
            print(len(p), len(q), 9 in q)
        """, "map_returned_twice")


    def test_string_keys_and_values(self):
        # the tokenizer of study/compare/ml/tiny_gpt_torch.py, word counts, string values
        self.run_diff("""
            TEXT = "hello world! " * 5
            vocab = sorted(set(TEXT))
            stoi = {c: i for i, c in enumerate(vocab)}
            encode = lambda text: [stoi[c] for c in text]
            decode = lambda ids: "".join(vocab[i] for i in ids)
            ids = encode("hello")
            print(stoi, ids, decode(ids), len(stoi), "h" in stoi, "z" in stoi)
            counts = {}
            for w in "the cat and the hat and the bat".split():
                counts[w] = counts.get(w, 0) + 1
            print(counts, sorted(counts), list(counts.values()))
            for k, v in counts.items():
                if v > 1:
                    print(k, v)
            names = {z: s for z, s in zip([1, 8, 6], ["H", "O", "C"])}
            print(names, names[8])
            upper = {s: s.upper() for s in ["a", "b"] if s != "c"}
            print(upper)
            e = {}
            e["x"] = 1.5
            e["y"] = e["x"] * 2
            print(e, e["y"], "x" in e)
        """, "string_maps")

    def test_a_missing_string_key_raises_key_error(self):
        rep = self.run_diff("""
            stoi = {c: i for i, c in enumerate("ab")}
            print([stoi[c] for c in "ab"])
            print([stoi[c] for c in "abc"])
        """, "string_keyerror")
        self.assertTrue(rep["stop"]["python"].startswith("KeyError"), rep["stop"])

    def test_a_dict_comprehension_reads_as_viv_s(self):
        out = tr("""
            vocab = sorted(set("hello"))
            stoi = {c: i for i, c in enumerate(vocab)}
            sq = {i: i * i for i in range(5) if i % 2 == 0}
            print(stoi, sq)
        """)["viv"]
        self.assertIn("vocab = unique(", out)
        self.assertIn(" in enumerate(vocab)}", out)
        self.assertRegex(out, r"sq = \{.+: .+ for \w+ to 5 if .+\}")


class Refused(unittest.TestCase):
    def assertRefused(self, src, words):
        ms = messages(src)
        self.assertTrue(any(words in m for m in ms), ms)

    def test_string_keys(self):
        # both kinds of key in one dict
        self.assertRefused("""
            d = {}
            d["a"] = 1
            d[2] = 1
        """, "both numbers and strings")
        self.assertRefused("""
            d = {c: 1 for c in "ab"}
            d[3] = 2
        """, "both numbers and strings")

    def test_list_values(self):
        self.assertRefused("""
            d = {}
            d[1] = [1, 2]
        """, "a dict value that is")

    def test_bool_keys(self):
        self.assertRefused("""
            d = {}
            d[True] = 1
        """, "a dict key that is a bool")

    def test_get_without_a_default(self):
        self.assertRefused("""
            d = {}
            d[1] = 2
            x = d.get(1)
        """, "without a default")

    def test_items_outside_a_loop(self):
        self.assertRefused("""
            d = {}
            d[1] = 2
            x = d.items()
        """, "outside a for loop")

    def test_a_loop_that_adds_and_removes(self):
        self.assertRefused("""
            d = {1: 2, 3: 4}
            for k in d:
                d[k + 1] = 0
                del d[k]
        """, "both add and remove")

    def test_printing_a_counter(self):
        self.assertRefused("""
            from collections import Counter
            c = Counter([1, 1, 2])
            print(c)
        """, "printing a Counter")

    def test_defaultdict_of_lists(self):
        self.assertRefused("""
            from collections import defaultdict
            d = defaultdict(list)
            d[1] = 2
        """, "defaultdict whose default")

    def test_int_and_float_keys_printed(self):
        # Python keeps the key first stored (1, not 1.0); which one that is is known only at run time
        self.assertRefused("""
            f = {}
            f[1] = 1
            f[1.0] = 2
            print(f)
        """, "keys are ints on some paths and floats")

    def test_mixed_int_and_float_values_printed(self):
        self.assertRefused("""
            d = {}
            d[1] = 2
            d[2] = 2.5
            print(d)
        """, "ints on some paths and floats")

    # ---- maps and infer.py's call-site typing together
    def test_a_map_parameter_annotated_with_other_value_types(self):
        self.assertRefused("""
            def show(m: dict[int, int], k: int) -> float:
                return m.get(k, -1) + 0.5
            d = {}
            for i in range(3):
                d[i] = i * 0.25
            print(show(d, 2))
        """, "a dict whose values are floats for `m`, which is annotated `dict[int, int]`")

    def test_a_list_for_a_parameter_annotated_as_a_dict(self):
        self.assertRefused("""
            def first(m: dict[int, float]) -> float:
                return m[0]
            print(first([1.0, 2.0]))
        """, "which is annotated `dict[int, float]`")

    def test_a_dict_for_a_parameter_annotated_as_a_list(self):
        self.assertRefused("""
            def first(m: list[float]) -> float:
                return m[0]
            d = {}
            for i in range(3):
                d[i] = i * 0.25
            print(first(d))
        """, "`first` called with a dict for `m`, which is annotated `list[float]`")

    def test_a_dict_and_a_counter_for_one_parameter(self):
        # two kinds of dict meeting in one name are refused, as in one variable
        self.assertRefused("""
            from collections import Counter
            def n(m):
                return len(m)
            a = {}
            a[1] = 2
            c = Counter([3, 3, 4])
            print(n(a), n(c))
        """, "and elsewhere with a")

    def test_a_dict_and_a_list_for_a_def_that_is_not_cloned(self):
        # a def used as a value is one def: the dict and the list cannot share its parameter
        self.assertRefused("""
            def size(c):
                return len(c)
            g = size
            d = {}
            d[1] = 2.0
            print(size(d), size([1, 2, 3]), g([1]))
        """, "and elsewhere with")

    def test_one_def_returning_maps_of_ints_and_of_floats(self):
        # the clones for an int and a float store into the one map the def makes: its values
        # are ints in one caller's dict and floats in the other's, which py2viv cannot tell apart
        self.assertRefused("""
            def mk(x):
                d = {}
                d[0] = x
                return d
            p = mk(1)
            q = mk(2.5)
            print(p)
        """, "ints on some paths and floats")


if __name__ == "__main__":
    unittest.main()

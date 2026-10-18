#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""corpus.py: run py2viv and the differential runner over the two correctness corpora,
and rank the refusals.

    python3 tools/py2viv/corpus.py posts           posts/refs/*.py that import neither SciPy nor torch
    python3 tools/py2viv/corpus.py thealgorithms   the functions study/idioms/thealgorithms-python ported
    python3 tools/py2viv/corpus.py all --out DIR

TheAlgorithms is read from the shallow clone the study uses, at commit b3233c9
(`../src/thealgorithms-python` beside this checkout, or TA_SRC=<path>). Its code needs
Python 3.12 (PEP 695 type parameters), so that corpus is translated and run with
PY312=<path> (default /opt/homebrew/bin/python3.12), which does not need NumPy.

Each unit is translated once: py2viv has one mode, in which a translation behaves exactly as
CPython runs the Python or is refused. A unit that translates is run by diff.py in three
engines against CPython, compared bit for bit unless a tolerance is stated (POST_TOL, with
the reason).
For TheAlgorithms, each translated function's printed values are also compared with the
hand-written Viv port's output (study/idioms/thealgorithms-python/out/<part>.values).
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import py2viv  # noqa: E402
import diff  # noqa: E402

# Stated tolerances for corpus programs whose translation uses an operation Viv does not compute
# bit for bit as NumPy/LAPACK/libm (the manifest's `inexact`), with the reason. Everything
# else is compared bit for bit.
POST_TOL = {
    "make-the-heat-equation-step-stable": (1e-8, 0.0, "mode m = 19 decays to 3.4e-9 over 16 solves by Viv's LU, "
                                                         "not LAPACK's; the ratio of two such values carries the difference (4.6e-9 relative)"),
    "make-euler-stable-on-a-stiff-equation": (1e-12, 0.0, "Viv's exp is not the platform libm's"),
}


# ---------------------------------------------------------------- refusal categories

RULES = [
    (r"^`\w+` holds .*", "a variable holds two types (outside straight-line code)"),
    (r"^a use of `\w+` \(from `([\w.]+)`.*", r"a use of a refused import (\1)"),
    (r"^`\w+` called with .*", "a function called with two kinds of argument"),
    (r"^`\w+` returns .*", "a function returns two types"),
    (r"^a call of `\w+`$", "a call of a name that is neither a def nor in the mapping"),
    (r"^\d+ indices into a rank-\d+ array$", "more indices than the array's rank"),
    (r"^`import (\w+)[\w.]*`$", r"import \1"),
    (r"^`from (\w+)[\w.]* import \w+`$", r"import \1"),
    (r"^`global \w+` rebinding.*", "`global` rebinding a module-level name"),
    (r"^the method `\.(\w+)` on .*", r"a method on a non-array (.\1)"),
    (r"^a call missing .*", "a call missing an argument"),
]


def category(what):
    for pat, rep in RULES:
        if re.match(pat, what):
            return re.sub(pat, rep, what)
    return what


def rank(units):
    """units: list of dicts with 'refusals' (list of manifest refusal dicts). Returns rows
    (category, files, occurrences, gap), most files first; cascades counted apart."""
    files = {}
    occ = {}
    gap = {}
    casc = {}
    for u in units:
        seen = set()
        for r in u.get("refusals", []):
            c = category(r["what"])
            if r.get("cascade"):
                casc[c] = casc.get(c, 0) + 1
                continue
            occ[c] = occ.get(c, 0) + 1
            gap.setdefault(c, r.get("gap"))
            if c not in seen:
                files[c] = files.get(c, 0) + 1
                seen.add(c)
    rows = sorted(files, key=lambda c: (-files[c], -occ[c], c))
    return [(c, files[c], occ[c], gap.get(c)) for c in rows], casc


# ---------------------------------------------------------------- posts/refs


def posts_units():
    d = os.path.join(ROOT, "posts", "refs")
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"):
            continue
        src = open(os.path.join(d, f)).read()
        if re.search(r"^\s*(import|from)\s+(scipy|torch)\b", src, re.M):
            continue
        out.append(os.path.join(d, f))
    return out


def run_posts(out_dir, python=None):
    results = []
    for path in posts_units():
        name = os.path.splitext(os.path.basename(path))[0]
        src = open(path).read()
        row = {"unit": name, "lines": src.count("\n")}
        res = py2viv.translate(src, path)
        man = py2viv.manifest(res, path, src, "exact")
        row["strict"] = {"translates": not res["refusals"], "refusals": man["refusals"], "inexact": man["inexact"]}
        if row["strict"]["translates"]:
            rtol, atol, why = POST_TOL.get(name, (None, None, "none declared: bit for bit"))
            rep = diff.differential(path, os.path.join(out_dir, "posts", name),
                                    rtol=rtol, atol=atol, python=python)
            row["diff"] = {"verdict": rep["verdict"], "rtol": rep["rtol"], "atol": rep["atol"],
                           "tolerance_reason": why, "engines_agree": rep.get("engines_agree")}
        results.append(row)
        print("%-72s translates %-3s %s" % (name, "yes" if row["strict"]["translates"] else "no",
                                           row.get("diff", {}).get("verdict", "")), flush=True)
    return results


# ---------------------------------------------------------------- TheAlgorithms


def ta_src():
    for p in (os.environ.get("TA_SRC"), os.path.join(ROOT, "..", "src", "thealgorithms-python"),
              os.path.join(ROOT, "..", "viv-wt", "src", "thealgorithms-python")):
        if p and os.path.isdir(p):
            return os.path.normpath(p)
    return None


# The functions the hand-written ports cover (count_lines.py's PORTED), one unit each.
PORTED = {
    "sorting": {
        "sorts/merge_sort.py": ["merge_sort"],
        "sorts/quick_sort.py": ["quick_sort"],
        "sorts/heap_sort.py": ["heapify", "heap_sort"],
        "sorts/insertion_sort.py": ["insertion_sort"],
        "searches/binary_search.py": ["binary_search", "bisect_left", "binary_search_by_recursion"],
    },
    "graphs": {
        "graphs/breadth_first_search.py": ["Graph"],
        "graphs/depth_first_search.py": ["depth_first_search"],
        "graphs/dijkstra.py": ["dijkstra"],
        "graphs/kahns_algorithm_topo.py": ["topological_sort"],
        "data_structures/disjoint_set/disjoint_set.py": ["Node", "make_set", "union_set", "find_set"],
        "graphs/minimum_spanning_tree_kruskal.py": ["kruskal"],
    },
    "dp": {
        "dynamic_programming/knapsack.py": ["mf_knapsack", "knapsack", "knapsack_with_example_solution", "_construct_solution"],
        "dynamic_programming/longest_common_subsequence.py": ["longest_common_subsequence"],
        "dynamic_programming/edit_distance.py": ["EditDistance"],
    },
    "strings": {
        "strings/knuth_morris_pratt.py": ["knuth_morris_pratt", "get_failure_array"],
        "strings/palindrome.py": ["is_palindrome", "is_palindrome_traversal", "is_palindrome_recursive", "is_palindrome_slice", "is_palindrome_ignore_case_and_spaces"],
        "strings/check_anagrams.py": ["check_anagrams"],
    },
    "maths": {
        "maths/prime_sieve_eratosthenes.py": ["prime_sieve_eratosthenes"],
        "maths/greatest_common_divisor.py": ["greatest_common_divisor", "gcd_by_iterative"],
        "maths/binary_exponentiation.py": ["binary_exp_recursive", "binary_exp_iterative", "binary_exp_mod_recursive", "binary_exp_mod_iterative"],
    },
    "structures": {
        "data_structures/stacks/stack.py": ["Stack"],
        "data_structures/queues/queue_by_list.py": ["QueueByList"],
        "data_structures/heap/heap.py": ["Heap"],
        "data_structures/linked_list/singly_linked_list.py": ["Node", "LinkedList"],
        "data_structures/hashing/hash_map.py": ["_Item", "_DeletedItem", "HashMap"],
    },
    "backtracking": {
        "backtracking/n_queens.py": ["is_safe", "solve"],
        "backtracking/all_permutations.py": ["generate_all_permutations", "create_state_space_tree"],
    },
}

# Drivers: the inputs ref.py gives each function, printed under the port's labels, written
# in the subset. A function without a driver is translated (for its refusals) but not run.
SORT_DATA = "[0, 5, 3, 2, 2, -45, -2, -5, 103, 9, 1, 7, 11]"
PAIRS = "[[24, 40], [1, 1], [1, 800], [11, 37], [3, 5], [16, 4], [-3, 9], [9, -3], [3, -9], [-3, -9], [0, 0], [1, -800]]"
EXP_CASES = "[[3, 5], [11, 13], [-1, 3], [0, 5], [3, 1], [3, 0], [1.5, 4]]"
MOD_CASES = "[[3, 4, 5], [11, 13, 7], [1.5, 4, 3], [7, 13, 10]]"
SEARCH = '''
hay = [0, 5, 7, 10, 10, 10, 15]
queries = [0, 5, 7, 10, 15, 6, -1, 16]
out = [0] * 8
for k in range(8):
    out[k] = {call}
print("{label}", out)
'''
DRIVERS = {
    "merge_sort": 'print("by_merge", merge_sort(list(%s)))' % SORT_DATA,
    "quick_sort": 'print("by_quick", quick_sort(list(%s)))' % SORT_DATA,
    "heap_sort": 'print("by_heap", heap_sort(list(%s)))' % SORT_DATA,
    "heapify": 'print("by_heap", heap_sort(list(%s)))' % SORT_DATA,
    "insertion_sort": 'print("by_insertion", insertion_sort(list(%s)))' % SORT_DATA,
    "binary_search": SEARCH.format(call="binary_search(hay, queries[k])", label="found"),
    "bisect_left": SEARCH.format(call="bisect_left(hay, queries[k])", label="lefts"),
    "binary_search_by_recursion": SEARCH.format(call="binary_search_by_recursion(hay, queries[k], 0, len(hay) - 1)", label="recursive"),
    "greatest_common_divisor": '''
pairs = %s
out = [0] * 12
for k in range(12):
    out[k] = greatest_common_divisor(pairs[k][0], pairs[k][1])
print("gcd_recursive", out)
''' % PAIRS,
    "gcd_by_iterative": '''
pairs = %s
out = [0] * 12
for k in range(12):
    out[k] = gcd_by_iterative(pairs[k][0], pairs[k][1])
print("gcd_iterative", out)
''' % PAIRS,
    "binary_exp_recursive": '''
cases = %s
out = [0.0] * 7
for k in range(7):
    out[k] = binary_exp_recursive(cases[k][0], cases[k][1])
print("exp_recursive", out)
''' % EXP_CASES,
    "binary_exp_iterative": '''
cases = %s
out = [0.0] * 7
for k in range(7):
    out[k] = binary_exp_iterative(cases[k][0], cases[k][1])
print("exp_iterative", out)
''' % EXP_CASES,
    "binary_exp_mod_recursive": '''
cases = %s
out = [0.0] * 4
for k in range(4):
    out[k] = binary_exp_mod_recursive(cases[k][0], cases[k][1], cases[k][2])
print("exp_mod_recursive", out)
print("bench_mod_recursive", binary_exp_mod_recursive(1269380576, 374, 34))
''' % MOD_CASES,
    "binary_exp_mod_iterative": '''
cases = %s
out = [0.0] * 4
for k in range(4):
    out[k] = binary_exp_mod_iterative(cases[k][0], cases[k][1], cases[k][2])
print("exp_mod_iterative", out)
print("bench_mod_iterative", binary_exp_mod_iterative(1269380576, 374, 34))
''' % MOD_CASES,
    "prime_sieve_eratosthenes": 'print("primes_10", prime_sieve_eratosthenes(10))\nprint("primes_100", prime_sieve_eratosthenes(100))',
    "is_safe": '''
board = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
safe = [0] * 4
for c in range(4):
    safe[c] = 1 if is_safe(board, 2, c) else 0
print("is_safe_row2", safe)
''',
    "kruskal": 'print("mst1", kruskal(4, [[0, 1, 3], [1, 2, 5], [2, 3, 1]]))',
    "get_failure_array": 'print("failure", get_failure_array("aabaabaaa"))',
    "knuth_morris_pratt": '''
print("kmp_1", knuth_morris_pratt("alskfjaldsabc1abc1abc12k23adsfabcabc", "abc1abc12"))
print("kmp_2", knuth_morris_pratt("alskfjaldsk23adsfabcabc", "abc1abc12"))
print("kmp_3", knuth_morris_pratt("ABABZABABYABABX", "ABABX"))
print("kmp_4", knuth_morris_pratt("ABAAAAAB", "AAAB"))
print("kmp_5", knuth_morris_pratt("abcxabcdabxabcdabcdabcy", "abcdabcy"))
print("kmp_finds", [knuth_morris_pratt("knuth_morris_pratt", s) for s in ["kn", "h_m", "rr", "tt", "not there"]])
''',
    # the strings of TheAlgorithms' own test_data (strings/palindrome.py), each function's answers as 0/1
    **{name: '''
keys = ["MALAYALAM", "String", "rotor", "level", "A", "BB", "ABC", "amanaplanacanalpanama", "abcdba", "AB"]
out = [0] * 10
for k in range(10):
    out[k] = 1 if %s(keys[k]) else 0
print("%s", out)
''' % (name, name) for name in ("is_palindrome", "is_palindrome_traversal", "is_palindrome_recursive", "is_palindrome_slice")},
    "is_palindrome_ignore_case_and_spaces": '''
keys = ["A man a plan a canal Panama", "Was it a car or a cat I saw?", "Hello World", "Never Odd or Even", ""]
out = [0] * 5
for k in range(5):
    out[k] = 1 if is_palindrome_ignore_case_and_spaces(keys[k]) else 0
print("loose_palindromes", out)
print("loose_first", is_palindrome_ignore_case_and_spaces(keys[0]), keys[0].lower(), keys[0].replace(" ", ""))
''',
    "check_anagrams": '''
print("anagram_silent", check_anagrams("Silent", "Listen"))
print("anagram_there", check_anagrams("There", "Their"))
''',
}


def fix_314(src):
    """ref.py's two syntax-only changes, so Python 3.12 compiles code written for 3.14."""
    src = re.sub(r"except ([A-Za-z]+(?:, *[A-Za-z]+)+):", r"except (\1):", src)
    return src


def extract(src, names):
    """The listed top-level defs/classes, the same-file defs they call (transitively), and the
    file's imports those use."""
    tree = ast.parse(src)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    want = list(names)
    i = 0
    while i < len(want):
        node = top.get(want[i])
        i += 1
        if node is None:
            continue
        for m in ast.walk(node):
            if isinstance(m, ast.Name) and m.id in top and m.id not in want and isinstance(top[m.id], ast.FunctionDef):
                want.append(m.id)
    used = set()
    for w in want:
        if w in top:
            used |= {m.id for m in ast.walk(top[w]) if isinstance(m, ast.Name)}
    lines = src.splitlines()
    parts = ["from __future__ import annotations"]
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
            bound = {(a.asname or a.name).split(".")[0] for a in n.names}
            if bound & used:
                parts.append("\n".join(lines[n.lineno - 1:n.end_lineno]))
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want:
            start = (n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1
            parts.append("\n".join(lines[start:n.end_lineno]))
    return "\n\n\n".join(parts) + "\n", want


def port_lines(part):
    p = os.path.join(ROOT, "study", "idioms", "thealgorithms-python", "out", part + ".values")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            if ": " in line:
                k, v = line.rstrip("\n").split(": ", 1)
                out.setdefault(k, v)
    return out


def g(x):
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    return "%g" % x


def run_thealgorithms(out_dir, python312):
    src_dir = ta_src()
    if not src_dir:
        print("TheAlgorithms clone not found (TA_SRC)")
        return []
    results = []
    unit_dir = os.path.join(out_dir, "thealgorithms")
    os.makedirs(unit_dir, exist_ok=True)
    for part, files in PORTED.items():
        ports = port_lines(part)
        for rel, names in files.items():
            src = fix_314(open(os.path.join(src_dir, rel)).read())
            for name in names:
                drv = DRIVERS.get(name)
                called = []
                if drv:
                    called = [m.func.id for m in ast.walk(ast.parse(drv)) if isinstance(m, ast.Call) and isinstance(m.func, ast.Name)]
                unit_src, deps = extract(src, [name] + [c for c in called if c != name])
                prog = unit_src + ("\n\n" + drv.strip() + "\n" if drv else "")
                uname = "%s__%s" % (rel.replace("/", "_")[:-3], name)
                path = os.path.join(unit_dir, uname + ".py")
                with open(path, "w") as f:
                    f.write(prog)
                row = {"unit": "%s::%s" % (rel, name), "part": part, "driver": bool(drv)}
                # translate under 3.12 (the source uses PEP 695 syntax)
                cmd = [python312, os.path.join(HERE, "py2viv.py"), path, "-o", os.devnull, "--manifest", path + ".json"]
                subprocess.run(cmd, capture_output=True, text=True)
                man = json.load(open(path + ".json"))
                row["strict"] = {"translates": not man["refusals"], "refusals": man["refusals"], "inexact": man.get("inexact", [])}
                if row["strict"]["translates"] and drv:
                    d = os.path.join(unit_dir, uname + ".py2viv")
                    cmd = [python312, os.path.join(HERE, "diff.py"), path, "--out", d, "--json"]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    try:
                        rep = json.loads(p.stdout)
                    except ValueError:
                        rep = {"verdict": "RUNNER FAILED: " + p.stderr[-300:]}
                    row["diff"] = {"verdict": rep["verdict"]}
                    # the hand-written port's line for each printed label
                    interp = os.path.join(d, "interp.out")
                    ptests = os.path.join(d, "python_tests.json")
                    if os.path.exists(interp) and os.path.exists(ptests):
                        man = json.load(open(os.path.join(d, uname + ".py2viv.json")))
                        pres = json.load(open(ptests))
                        lines = diff.split_viv(open(interp).read().split("\n--- stderr ---\n")[0], "")[0]
                        events = diff.read_calls(lines, pres.get("prints", []), pres.get("print_sites", []), man["sites"])[0]
                        cmp_rows = []
                        for ev in events:
                            site = next(s for s in man["sites"] if s["site"] == ev["site"])
                            label = site["parts"][0]["text"] if site["parts"][0]["kind"] == "str" else None
                            if label is None or label not in ports or not ev["ok"]:
                                continue
                            vals = ev["vals"].get(0, {}).get("v", [])
                            is_arr = site["parts"][2]["type"]["kind"] == "arr" if len(site["parts"]) > 2 else False
                            ours = ("[" + ", ".join(g(v) for v in vals) + "]") if is_arr else g(vals[0])
                            cmp_rows.append({"label": label, "translated": ours, "port": ports[label], "same": ours == ports[label]})
                        row["vs_port"] = cmp_rows
                results.append(row)
                vp = row.get("vs_port")
                print("%-70s translates %-3s %s %s" % (row["unit"], "yes" if row["strict"]["translates"] else "no",
                                                       row.get("diff", {}).get("verdict", "(no driver)" if row["strict"]["translates"] and not drv else ""),
                                                                ("port: %d/%d lines identical" % (sum(r["same"] for r in vp), len(vp))) if vp else ""), flush=True)
    return results


# ---------------------------------------------------------------- report


def summarize(name, results):
    o = ["", "== %s: %d units" % (name, len(results))]
    s = sum(r["strict"]["translates"] for r in results)
    ran = [r for r in results if "diff" in r]
    passed = [r for r in ran if r["diff"]["verdict"].startswith("PASS")]
    bitwise = [r for r in passed if "bitwise" in r["diff"]["verdict"]]
    o.append("translate: %d/%d (one mode: every translation is exact or refused); differential PASS: %d of %d run (%d bit for bit)"
             % (s, len(results), len(passed), len(ran), len(bitwise)))
    for r in ran:
        if not r["diff"]["verdict"].startswith("PASS"):
            o.append("  %-60s %s" % (r["unit"], r["diff"]["verdict"]))
    for mode in ("strict",):
        rows, casc = rank([r[mode] for r in results])
        o.append("")
        o.append("refusal reasons, %s (files affected, occurrences, gap):" % mode)
        for c, f, n, gap in rows:
            o.append("  %3d %4d  %-6s %s" % (f, n, gap or "", c))
        if casc:
            o.append("  cascades (unknown types from refusals above): %s" % ", ".join("%s x%d" % (k, v) for k, v in casc.items()))
    return "\n".join(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["posts", "thealgorithms", "all"])
    ap.add_argument("--out", default=os.path.join(HERE, "corpus_out"))
    ap.add_argument("--py312", default=os.environ.get("PY312", "/opt/homebrew/bin/python3.12"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    report = {}
    text = []
    if args.which in ("posts", "all"):
        report["posts"] = run_posts(args.out)
        text.append(summarize("posts/refs NumPy programs (no SciPy, no torch)", report["posts"]))
    if args.which in ("thealgorithms", "all"):
        report["thealgorithms"] = run_thealgorithms(args.out, args.py312)
        text.append(summarize("TheAlgorithms functions ported in study/idioms", report["thealgorithms"]))
    with open(os.path.join(args.out, "corpus.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(args.out, "corpus.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

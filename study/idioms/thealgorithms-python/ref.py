# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Run the original TheAlgorithms/Python code on the inputs the Viv ports use, and print
the same lines the ports print, in Viv's `show` format, to ref/<name>.out.

The repository is a shallow clone at ../../../../src/thealgorithms-python (see NOTES.md).
Its code targets Python 3.14 (`except A, B:` without parentheses, annotations evaluated
lazily); this machine has 3.12, so `load` adds the parentheses and turns on
`from __future__ import annotations` before compiling. Nothing else in the source is changed,
and module-level prints (dijkstra.py, n_queens.py, all_permutations.py) are captured.

    /opt/homebrew/bin/python3.12 ref.py
"""

import contextlib
import io
import os
import random
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../../../../src/thealgorithms-python"))
OUT = os.path.join(HERE, "ref")
os.makedirs(OUT, exist_ok=True)


def load(rel):
    """Import one file of the repository as a module; returns (module, what it printed)."""
    path = os.path.join(REPO, rel)
    src = open(path).read()
    src = re.sub(r"except ([A-Za-z]+(?:, *[A-Za-z]+)+):", r"except (\1):", src)
    if "from __future__ import annotations" not in src:
        src = "from __future__ import annotations\n" + src  # 3.14 evaluates annotations lazily
    mod = types.ModuleType(rel.replace("/", ".")[:-3])
    mod.__file__ = path
    sys.modules[mod.__name__] = mod
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, path, "exec"), mod.__dict__)
    return mod, buf.getvalue()


# ---- Viv's `show` format: numbers as C's %g, booleans as true/false, matrices one row a line
def num(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    return "%g" % x


def vec(xs):
    return "[" + ", ".join(num(x) for x in xs) + "]"


def fmt(name, v):
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        pad = " " * (len(name) + 3)
        return name + ": [" + (",\n" + pad).join(vec(r) for r in v) + "]"
    if isinstance(v, (list, tuple)):
        return name + ": " + vec(v)
    return name + ": " + num(v)


class Out:
    def __init__(self, name):
        self.f = open(os.path.join(OUT, name + ".out"), "w")

    def show(self, name, v):
        self.f.write(fmt(name, v) + "\n")

    def text(self, s):
        self.f.write("text: " + s + "\n")


def codes(s):
    return [ord(c) for c in s]


# ------------------------------------------------------------------ sorting
def sorting():
    o = Out("sorting")
    merge, _ = load("sorts/merge_sort.py")
    quick, _ = load("sorts/quick_sort.py")
    heap, _ = load("sorts/heap_sort.py")
    ins, _ = load("sorts/insertion_sort.py")
    bs, _ = load("searches/binary_search.py")
    data = [0, 5, 3, 2, 2, -45, -2, -5, 103, 9, 1, 7, 11]
    random.seed(7)
    o.show("by_merge", merge.merge_sort(list(data)))
    o.show("by_quick", quick.quick_sort(list(data)))
    o.show("by_heap", heap.heap_sort(list(data)))
    o.show("by_insertion", ins.insertion_sort(list(data)))
    hay = [0, 5, 7, 10, 10, 10, 15]
    queries = [0, 5, 7, 10, 15, 6, -1, 16]
    o.show("found", [bs.binary_search(hay, q) for q in queries])
    o.show("lefts", [bs.bisect_left(hay, q) for q in queries])
    o.show("recursive", [bs.binary_search_by_recursion(hay, q, 0, len(hay) - 1) for q in queries])


# ------------------------------------------------------------------ graphs
def graphs():
    o = Out("graphs")
    bfs_mod, _ = load("graphs/breadth_first_search.py")
    g = bfs_mod.Graph()
    for a, b in [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 3)]:
        g.add_edge(a, b)
    o.show("bfs_from_2", sorted(g.bfs(2)))
    o.show("bfs_from_1", sorted(g.bfs(1)))
    dfs_mod, _ = load("graphs/depth_first_search.py")
    a = sorted(dfs_mod.depth_first_search(dfs_mod.G, "A"))
    gg = sorted(dfs_mod.depth_first_search(dfs_mod.G, "G"))
    o.show("dfs_from_A", [ord(c) - 65 for c in a])
    o.show("dfs_from_G", [ord(c) - 65 for c in gg])
    o.text("".join(a))
    dj, _ = load("graphs/dijkstra.py")
    o.show("dijkstra_G_E_C", dj.dijkstra(dj.G, "E", "C"))
    o.show("dijkstra_G2_E_F", dj.dijkstra(dj.G2, "E", "F"))
    o.show("dijkstra_G3_E_F", dj.dijkstra(dj.G3, "E", "F"))
    kahn, _ = load("graphs/kahns_algorithm_topo.py")
    none_is_empty = lambda r: [] if r is None else r
    o.show("topo_dag", none_is_empty(kahn.topological_sort({0: [1, 2], 1: [3], 2: [3], 3: [4, 5], 4: [], 5: []})))
    o.show("topo_cycle", none_is_empty(kahn.topological_sort({0: [1], 1: [2], 2: [0]})))
    o.show("topo_sparse", none_is_empty(kahn.topological_sort({10: [20], 20: []})))
    o.show("topo_sparse_cycle", none_is_empty(kahn.topological_sort({10: [20], 20: [10]})))
    kr, _ = load("graphs/minimum_spanning_tree_kruskal.py")
    o.show("mst1", kr.kruskal(4, [(0, 1, 3), (1, 2, 5), (2, 3, 1)]))
    o.show("mst2", kr.kruskal(4, [(0, 1, 3), (1, 2, 5), (2, 3, 1), (0, 2, 1), (0, 3, 2)]))
    o.show("mst3", kr.kruskal(4, [(0, 1, 3), (1, 2, 5), (2, 3, 1), (0, 2, 1), (0, 3, 2), (2, 1, 1)]))
    ds, _ = load("data_structures/disjoint_set/disjoint_set.py")
    ds.test_disjoint_set()


# ------------------------------------------------------------------ dp
def dp():
    o = Out("dp")
    ks, _ = load("dynamic_programming/knapsack.py")
    val, wt = [3, 2, 4, 4], [4, 3, 2, 3]
    best, _table = ks.knapsack(6, wt, val, 4)
    ks.f = [[0] * 7] + [[0] + [-1] * 7 for _ in range(5)]
    o.show("knapsack_value", best)
    o.show("mf_knapsack_value", ks.mf_knapsack(4, wt, val, 6))
    v6, s6 = ks.knapsack_with_example_solution(6, wt, val)
    v10, s10 = ks.knapsack_with_example_solution(10, [1, 3, 5, 2], [10, 20, 100, 22])
    o.show("knapsack_items_6", sorted(s6))
    o.show("knapsack_value_10", v10)
    o.show("knapsack_items_10", sorted(s10))
    lcs, _ = load("dynamic_programming/longest_common_subsequence.py")
    n, s = lcs.longest_common_subsequence("AGGTAB", "GXTXAYB")
    o.show("lcs_len", n)
    o.text(s)
    for x, y in [("programming", "gaming"), ("physics", "smartphone"), ("ABCD", "ACBD")]:
        o.text(lcs.longest_common_subsequence(x, y)[1])
    ed, _ = load("dynamic_programming/edit_distance.py")
    o.show("edit_top_down", ed.EditDistance().min_dist_top_down("intention", "execution"))
    o.show("edit_bottom_up", ed.EditDistance().min_dist_bottom_up("intention", "execution"))
    o.show("edit_kitten", ed.EditDistance().min_dist_bottom_up("kitten", "sitting"))


# ------------------------------------------------------------------ strings
def strings():
    o = Out("strings")
    k, _ = load("strings/knuth_morris_pratt.py")
    kmp = k.knuth_morris_pratt
    o.show("kmp_1", kmp("alskfjaldsabc1abc1abc12k23adsfabcabc", "abc1abc12"))
    o.show("kmp_2", kmp("alskfjaldsk23adsfabcabc", "abc1abc12"))
    o.show("kmp_3", kmp("ABABZABABYABABX", "ABABX"))
    o.show("kmp_4", kmp("ABAAAAAB", "AAAB"))
    o.show("kmp_5", kmp("abcxabcdabxabcdabcdabcy", "abcdabcy"))
    o.show("kmp_finds", [kmp("knuth_morris_pratt", s) for s in ("kn", "h_m", "rr", "tt", "not there")])
    o.show("failure", k.get_failure_array("aabaabaaa"))
    p, _ = load("strings/palindrome.py")
    rows = []
    for key in p.test_data:
        rows.append([p.is_palindrome(key), p.is_palindrome_traversal(key),
                     p.is_palindrome_recursive(key), p.is_palindrome_slice(key)])
    o.show("palindromes", [[int(b) for b in r] for r in rows])
    o.show("loose_palindromes", [int(p.is_palindrome_ignore_case_and_spaces(s)) for s in p.test_data_ignore_case_and_spaces])
    a, _ = load("strings/check_anagrams.py")
    o.show("anagram_silent", a.check_anagrams("Silent", "Listen"))
    o.show("anagram_string", a.check_anagrams("This is a string", "Is this a string"))
    o.show("anagram_spaces", a.check_anagrams("This is    a      string", "Is     this a string"))
    o.show("anagram_there", a.check_anagrams("There", "Their"))


# ------------------------------------------------------------------ maths
def maths():
    o = Out("maths")
    sv, _ = load("maths/prime_sieve_eratosthenes.py")
    o.show("primes_10", sv.prime_sieve_eratosthenes(10))
    o.show("primes_100", sv.prime_sieve_eratosthenes(100))
    o.show("primes_1", sv.prime_sieve_eratosthenes(1))
    g, _ = load("maths/greatest_common_divisor.py")
    pairs = [(24, 40), (1, 1), (1, 800), (11, 37), (3, 5), (16, 4), (-3, 9), (9, -3), (3, -9), (-3, -9), (0, 0), (1, -800)]
    o.show("gcd_recursive", [g.greatest_common_divisor(a, b) for a, b in pairs])
    o.show("gcd_iterative", [g.gcd_by_iterative(a, b) for a, b in pairs])
    e, _ = load("maths/binary_exponentiation.py")
    cases = [(3, 5), (11, 13), (-1, 3), (0, 5), (3, 1), (3, 0), (1.5, 4)]
    o.show("exp_recursive", [e.binary_exp_recursive(a, b) for a, b in cases])
    o.show("exp_iterative", [e.binary_exp_iterative(a, b) for a, b in cases])
    mcases = [(3, 4, 5), (11, 13, 7), (1.5, 4, 3), (7, 13, 10)]
    o.show("exp_mod_recursive", [e.binary_exp_mod_recursive(*c) for c in mcases])
    o.show("exp_mod_iterative", [e.binary_exp_mod_iterative(*c) for c in mcases])
    a, b, c = 1269380576, 374, 34
    o.show("bench_mod_recursive", e.binary_exp_mod_recursive(a, b, c))
    o.show("bench_mod_iterative", e.binary_exp_mod_iterative(a, b, c))
    big = e.binary_exp_iterative(a, b)
    # %g of a 3,400-digit integer overflows a float; say what %g would print for it
    o.show("bench_iterative", float("inf") if big.bit_length() > 1024 else big)
    print("binary_exp_iterative(1269380576, 374) has", len(str(big)), "digits in Python", file=sys.stderr)


# ------------------------------------------------------------------ structures
def structures():
    o = Out("structures")
    hp, _ = load("data_structures/heap/heap.py")
    for unsorted in [[0], [2], [3, 5], [5, 3], [5, 5], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 3, 5], [0, 2, 2, 3, 5], [2, 5, 3, 0, 2, 3, 0, 3], [6, 1, 2, 7, 9, 3, 4, 5, 10, 8], [103, 9, 1, 7, 11, 15, 25, 201, 209, 107, 5], [-45, -2, -5]]:
        h = hp.Heap()
        h.build_max_heap(unsorted)
        o.show("after_build", list(h.h))
        o.show("max_value", h.extract_max())
        o.show("after_extract", list(h.h))
        h.insert(100)
        o.show("after_insert", list(h.h))
        h.heap_sort()
        o.show("heap_sorted", list(h.h))
    qm, _ = load("data_structures/queues/queue_by_list.py")
    q = qm.QueueByList([10, 20, 30, 40])
    q.rotate(1)
    o.show("rotated_1", q.entries)
    q.rotate(2)
    o.show("rotated_3", q.entries)
    lm, _ = load("data_structures/linked_list/singly_linked_list.py")
    ll = lm.LinkedList()
    for i in range(10):
        ll.insert_nth(i, i + 1)
    ll.insert_head(0)
    ll.insert_tail(11)
    o.show("linked_0_to_11", list(ll))
    o.show("deleted", [ll.delete_head(), ll.delete_nth(9), ll.delete_tail()])
    o.show("linked_1_to_9", list(ll))
    for i in range(9):
        ll[i] = -i
    ll.reverse()
    o.show("linked_reversed", list(ll))
    lm.test_singly_linked_list()
    hmm, _ = load("data_structures/hashing/hash_map.py")
    hm = hmm.HashMap(8, 0.75)
    for i in range(12):
        hm[i] = 10 * i
    o.show("hm_buckets_after_insert", len(hm._buckets))
    o.show("hm_keys_after_insert", list(hm))
    for i in range(6):
        del hm[2 * i]
    o.show("hm_buckets_after_delete", len(hm._buckets))
    o.show("hm_keys_after_delete", list(hm))
    o.show("hm_values_after_delete", list(hm.values()))
    neg = hmm.HashMap(5, 0.75)
    neg[-5] = 10
    neg[6] = 30
    neg[-7] = 20
    neg[-1] = 40
    o.show("hm_neg_keys", list(neg))
    st, _ = load("data_structures/stacks/stack.py")
    st.test_stack()


# ------------------------------------------------------------------ backtracking
def backtracking():
    o = Out("backtracking")
    nq, printed = load("backtracking/n_queens.py")
    first_board = printed.split("\n\n")[0].strip().split("\n")
    o.show("first_8_queens", [[1 if c == "Q" else 0 for c in row.split()] for row in first_board])
    counts = []
    for n in range(1, 9):
        nq.solution.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            nq.solve([[0] * n for _ in range(n)], 0)
        counts.append(len(nq.solution))
    o.show("queens_counts", counts)
    ap, printed = load("backtracking/all_permutations.py")
    for line in printed.strip().split("\n"):
        items = eval(line)
        if isinstance(items[0], str):
            o.text("".join(items))
        else:
            o.show("current", items)


if __name__ == "__main__":
    for part in [sorting, graphs, dp, strings, maths, structures, backtracking]:
        part()
        print("wrote ref/%s.out" % part.__name__)

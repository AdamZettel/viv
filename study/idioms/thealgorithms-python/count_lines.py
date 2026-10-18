# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Code lines of the ported functions: the reference's (by AST range) and the Viv defs.
A code line is non-blank, not a comment, and not part of a docstring. Tests, __main__
blocks and the runs at the bottom of each .viv are not counted.

    /opt/homebrew/bin/python3.12 count_lines.py
"""

import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../../../../src/thealgorithms-python"))

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

# Python methods the Viv port leaves out on purpose (printing, dunder conveniences with
# no Viv counterpart), so both sides count the same algorithm.
SKIP_METHODS = {"__repr__", "__str__", "print_graph", "__iter__", "_step_by_step", "__bool__", "__contains__"}


def py_lines(path, names):
    src = open(path).read()
    src = re.sub(r"except ([A-Za-z]+(?:, *[A-Za-z]+)+):", r"except (\1):", src)
    tree = ast.parse(src)
    lines = src.splitlines()
    doc = set()
    keep = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc.update(range(first.lineno, first.end_lineno + 1))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.update(range(node.lineno, node.end_lineno + 1))
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef) and m.name in SKIP_METHODS:
                        keep.difference_update(range(m.lineno - len(m.decorator_list), m.end_lineno + 1))
    n = 0
    for i in keep - doc:
        s = lines[i - 1].strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def viv_lines(path):
    """Lines inside top-level defs, minus comments, blanks and docstrings."""
    n = 0
    depth = 0
    in_def = False
    for raw in open(path):
        s = raw.strip()
        if not in_def and s.startswith("def "):
            in_def = True
        if in_def:
            if s and not s.startswith("#") and not (s.startswith('"') and s.endswith('"')):
                n += 1
            depth += s.count("{") - s.count("}")
            if depth == 0:
                in_def = False
    return n


if __name__ == "__main__":
    total_py = total_viv = 0
    print("%-13s %6s %6s" % ("port", "python", "viv"))
    for part, files in PORTED.items():
        p = sum(py_lines(os.path.join(REPO, f), names) for f, names in files.items())
        v = viv_lines(os.path.join(HERE, part + ".viv"))
        total_py += p
        total_viv += v
        print("%-13s %6d %6d" % (part, p, v))
    s = viv_lines(os.path.join(HERE, "seq.viv"))
    print("%-13s %6s %6d" % ("seq (shared)", "-", s))
    print("%-13s %6d %6d" % ("total", total_py, total_viv + s))

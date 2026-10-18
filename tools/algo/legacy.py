#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""legacy.py: copy the corpus's hand-written algorithms into lib/algo/legacy, for testing.

    python3 tools/algo/legacy.py           write lib/algo/legacy/<module>.viv
    python3 tools/algo/legacy.py --check   exit 1 if a committed copy differs from what
                                           this writes now (the corpus changed, or a copy
                                           was edited by hand)

Each copy is the corpus defs, text for text, with two kinds of change and no others:

  1. every comparison of two ELEMENTS (not of two indices) is routed through a def of
     lib/algo/legacy/order.viv: `a < b` becomes `lg_lt(a, b)`, and `<=`, `>`, `==`,
     `!=` become lg_le, lg_gt, lg_eq, lg_ne. Each def is its operator, so the copy
     computes what the original computes; being a def, each call is counted by the
     interpreter's call graph, and McIlroy's adversary can answer it (order.viv).
  2. a def may be renamed (seq.viv's `argsort` shadows the builtin; the copy is
     `seq_argsort`), and the calls of a renamed def follow it.

Every routing below must match exactly once in its def, so a corpus edit that changes a
routed line stops this script instead of silently dropping the routing. The originals
are not touched (posts and study programs are the owner's).
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imports"))
from viv_imports import migrated  # noqa: E402  (tools/imports/viv_imports.py)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "lib", "algo", "legacy")

TA = "study/idioms/thealgorithms-python"

# module -> (what it is, [(source file, [def or (def, new name)], [globals])], routings)
# a routing is (def, original text, routed text)
MODULES = {
    "ta_sorting": (
        "TheAlgorithms/Python's sorts and binary searches, as ported in the study corpus.",
        [(f"{TA}/sorting.viv", ["merge", "merge_sort", "partition", "sign", "quick_sort", "heapify", "heap_sort",
                                 "insertion_sort", "binary_search", "bisect_left", "search_between",
                                 "binary_search_by_recursion"], [])],
        [
            ("merge", "right[j] < left[i]", "lg_lt(right[j], left[i])"),
            ("heapify", "a[largest] < a[left]", "lg_lt(a[largest], a[left])"),
            ("heapify", "a[largest] < a[right]", "lg_lt(a[largest], a[right])"),
            ("insertion_sort", "insert_value < a[j - 1.0]", "lg_lt(insert_value, a[j - 1.0])"),
            ("binary_search", "sorted[mid] == item", "lg_eq(sorted[mid], item)"),
            ("binary_search", "item < sorted[mid]", "lg_lt(item, sorted[mid])"),
            ("bisect_left", "sorted[mid] < item", "lg_lt(sorted[mid], item)"),
            ("search_between", "sorted[mid] == item", "lg_eq(sorted[mid], item)"),
            ("search_between", "item < sorted[mid]", "lg_lt(item, sorted[mid])"),
        ],
    ),
    "seq_argsort": (
        "The merge-sort argsort every TheAlgorithms port uses (seq.viv); it shadows the builtin there.",
        [(f"{TA}/seq.viv", [("argsort", "seq_argsort")], [])],
        [("seq_argsort", "keys[idx[i]] <= keys[idx[j]]", "lg_le(keys[idx[i]], keys[idx[j]])")],
    ),
    "cmp_merge_sort": (
        "The /compare page's recursive merge sort on lists.",
        [("study/compare/recursion/merge_sort.viv", ["merge", "merge_sort"], [])],
        [("merge", "a[i] <= b[j]", "lg_le(a[i], b[j])")],
    ),
    "cmp_quicksort": (
        "The /compare page's quicksort: Lomuto's partition around the last element.",
        [("study/compare/recursion/quicksort.viv", ["partition", "quicksort"], [])],
        [("partition", "a[j] < pivot", "lg_lt(a[j], pivot)")],
    ),
    "cmp_insertion_sort": (
        "The /compare page's insertion sort on a copy.",
        [("study/compare/lists/sorting.viv", ["insertion_sort"], [])],
        [("insertion_sort", "a[j - 1] > v", "lg_gt(a[j - 1], v)")],
    ),
    "bench_qsort": (
        "The Julia microbenchmark's quicksort (Hoare-style, middle pivot) and issorted.",
        [("study/benchmarks/viv/micro/quicksort.viv", ["qsort", "issorted"], [])],
        [
            ("qsort", "a[i] < pivot", "lg_lt(a[i], pivot)"),
            ("qsort", "a[j] > pivot", "lg_gt(a[j], pivot)"),
            ("issorted", "a[k - 1] > a[k]", "lg_gt(a[k - 1], a[k])"),
        ],
    ),
    "eig_sort": (
        "lib/hf's eigenvalue sort: a selection sort that moves each eigenvector with its value.",
        [("lib/hf/eigen.viv", ["eig_sort"], [])],
        [("eig_sort", "d[j] < p", "lg_lt(d[j], p)")],
    ),
    "searches": (
        "Two hand-written binary searches: lib/nprandom's searchsorted(side='right') and the "
        "fasta benchmark's bisect.bisect.",
        [("lib/nprandom/legacy.viv", ["npr_lg_search"], []), ("study/benchmarks/viv/bg/fasta.viv", ["bisect"], [])],
        [("npr_lg_search", "cdf[mid] <= u", "lg_le(cdf[mid], u)"), ("bisect", "x < a[mid]", "lg_lt(x, a[mid])")],
    ),
    "heaps": (
        "The two heaps of the TheAlgorithms ports: heap.py's max-heap of numbers, and the "
        "(cost, vertex) min-heap Dijkstra uses (heapq.heappush and heappop).",
        [(f"{TA}/structures.viv", ["heap_new", "heap_swap", "max_heapify", "build_max_heap", "extract_max",
                                    "heap_insert", "heap_sort_in_place", "heap_items"], ["H_SIZE", "H_ITEMS"]),
         (f"{TA}/graphs.viv", ["less", "swap_rows", "heappush", "heappop"], [])],
        [
            ("max_heapify", "h[H_ITEMS + left] > h[H_ITEMS + violation]", "lg_gt(h[H_ITEMS + left], h[H_ITEMS + violation])"),
            ("max_heapify", "h[H_ITEMS + right] > h[H_ITEMS + violation]", "lg_gt(h[H_ITEMS + right], h[H_ITEMS + violation])"),
            ("less", "h[i, 0] != h[j, 0]", "lg_ne(h[i, 0], h[j, 0])"),
            ("less", "h[i, 0] < h[j, 0]", "lg_lt(h[i, 0], h[j, 0])"),
            ("less", "h[i, 1] < h[j, 1]", "lg_lt(h[i, 1], h[j, 1])"),
        ],
    ),
    "gcd_post": (
        "The gcd of posts/2026-09-25-the-error-follows-the-condition-number.viv.",
        [("posts/2026-09-25-the-error-follows-the-condition-number.viv", ["gcd"], [])],
        [],
    ),
    "gcd_julia": (
        "The gcd of the Julia idioms port (numtheory.viv).",
        [("study/idioms/julia-idioms/numtheory.viv", ["gcd"], [])],
        [],
    ),
    "gcd_ta": (
        "TheAlgorithms' two gcds (maths.viv).",
        [(f"{TA}/maths.viv", ["greatest_common_divisor", "gcd_by_iterative"], [])],
        [],
    ),
    "permutations": (
        "TheAlgorithms' all_permutations.py backtracking (backtracking.viv).",
        [(f"{TA}/backtracking.viv", ["create_state_space_tree", "generate_all_permutations"], [])],
        [],
    ),
    "topo": (
        "TheAlgorithms' Kahn topological sort (graphs.viv), with the helpers it calls.",
        [(f"{TA}/graphs.viv", ["row_offsets", "by_source", "index_of", "topological_sort"], [])],
        [("index_of", "keys[i] == k", "lg_eq(keys[i], k)")],
    ),
}

# a module whose defs call a def renamed in another copy uses that copy
USES = {"topo": ["seq_argsort"]}
RENAMED_CALLS = {"topo": {"argsort": "seq_argsort"}}


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def def_text(src, name, rel):
    """The def `name` of `src`: its `#|` lines, then `def name(` to the `}` in column 0."""
    lines = src.split("\n")
    for k, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = k
            while start > 0 and lines[start - 1].startswith("#|"):
                start -= 1
            end = k
            while lines[end] != "}":
                end += 1
            return "\n".join(lines[start:end + 1])
    raise SystemExit(f"legacy.py: {rel} has no `def {name}(`")


def global_text(src, name, rel):
    for line in src.split("\n"):
        if re.match(rf"{name} = ", line):
            return line
    raise SystemExit(f"legacy.py: {rel} has no global `{name}`")


def module_text(mod):
    what, sources, routings = MODULES[mod]
    out = ["# Generated by tools/algo/legacy.py from the corpus: do not edit.",
           f"# {what}",
           "# Element comparisons are routed through lib/algo/legacy/order.viv; nothing else changes.",
           "#! use lib/algo/legacy/order"]
    for u in USES.get(mod, []):
        out.append(f"#! use lib/algo/legacy/{u}")
    for rel, defs, globs in sources:
        src = read(rel)
        names = [d if isinstance(d, str) else d[0] for d in defs]
        out.append("")
        out.append(f"# from {rel}: {' '.join(names)}")
        for g in globs:
            out.append(global_text(src, g, rel))
        for d in defs:
            old, new = (d, d) if isinstance(d, str) else d
            text = def_text(src, old, rel)
            if new != old:
                text = text.replace(f"def {old}(", f"def {new}(", 1)
            for callee, renamed in RENAMED_CALLS.get(mod, {}).items():
                text = re.sub(rf"(?<![A-Za-z_0-9]){callee}\(", f"{renamed}(", text)
            for dname, a, b in routings:
                if dname != new:
                    continue
                count = text.count(a)
                if count != 1:
                    raise SystemExit(f"legacy.py: `{a}` appears {count} times in {rel}'s {old}; expected once")
                text = text.replace(a, b)
            out.append("")
            out.append(text)
    return "\n".join(out) + "\n"


def main():
    bad = 0
    for mod in MODULES:
        path = os.path.join(OUT, mod + ".viv")
        text = migrated(module_text(mod), path)
        if "--check" in sys.argv:
            have = open(path).read() if os.path.exists(path) else ""
            if have != text:
                print(f"lib/algo/legacy/{mod}.viv differs from its corpus source; run tools/algo/legacy.py")
                bad = 1
        else:
            with open(path, "w") as f:
                f.write(text)
    if "--check" not in sys.argv:
        print(f"{len(MODULES)} legacy modules written to lib/algo/legacy")
    sys.exit(bad)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""check.py: lib/algo's specifications, checked.

    python3 tools/algo/check.py              the static checks (standard library only, so CI
                                             runs them anywhere; see below)
    python3 tools/algo/check.py --run        everything: write the test programs, run them,
                                             check every postcondition in Python as well,
                                             compare with Python and NumPy, measure the
                                             comparison counts, run the three engines, and
                                             write the ledger and the report
    python3 tools/algo/check.py --run --only ta_   only the impls whose id contains ta_
                                             (no ledger written)
    python3 tools/algo/check.py --run --no-engines   skip tools/three-engines.sh
    python3 tools/algo/check.py --write      write the test programs only

The static checks:
  1. lib/algo/algo.spec parses; every algo has a tier, a signature, and (tiers 1 and 2) a
     bound and its postconditions; every post, note and oracle names a property or an
     oracle this file knows; every impl names an algo, a module and its call;
  2. the legacy copies are what tools/algo/legacy.py writes from the corpus now;
  3. the generated test programs (lib/algo/tests, lib/algo/legacy/tests) are what this
     file writes from the spec now;
  4. the ledger (tools/algo/results/ledger.json) was written from these files (a
     hash), and records no unexpected result.

What --run does, per impl of the spec:
  properties   the impl's program runs every (kind, size) case in the interpreter. For each
               case it prints the input, the results, and the verdict of each property as
               lib/algo/props.viv computes it. This file computes each property again from
               the printed values; the two verdicts must agree (a disagreement is a bug in
               props.viv or here). A property that fails on a kind inside the impl's domain
               fails the run unless the spec records it as `known`.
  oracles      each `oracle` line: the printed value must equal the Python or NumPy result
               byte for byte (repr of every double: -0.0 and nan included).
  complexity   for an impl with `count`, the comparisons are the calls of the named defs of
               lib/algo/legacy/order.viv, read from the interpreter's call graph (`__graph__`).
               Runs at n = 64 .. 2048 on each kind, and on McIlroy's adversary when the impl
               says `adversary yes`; the counts are fit and held to the impl's `growth` (or
               the algo's `bound`).
  self-test    every tier-1 postcondition is checked on a Python reference (it must hold)
               and on mutants of it (each must be caught), so a property that cannot fail is
               found before any Viv implementation exists.
  engines      tools/three-engines.sh on lib/algo, lib/algo/legacy, and both tests dirs.
"""

import bisect
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "lib", "algo", "algo.spec")
RESULTS = os.path.join(ROOT, "tools", "algo", "results")
LEDGER = os.path.join(RESULTS, "ledger.json")
REPORT = os.path.join(RESULTS, "report.txt")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imports"))
from viv_imports import migrated  # noqa: E402  (tools/imports/viv_imports.py)

VIV = os.environ.get("VIV", os.path.join(ROOT, "target", "release", "viv"))

KINDS = ["random", "dups", "sorted", "reversed", "equal", "organ", "sawtooth", "nearly", "zeros", "special"]
SIZES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257]
COMPLEXITY_SIZES = [64, 128, 256, 512, 1024, 2048]
COMPLEXITY_KINDS = ["random", "dups", "sorted", "reversed", "equal", "organ"]


# ================================================================ the spec

class Block:
    def __init__(self, kind, ident, title, line):
        self.kind, self.id, self.title, self.line = kind, ident, title, line
        self.fields = []  # (key, value, line)

    def get(self, key, default=None):
        for k, v, _ in self.fields:
            if k == key:
                return v
        return default

    def all(self, key):
        return [v for k, v, _ in self.fields if k == key]


def parse_spec(path=SPEC):
    blocks, cur, errors = [], None, []
    with open(path) as f:
        lines = f.read().split("\n")
    for no, raw in enumerate(lines, 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        if not raw.startswith(" "):
            m = re.match(r"(algo|impl|order) (\S+)\s+(.*)$", raw)
            if not m:
                errors.append(f"algo.spec:{no}: a block starts `algo ID  title`, `impl ID  title` or `order ID  title`")
                cur = None
                continue
            cur = Block(m.group(1), m.group(2), m.group(3).strip(), no)
            blocks.append(cur)
            continue
        if cur is None:
            errors.append(f"algo.spec:{no}: a field outside a block")
            continue
        m = re.match(r"  (\S+)(?: {1,}(.*))?$", raw)
        if not m:
            errors.append(f"algo.spec:{no}: a field is `  key  value`")
            continue
        key, value = m.group(1), m.group(2) or ""
        if key == "call":
            # a call line keeps its own indentation after `call` and five spaces
            value = raw[len("  call     "):] if raw.startswith("  call     ") else value.strip()
        else:
            value = value.strip()
        cur.fields.append((key, value, no))
    return blocks, errors


# ================================================================ values printed by Viv

def parse_value(text):
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    safe = re.sub(r"\bnan\b", "NAN", re.sub(r"-inf\b", "-INF", re.sub(r"(?<![-\w])inf\b", "INF", text)))
    return eval(safe, {"__builtins__": {}}, {"NAN": math.nan, "INF": math.inf})  # noqa: S307 (our own output)


def same(x, y):
    """Bitwise equality of doubles, nan equal to nan."""
    if isinstance(x, list) or isinstance(y, list):
        return isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return x == y and math.copysign(1.0, x) == math.copysign(1.0, y)


def text(v):
    if isinstance(v, list):
        return "[" + ", ".join(text(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(float(v))


# ================================================================ the order and the properties

def before(x, y):
    """NumPy's order: ascending, -0.0 equal to 0.0, NaN after every number."""
    if math.isnan(x):
        return False
    if math.isnan(y):
        return True
    return x < y


def key_numpy(x):
    return (1, 0.0) if math.isnan(x) else (0, x)


def key_total(x):
    return (1, 0.0, 0) if math.isnan(x) else (0, x, 0 if math.copysign(1.0, x) < 0 else 1)


def sort_numpy(a):
    return sorted(a, key=key_numpy)


def bits(x):
    return "nan" if math.isnan(x) else repr(float(x))


def p_permutation(r, x):
    return sorted(bits(v) for v in r) == sorted(bits(v) for v in x)


def p_ordered(r):
    return all(not before(r[i], r[i - 1]) for i in range(1, len(r)))


def p_ordered_total(r):
    return all(key_total(r[i - 1]) <= key_total(r[i]) for i in range(1, len(r)))


def p_ordered_desc(r):
    return all(not before(r[i - 1], r[i]) for i in range(1, len(r)))


def zero_signs(a):
    return [math.copysign(1.0, v) for v in a if v == 0.0]


def p_stable_values(r, x):
    return zero_signs(r) == zero_signs(x)


def p_index_perm(p, x):
    return sorted(p) == [float(i) for i in range(len(x))]


def p_ordered_by(p, x):
    return all(not before(x[int(p[i])], x[int(p[i - 1])]) for i in range(1, len(p)))


def p_stable_by(p, x):
    for i in range(1, len(p)):
        a, b = x[int(p[i - 1])], x[int(p[i])]
        if not before(a, b) and not before(b, a) and p[i - 1] > p[i]:
            return False
    return True


def lower(v, x):
    return sum(1 for e in v if before(e, x))


def upper(v, x):
    return sum(1 for e in v if not before(x, e))


def p_lower_bounds(rs, v, q):
    return len(rs) == len(q) and all(rs[k] == lower(v, q[k]) and p_is_partition_at(v, q[k], rs[k], before) for k in range(len(q)))


def p_upper_bounds(rs, v, q):
    return len(rs) == len(q) and all(rs[k] == upper(v, q[k]) for k in range(len(q)))


def p_is_partition_at(v, x, r, rel):
    return all(rel(e, x) == (i < r) for i, e in enumerate(v))


def first_index(v, x):
    for i, e in enumerate(v):
        if e == x:
            return float(i)
    return -1.0


def p_first_or_none(rs, v, q):
    return len(rs) == len(q) and all(rs[k] == first_index(v, q[k]) for k in range(len(q)))


def p_contains(rs, v, q):
    return len(rs) == len(q) and all(bool(rs[k]) == any(not before(e, q[k]) and not before(q[k], e) for e in v) for k in range(len(q)))


def p_max_heap(h):
    return all(not h[(i - 1) // 2] < h[i] for i in range(1, len(h)))


def p_min_pos(r, x):
    if not x:
        return False
    best = 0
    for i in range(1, len(x)):
        if before(x[i], x[best]):
            best = i
    return r == best


def p_max_pos(r, x):
    if not x:
        return False
    best = 0
    for i in range(1, len(x)):
        if before(x[best], x[i]):
            best = i
    return r == best


def p_is_sorted_answer(f, x):
    return f == p_ordered(x)


def p_follows(d, p, x):
    return len(d) == len(x) and all(same(d[i], x[int(p[i])]) for i in range(len(d)))


def p_gcd(r, a, b):
    return len(r) == len(a) and all(r[i] == math.gcd(int(a[i]), int(b[i])) for i in range(len(r)))


def p_lcm(r, a, b):
    return all(r[i] == math.lcm(int(a[i]), int(b[i])) for i in range(len(r)))


def p_factorial_count(f, x):
    return f == math.factorial(len(x))


def p_topological(r, k, src, dst):
    if sorted(r) != sorted(k):
        return False
    pos = {v: i for i, v in enumerate(r)}
    return all(s in pos and d in pos and pos[s] < pos[d] for s, d in zip(src, dst))


def p_partial_sorted(r, x, k):
    k = int(k)
    return [bits(v) for v in r[:k]] == [bits(v) for v in sort_numpy(x)[:k]] or (
        all(not before(a, b) and not before(b, a) for a, b in zip(r[:k], sort_numpy(x)[:k])) and p_ordered(r[:k]))


def p_nth(r, x, k):
    k = int(k)
    if not x:
        return r == []
    s = sort_numpy(x)
    pivot = r[k]
    return (not before(pivot, s[k]) and not before(s[k], pivot)
            and all(not before(pivot, e) for e in r[:k]) and all(not before(e, pivot) for e in r[k + 1:]))


def pos_pred(v):
    return v > 0.0


def p_partitioned(r, k):
    k = int(k)
    return all(pos_pred(v) for v in r[:k]) and not any(pos_pred(v) for v in r[k:])


def p_stable_partition(r, x, k):
    want = [v for v in x if pos_pred(v)] + [v for v in x if not pos_pred(v)]
    return same(r, want) and k == sum(1 for v in x if pos_pred(v))


def p_partition_point(k, x):
    return k == sum(1 for v in x if pos_pred(v))


def p_reversed(r, x):
    return same(r, x[::-1])


def p_rotated(r, x, k):
    k = int(k)
    return same(r, x[k:] + x[:k])


def groups(a):
    """The sorted values of a, as (value, count) by NumPy's order (the first value of each tie)."""
    out = []
    for v in sort_numpy(a):
        if out and not before(out[-1][0], v) and not before(v, out[-1][0]):
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def p_unique_sorted(r, x):
    return [bits(v) for v in r] == [bits(v) for v, _ in groups(x)] or (
        len(r) == len(groups(x)) and all(not before(a, b) and not before(b, a) for a, (b, _) in zip(r, groups(x))))


def p_unique_runs(r, x):
    want = []
    for v in x:
        if not want or before(want[-1], v) or before(v, want[-1]):
            want.append(v)
    return same(r, want)


def stable_merge(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if before(b[j], a[i]):
            out.append(b[j])
            j += 1
        else:
            out.append(a[i])
            i += 1
    return out + a[i:] + b[j:]


def p_merged(r, a, b):
    return same(r, stable_merge(a, b))


def multiset_op(a, b, op):
    """STL set operations on sorted arrays with repeats; values from a where both have them."""
    out, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if j == len(b) or (i < len(a) and before(a[i], b[j])):
            if op in ("union", "difference", "symmetric"):
                out.append(a[i])
            i += 1
        elif i == len(a) or before(b[j], a[i]):
            if op in ("union", "symmetric"):
                out.append(b[j])
            j += 1
        else:
            if op in ("union", "intersection"):
                out.append(a[i])
            i += 1
            j += 1
    return out


def p_set(op):
    return lambda r, a, b: same(r, multiset_op(a, b, op))


def p_includes(f, a, b):
    return f == (multiset_op(b, a, "difference") == [])


def next_perm(x):
    a = list(x)
    i = len(a) - 2
    while i >= 0 and not before(a[i], a[i + 1]):
        i -= 1
    if i < 0:
        return sort_numpy(a), False
    j = len(a) - 1
    while not before(a[i], a[j]):
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = a[i + 1:][::-1]
    return a, True


def p_next_perm(r, f, x):
    want, flag = next_perm(x)
    return same(r, want) and f == flag


def p_is_permutation_answer(f, a, b):
    return f == p_permutation(a, b)


# name -> (python check, viv def or None, viv argument form)
PROPS = {
    "permutation": (p_permutation, "algo_is_permutation({0}, {1})"),
    "ordered": (p_ordered, "algo_is_sorted({0})"),
    "ordered-total": (p_ordered_total, "algo_is_sorted_total({0})"),
    "ordered-desc": (p_ordered_desc, "algo_is_sorted_desc({0})"),
    "stable-values": (p_stable_values, "algo_is_stable_values({0}, {1})"),
    "index-perm": (p_index_perm, "algo_is_index_perm({0}, len({1}))"),
    "ordered-by": (p_ordered_by, "algo_is_sorted_by({0}, {1})"),
    "stable-by": (p_stable_by, "algo_is_stable_by({0}, {1})"),
    "lower-bounds": (p_lower_bounds, "algo_all_lower_bounds({0}, {1}, {2})"),
    "upper-bounds": (p_upper_bounds, "algo_all_upper_bounds({0}, {1}, {2})"),
    "first-or-none": (p_first_or_none, "algo_all_first_or_none({0}, {1}, {2})"),
    "contains": (p_contains, None),
    "max-heap": (p_max_heap, "algo_is_max_heap({0})"),
    "min-pos": (p_min_pos, "algo_is_min_pos({0}, {1})"),
    "max-pos": (p_max_pos, "algo_is_max_pos({0}, {1})"),
    "is-sorted-answer": (p_is_sorted_answer, "algo_is_sorted_answer({0}, {1})"),
    "follows": (p_follows, "algo_follows({0}, {1}, {2})"),
    "gcd": (p_gcd, "algo_all_gcd({0}, {1}, {2})"),
    "lcm": (p_lcm, None),
    "factorial-count": (p_factorial_count, "algo_is_factorial_count({0}, {1})"),
    "topological": (p_topological, "algo_is_topological({0}, {1}, {2}, {3})"),
    "partial-sorted": (p_partial_sorted, None),
    "nth": (p_nth, None),
    "partitioned": (p_partitioned, None),
    "stable-partition": (p_stable_partition, None),
    "partition-point": (p_partition_point, None),
    "reversed": (p_reversed, None),
    "rotated": (p_rotated, None),
    "unique-sorted": (p_unique_sorted, None),
    "unique-runs": (p_unique_runs, None),
    "merged": (p_merged, None),
    "set-union": (p_set("union"), None),
    "set-intersection": (p_set("intersection"), None),
    "set-difference": (p_set("difference"), None),
    "set-symmetric-difference": (p_set("symmetric"), None),
    "includes": (p_includes, None),
    "next-perm": (p_next_perm, None),
    "is-permutation-answer": (p_is_permutation_answer, None),
}


# ================================================================ oracles (Python and NumPy)

def np_mod():
    import numpy as np  # only --run needs NumPy
    return np


def fl(a):
    return [float(v) for v in a]


def kahn(k, src, dst):
    """TheAlgorithms' kahns_algorithm_topo.py on the graph the keys and edges describe."""
    idx = {v: i for i, v in enumerate(k)}
    graph = {i: [] for i in range(len(k))}
    for s, d in zip(src, dst):
        graph[idx[s]].append(idx[d])
    indegree = [0] * len(graph)
    for values in graph.values():
        for i in values:
            indegree[i] += 1
    queue = [i for i in range(len(indegree)) if indegree[i] == 0]
    topo = []
    while queue:
        vertex = queue.pop(0)
        topo.append(vertex)
        for neighbor in graph[vertex]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return [k[i] for i in topo] if len(topo) == len(k) else []


ORACLES = {
    "np-sort-stable": lambda x: fl(np_mod().sort(np_mod().array(x, dtype=float), kind="stable")),
    "np-sort-total": lambda x: sorted(x, key=key_total),
    "np-argsort-stable": lambda x: fl(np_mod().argsort(np_mod().array(x, dtype=float), kind="stable")),
    "np-searchsorted-left": lambda v, q: fl(np_mod().searchsorted(np_mod().array(v, dtype=float), np_mod().array(q, dtype=float), side="left")),
    "np-searchsorted-right": lambda v, q: fl(np_mod().searchsorted(np_mod().array(v, dtype=float), np_mod().array(q, dtype=float), side="right")),
    "bisect-left": lambda v, q: [float(bisect.bisect_left(v, x)) for x in q],
    "bisect-right": lambda v, q: [float(bisect.bisect_right(v, x)) for x in q],
    "first-index": lambda v, q: [first_index(v, x) for x in q],
    "np-argmin": lambda x: float(np_mod().argmin(np_mod().array(x, dtype=float))),
    "np-argmax": lambda x: float(np_mod().argmax(np_mod().array(x, dtype=float))),
    "python-is-sorted": lambda x: all(x[i - 1] <= x[i] for i in range(1, len(x))),
    "sorted-desc": lambda x: sorted(x, reverse=True),
    "heapq-pops": lambda x: heapq_pops(x),
    "math-gcd": lambda a, b: [float(math.gcd(int(p), int(q))) for p, q in zip(a, b)],
    "math-lcm": lambda a, b: [float(math.lcm(int(p), int(q))) for p, q in zip(a, b)],
    "itertools-permutations": lambda x: [list(p) for p in itertools.permutations(x)],
    "itertools-next": lambda x: next_perm(x)[0],
    "kahn": kahn,
    "np-partition": lambda x, k: fl(np_mod().partition(np_mod().array(x, dtype=float), int(k))),
    "np-unique": lambda x: fl(np_mod().unique(np_mod().array(x, dtype=float))),
    "reversed": lambda x: x[::-1],
    "np-roll-left": lambda x, k: fl(np_mod().roll(np_mod().array(x, dtype=float), -int(k))),
}


def heapq_pops(x):
    h = []
    for i, c in enumerate(x):
        heapq.heappush(h, (c, float(i)))
    return [heapq.heappop(h)[0] for _ in range(len(x))]


# ================================================================ the spec, checked

def check_spec(blocks):
    errs = []
    algos = {b.id: b for b in blocks if b.kind == "algo"}
    orders = {b.id for b in blocks if b.kind == "order"}
    seen = set()
    for b in blocks:
        if (b.kind, b.id) in seen:
            errs.append(f"algo.spec:{b.line}: {b.kind} `{b.id}` is declared twice")
        seen.add((b.kind, b.id))
    for b in blocks:
        where = f"algo.spec:{b.line}: {b.kind} {b.id}"
        if b.kind == "algo":
            tier = b.get("tier")
            if tier not in ("1", "2", "3", "4"):
                errs.append(f"{where}: no tier 1 to 4")
            if not b.get("sig"):
                errs.append(f"{where}: no sig")
            if tier in ("1", "2") and not b.get("bound"):
                errs.append(f"{where}: a tier-{tier} algo states its bound (cppreference's, quoted)")
            if tier == "1" and not b.all("post"):
                errs.append(f"{where}: a tier-1 algo states its postconditions")
            if b.get("order") and b.get("order") not in orders:
                errs.append(f"{where}: no order `{b.get('order')}`")
            if b.get("default") and b.get("default") not in (b.get("members") or "").split():
                errs.append(f"{where}: the default is not a member")
        if b.kind == "impl":
            a = b.get("algo")
            if a not in algos:
                errs.append(f"{where}: no algo `{a}`")
            if not b.get("module"):
                errs.append(f"{where}: no module")
            if not b.all("call"):
                errs.append(f"{where}: no call")
            for k in (b.get("kinds") or "").split():
                if k not in KINDS:
                    errs.append(f"{where}: no kind `{k}`")
            for line in b.all("outside"):
                for k in line.split('"')[0].split():
                    if k not in KINDS:
                        errs.append(f"{where}: `outside` names no kind `{k}`")
        for key in ("post", "note"):
            for line in b.all(key):
                name = line.split()[0]
                if name not in PROPS:
                    errs.append(f"{where}: `{key} {name}` is no property this file checks")
        for line in b.all("oracle"):
            parts = line.split()
            if len(parts) < 3 or parts[1] not in ORACLES:
                errs.append(f"{where}: `oracle {line}` names no oracle this file knows")
    return errs


def impl_posts(impl, algos):
    """(property, args) the impl must satisfy: its own `post` lines, or its algo's."""
    own = impl.all("post")
    lines = own if own else algos[impl.get("algo")].all("post")
    return [(l.split()[0], l.split()[1:]) for l in lines]


def impl_notes(impl):
    return [(l.split()[0], l.split()[1:]) for l in impl.all("note")]


def impl_oracles(impl, algos):
    own = impl.all("oracle")
    lines = own if own else algos[impl.get("algo")].all("oracle")
    out = []
    for l in lines:
        parts = l.split()
        out.append((parts[0], parts[1], [p for p in parts[2:] if p != "info"], "info" in parts[2:]))
    return out


def impl_kinds(impl):
    return (impl.get("kinds") or " ".join(KINDS)).split()


def impl_sizes(impl):
    return [int(s) for s in (impl.get("sizes") or " ".join(map(str, SIZES))).split()]


def impl_outside(impl):
    out = {}
    for line in impl.all("outside"):
        kinds, _, why = line.partition('"')
        for k in kinds.split():
            out[k] = why.rstrip('"')
    return out


def impl_known(impl):
    """(property, kind) -> reason for results the spec records as known failures."""
    out = {}
    for line in impl.all("known"):
        head, _, why = line.partition('"')
        parts = head.split()
        for k in parts[1:]:
            out[(parts[0], k)] = why.rstrip('"')
    return out


# ================================================================ the generated programs

VARS = ["X", "R", "P", "D", "W", "V", "Q", "RS", "A", "B", "H", "F", "K", "SRC", "DST", "LO", "HI", "size"]


def module_of(impl):
    m = impl.get("module")
    return None if m == "builtin" else m


def program_path(module):
    if module is None:
        return os.path.join(ROOT, "lib", "algo", "tests", "builtins.viv")
    return os.path.join(ROOT, "lib", "algo", "legacy", "tests", os.path.basename(module) + ".viv")


def printed_vars(impl, algos):
    names = ["X"]
    for _, args in impl_posts(impl, algos) + impl_notes(impl):
        names += args
    for var, _, args, _ in impl_oracles(impl, algos):
        if var != "SHOWN":
            names.append(var)
        names += args
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def viv_name(impl):
    return impl.id.replace("-", "_")


def case_def(impl, algos):
    """A def that runs one case of `impl` and prints it; returns whether its in-domain
    properties hold."""
    name = viv_name(impl)
    lines = [f"#| {impl.id}: {impl.title}",
             f"def algo_t_{name}(kind, n, inside, skip: Arr) -> Bool {{",
             "    X = algo_case(kind, n, 1 + 97 * kind + n)"]
    lines.append(f'    print("@ {impl.id}", CASE_NAMES[kind], n)')
    for c in impl.all("call"):
        lines.append("    " + c)
    for v in printed_vars(impl, algos):
        # a flag or a count is printed as it is (fmt takes numbers and arrays only)
        lines.append(f'    print("= {v}", {v})' if v == "F" else f'    print("= {v}", fmt({v}))')
    checks = []
    for i, (prop, args) in enumerate(impl_posts(impl, algos) + impl_notes(impl)):
        tmpl = PROPS[prop][1]
        checks.append(tmpl.format(*args) if tmpl else "true")
    if checks:
        for i, c in enumerate(checks):
            lines.append(f"    p{i} = {c}")
        lines.append('    print("?", ' + ", ".join(f"p{i}" for i in range(len(checks))) + ")")
    nposts = len(impl_posts(impl, algos))
    ok = " and ".join(f"(p{i} or skip[{i}] == 1)" for i in range(nposts)) or "true"
    lines.append(f"    return inside == 0 or ({ok})")
    lines.append("}")
    return "\n".join(lines)


def impl_driver(impl, algos):
    name = viv_name(impl)
    kinds = impl_kinds(impl)
    outside = impl_outside(impl)
    known = impl_known(impl)
    posts = impl_posts(impl, algos)
    lines = [f"algo_t_ok_{name} = true",
             f"algo_t_kinds_{name} = [{', '.join(str(float(KINDS.index(k))) for k in kinds)}]",
             f"algo_t_inside_{name} = [{', '.join('0.0' if k in outside else '1.0' for k in kinds)}]",
             f"algo_t_sizes_{name} = [{', '.join(str(float(s)) for s in impl_sizes(impl))}]"]
    # the properties not asserted, per kind (the spec's `known` failures), one row per kind
    np_ = max(len(posts), 1)
    flags = []
    for k in kinds:
        flags += ["1.0" if (p, k) in known else "0.0" for p, _ in posts] or ["0.0"]
    lines.append(f"algo_t_skip_{name} = [{', '.join(flags)}]")
    lines.append(f"for algo_t_k to len(algo_t_kinds_{name}) {{")
    lines.append(f"    for algo_t_s to len(algo_t_sizes_{name}) {{")
    lines.append(f"        algo_t_r = algo_t_{name}(algo_t_kinds_{name}[algo_t_k], algo_t_sizes_{name}[algo_t_s], "
                 f"algo_t_inside_{name}[algo_t_k], algo_t_skip_{name}[algo_t_k * {np_}:(algo_t_k + 1) * {np_}])")
    lines.append(f"        algo_t_ok_{name} = algo_t_ok_{name} and algo_t_r")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def impl_test(impl, algos):
    name = viv_name(impl)
    posts = ", ".join(" ".join([p] + a) for p, a in impl_posts(impl, algos))
    inside = [k for k in impl_kinds(impl) if k not in impl_outside(impl)]
    return (f'test "{impl.id}: {posts} (kinds {" ".join(inside)})" {{\n'
            f"    assert_true(algo_t_ok_{name})\n}}")


def program_text(module, impls, algos):
    out = ["# Generated by tools/algo/check.py from lib/algo/algo.spec: do not edit.",
           "# Each case prints its input (= X), its results, and the verdict of each postcondition",
           "# (? ...) as lib/algo/props.viv computes it; check.py checks them again in Python.",
           "#! use lib/algo/cases",
           "#! use lib/algo/props"]
    if module:
        out.append(f"#! use {module}")
    for impl in impls:
        out.append("")
        out.append(case_def(impl, algos))
    for impl in impls:
        out.append("")
        out.append(impl_driver(impl, algos))
    for impl in impls:
        out.append("")
        out.append(impl_test(impl, algos))
    return "\n".join(out) + "\n"


def programs(blocks):
    algos = {b.id: b for b in blocks if b.kind == "algo"}
    by_module = {}
    for b in blocks:
        if b.kind == "impl":
            by_module.setdefault(module_of(b), []).append(b)
    return {program_path(m): migrated(program_text(m, impls, algos), program_path(m)) for m, impls in by_module.items()}


# ================================================================ running

def run_viv(path, timeout=1200):
    p = subprocess.run([VIV, "run", path], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    return p.stdout + p.stderr


def parse_cases(out):
    """impl -> list of {kind, n, vars, viv, shown}."""
    cases = {}
    cur = None
    for line in out.split("\n"):
        if line.startswith("@ "):
            _, impl, kind, n = line.split()
            cur = {"kind": kind, "n": int(float(n)), "vars": {}, "viv": [], "shown": []}
            cases.setdefault(impl, []).append(cur)
        elif cur is None or line.startswith("__"):
            continue
        elif line.startswith("= "):
            _, var, val = line.split(" ", 2)
            cur["vars"][var] = parse_value(val)
        elif line.startswith("? "):
            cur["viv"] = [v == "true" for v in line[2:].split()]
        elif line.startswith("current: "):
            cur["shown"].append(parse_value(line[len("current: "):]))
    return cases


def evaluate(impl, algos, cases):
    """The verdicts of one impl's cases."""
    posts = impl_posts(impl, algos)
    notes = impl_notes(impl)
    oracles = impl_oracles(impl, algos)
    outside = impl_outside(impl)
    known = impl_known(impl)
    res = {"id": impl.id, "algo": impl.get("algo"), "title": impl.title, "cases": len(cases), "props": {}, "oracles": {},
           "disagree": [], "unexpected": [], "known_now_pass": [], "examples": {}}
    expect = {(k, n) for k in impl_kinds(impl) for n in impl_sizes(impl)}
    got = {(c["kind"], c["n"]) for c in cases}
    if expect != got:
        res["unexpected"].append(f"ran {len(got)} of {len(expect)} cases (the program stopped?)")
    for idx, (prop, args) in enumerate(posts + notes):
        is_note = idx >= len(posts)
        label = ("note " if is_note else "") + " ".join([prop] + args)
        per_kind = {}
        for c in cases:
            vals = [c["vars"][a] for a in args]
            ok = bool(PROPS[prop][0](*vals))
            if c["viv"] and idx < len(c["viv"]) and PROPS[prop][1] and c["viv"][idx] != ok:
                res["disagree"].append(f"{label} on {c['kind']} n={c['n']}: Viv says {c['viv'][idx]}, Python {ok}")
            t = per_kind.setdefault(c["kind"], [0, 0])
            t[0 if ok else 1] += 1
            if not ok:
                ex = res["examples"].setdefault(label, {})
                if c["kind"] not in ex or len(c["vars"]["X"]) < len(ex[c["kind"]]["X"]):
                    ex[c["kind"]] = {a: c["vars"][a] for a in ["X"] + args}
        res["props"][label] = per_kind
        if is_note:
            continue
        for kind, (npass, nfail) in per_kind.items():
            if kind in outside:
                continue
            if nfail and (prop, kind) not in known:
                res["unexpected"].append(f"{label} fails on {nfail} {kind} cases")
            if not nfail and (prop, kind) in known:
                res["known_now_pass"].append(f"{label} on {kind}: recorded as known, now passes")
    for var, name, args, info in oracles:
        per_kind = {}
        for c in cases:
            try:
                want = ORACLES[name](*[c["vars"][a] for a in args])
            except Exception as e:  # an oracle that raises on an input (Python's own NaN policy)
                want = f"raises {type(e).__name__}"
            have = c["shown"] if var == "SHOWN" else c["vars"].get(var)
            if var == "SHOWN" and not isinstance(want, str):
                # `show` prints six significant digits (%g)
                want = [[float(f"{v:g}") for v in row] for row in want]
            ok = (not isinstance(want, str)) and same_any(have, want)
            t = per_kind.setdefault(c["kind"], [0, 0])
            t[0 if ok else 1] += 1
            if not ok:
                ex = res["examples"].setdefault(f"oracle {var} = {name}", {})
                if c["kind"] not in ex or len(c["vars"]["X"]) < len(ex[c["kind"]]["X"]):
                    ex[c["kind"]] = {"X": c["vars"]["X"], var: have, "want": want}
        res["oracles"][f"{var} = {name}" + (" (info)" if info else "")] = per_kind
        if info:
            continue
        for kind, (npass, nfail) in per_kind.items():
            if kind in outside or nfail == 0:
                continue
            if ("oracle", kind) not in known:
                res["unexpected"].append(f"{var} differs from {name} on {nfail} {kind} cases")
    return res


def same_any(have, want):
    if isinstance(want, list) and want and isinstance(want[0], list):
        return isinstance(have, list) and len(have) == len(want) and all(same(h, w) for h, w in zip(have, want))
    if isinstance(want, bool):
        return have == want
    return have is not None and same(have, want)


# ================================================================ complexity

def growth_of(impl, algos):
    g = impl.get("growth")
    if g:
        return g.split('"')[0].strip()
    b = algos[impl.get("algo")].get("bound") or ""
    parts = b.split('"')[0].split()
    return " ".join(parts[1:])


def complexity_program(impl, x_lit):
    module = module_of(impl)
    out = ["#! use lib/algo/cases", "#! use lib/algo/props", f"#! use {module}", "",
           "def algo_t_run(X: Arr, kind) {"]
    for c in impl.all("call"):
        out.append("    " + c)
    out.append("    return 0")
    out.append("}")
    out.append(f"algo_t_x = {x_lit}")
    out.append("algo_t_run(algo_t_x, 0)")
    return "\n".join(out) + "\n"


def adversary_program(impl, n):
    module = module_of(impl)
    # the adversary's items are lib/algo/legacy/order's LG_VAL, which the
    # program sizes: a module's global, so it is written through the module
    out = ["#! use lib/algo/cases", "#! use lib/algo/props", f"#! use {module}",
           "import algo.legacy.order as lg_order", "",
           "def algo_t_run(X: Arr, kind) {"]
    for c in impl.all("call"):
        out.append("    " + c)
    out.append("    return 0")
    out.append("}")
    out += [f"algo_t_ids = zeros({n})", f"for algo_t_i to {n} {{", "    algo_t_ids[algo_t_i] = algo_t_i", "}",
            f"lg_order.LG_VAL = zeros({n})", "lg_order.lg_adversary_start()", "algo_t_run(algo_t_ids, 0)", "lg_order.lg_adversary_stop()",
            'print("killer", fmt(lg_order.LG_VAL))']
    return "\n".join(out) + "\n"


def count_calls(out, defs):
    m = re.search(r"^__graph__ (.*)$", out, re.M)
    if not m:
        raise RuntimeError("no __graph__ line:\n" + out[-2000:])
    g = json.loads(m.group(1))
    return sum(e["count"] for e in g["edges"] if e["to"] in defs)


def fit(ns, cs):
    """Least-squares slope of log c against log n."""
    xs = [math.log(n) for n in ns]
    ys = [math.log(max(c, 1)) for c in cs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def judge(growth, ns, cs, per):
    """Whether counts c(n) stay within `growth`. Returns (ok, the measure, words)."""
    p = fit(ns, cs)
    top = ns[-1]
    if growth.startswith("exact"):
        expr = growth[len("exact"):].strip()
        # `3 n`, `n - 1`, `n / 2`: a product may be written as juxtaposition
        py = re.sub(r"(\d)\s*(?=[n(])", r"\1*", expr)
        limits = [eval(re.sub(r"\bn\b", str(n), py), {}) for n in ns]  # noqa: S307 (the spec's own text)
        ok = all(c <= lim for c, lim in zip(cs, limits))
        return ok, p, f"max count / limit = {max(c / lim for c, lim in zip(cs, limits)):.3f}"
    m = re.fullmatch(r"(\d*)\s*log n", growth)
    if m:
        # at most k log2(n) + O(1) per query; the O(1) is held to 4
        k = int(m.group(1) or 1)
        avg = [c / per_n for c, per_n in zip(cs, per)]
        excess = [a - k * math.log2(n) for a, n in zip(avg, ns)]
        ok = max(excess) <= 4.0
        return ok, p, f"per query: {avg[-1]:.2f} at n={top} ({k} log2 n = {k * math.log2(top):.0f}); largest excess over {k} log2 n {max(excess):.2f}"
    if growth == "n log n":
        ratio = [c / (n * math.log2(n)) for c, n in zip(cs, ns)]
        q = fit(ns, ratio) if min(ratio) > 0 else 0.0
        ok = q < 0.15
        return ok, p, f"count / (n log2 n) = {ratio[0]:.3f} at n={ns[0]}, {ratio[-1]:.3f} at n={top}; its slope {q:+.3f}"
    if growth == "n":
        return p < 1.15, p, f"exponent {p:.3f}"
    if growth == "n^2":
        return p < 2.15, p, f"exponent {p:.3f}"
    return True, p, f"exponent {p:.3f} (growth `{growth}` not judged)"


def complexity(impl, algos, tmp, log):
    defs = set(impl.get("count").split())
    growth = growth_of(impl, algos)
    per_var = impl.get("per")
    kinds = list(COMPLEXITY_KINDS)
    if impl.get("adversary") == "yes":
        kinds.append("adversary")
    rows = {}
    for kind in kinds:
        ns, cs, per = [], [], []
        for n in COMPLEXITY_SIZES:
            if kind == "adversary":
                src = os.path.join(tmp, f"{impl.id}_adv_{n}.viv")
                open(src, "w").write(migrated(adversary_program(impl, n), src))
                out = run_viv(src)
                m = re.search(r"^killer (.*)$", out, re.M)
                if not m:
                    raise RuntimeError(f"{impl.id}: the adversary run printed no killer input:\n{out[-1500:]}")
                adv_count = count_calls(out, defs)
                x_lit = m.group(1)
            else:
                x_lit = f"algo_case({float(KINDS.index(kind))}, {float(n)}, {float(1 + 97 * KINDS.index(kind) + n)})"
                adv_count = None
            src = os.path.join(tmp, f"{impl.id}_{kind}_{n}.viv")
            open(src, "w").write(migrated(complexity_program(impl, x_lit), src))
            out = run_viv(src)
            c = count_calls(out, defs)
            if adv_count is not None and adv_count != c:
                log.append(f"{impl.id}: the adversary saw {adv_count} comparisons at n={n}, its input replayed {c}")
            ns.append(n)
            cs.append(c)
            per.append(2 * n + 2 if per_var == "Q" else 1)
        ok, p, words = judge(growth, ns, cs, per)
        rows[kind] = {"n": ns, "count": cs, "exponent": round(p, 3), "ok": ok, "words": words}
    return {"growth": growth, "defs": sorted(defs), "kinds": rows}


# ================================================================ self-test of the properties

def self_test():
    """Each tier-1 postcondition holds on a Python reference and catches its mutants."""
    rng = random.Random(20261001)
    pools = {
        "random": lambda n: [rng.random() - 0.5 for _ in range(n)],
        "dups": lambda n: [float(rng.randrange(4)) for _ in range(n)],
        "zeros": lambda n: [rng.choice([0.0, -0.0, 1.0, -1.0]) for _ in range(n)],
        "special": lambda n: [rng.choice([0.0, -0.0, 1.0, -1.0, math.nan, math.inf, -math.inf, 2.5]) for _ in range(n)],
    }
    nan_last_sorted = lambda x: sorted(x, key=key_numpy)
    swap_first_pair = lambda r: r[1:2] + r[:1] + r[2:] if len(r) > 1 and not same(r[0], r[1]) else None

    def mut_drop_last(r):
        return r[:-1] + [r[0]] if len(r) > 1 and not same(r[-1], r[0]) else None

    def mut_unstable(r):
        # swap the first -0.0 and 0.0 that are both present
        z = [i for i, v in enumerate(r) if v == 0.0]
        for i in z:
            for j in z:
                if i < j and math.copysign(1, r[i]) != math.copysign(1, r[j]):
                    s = list(r)
                    s[i], s[j] = s[j], s[i]
                    return s
        return None

    def off_by_one(rs):
        return [rs[0] + 1.0] + rs[1:] if rs else None

    k_of = lambda x: len(x) // 2
    tests = [
        # (property, make args from x, reference result builder, [mutants of the result])
        ("permutation", lambda x: (nan_last_sorted(x), x), [mut_drop_last]),
        ("ordered", lambda x: (nan_last_sorted(x),), [swap_first_pair]),
        ("ordered-total", lambda x: (sorted(x, key=key_total),), [swap_first_pair, mut_unstable]),
        ("stable-values", lambda x: (nan_last_sorted(x), x), [mut_unstable]),
        ("index-perm", lambda x: (sorted(range(len(x)), key=lambda i: key_numpy(x[i])), x), [lambda p: [p[0]] * len(p) if len(p) > 1 else None]),
        ("ordered-by", lambda x: ([float(i) for i in sorted(range(len(x)), key=lambda i: key_numpy(x[i]))], x), [swap_first_pair]),
        ("stable-by", lambda x: ([float(i) for i in sorted(range(len(x)), key=lambda i: key_numpy(x[i]))], x),
         [lambda p: ([float(i) for i in sorted(range(len(x)), key=lambda i: (key_numpy(x[i]), -i))] if len(set(map(bits, x))) < len(x) else None)]),
        ("lower-bounds", lambda x: ([float(lower(nan_last_sorted(x), q)) for q in needles(x)], nan_last_sorted(x), needles(x)), [off_by_one]),
        ("upper-bounds", lambda x: ([float(upper(nan_last_sorted(x), q)) for q in needles(x)], nan_last_sorted(x), needles(x)), [off_by_one]),
        ("partial-sorted", lambda x: (nan_last_sorted(x)[:k_of(x)] + x[k_of(x):] if False else partial(x, k_of(x)), x, k_of(x)), [lambda r: swap_first_pair(r)]),
        ("nth", lambda x: (nth_ref(x, k_of(x)), x, k_of(x)) if x else ([], x, 0), [lambda r: list(reversed(r)) if len(set(map(bits, r))) > 2 else None]),
        ("partitioned", lambda x: ([v for v in x if pos_pred(v)] + [v for v in x if not pos_pred(v)], sum(map(pos_pred, x))), [lambda r: r[::-1] if any(map(pos_pred, r)) and not all(map(pos_pred, r)) else None]),
        ("stable-partition", lambda x: ([v for v in x if pos_pred(v)] + [v for v in x if not pos_pred(v)], x, sum(map(pos_pred, x))), [swap_first_pair]),
        ("reversed", lambda x: (x[::-1], x), [lambda r: r[::-1] if not same(r, r[::-1]) else None]),
        ("rotated", lambda x: (x[k_of(x):] + x[:k_of(x)], x, k_of(x)), [lambda r: r[1:] + r[:1] if len(set(map(bits, r))) > 1 else None]),
        ("unique-sorted", lambda x: ([v for v, _ in groups(x)], x), [lambda r: r + r[:1] if r else None]),
        ("unique-runs", lambda x: (runs(x), x), [lambda r: r[:-1] if r else None]),
        ("merged", lambda x: (stable_merge(nan_last_sorted(x[::2]), nan_last_sorted(x[1::2])), nan_last_sorted(x[::2]), nan_last_sorted(x[1::2])), [swap_first_pair]),
        ("set-union", lambda x: (multiset_op(*halves(x), "union"),) + halves(x), [mut_drop_last, lambda r: r[:-1] if r else None]),
        ("set-intersection", lambda x: (multiset_op(*halves(x), "intersection"),) + halves(x), [lambda r: r + r[-1:] if r else None]),
        ("set-difference", lambda x: (multiset_op(*halves(x), "difference"),) + halves(x), [lambda r: r[:-1] if r else None]),
        ("set-symmetric-difference", lambda x: (multiset_op(*halves(x), "symmetric"),) + halves(x), [lambda r: r[:-1] if r else None]),
        ("next-perm", lambda x: (next_perm(x)[0], next_perm(x)[1], x), [lambda r: nan_last_sorted(r) if not same(nan_last_sorted(r), r) else None]),
        ("min-pos", lambda x: (float(min(range(len(x)), key=lambda i: (key_numpy(x[i]), i))), x) if x else (0.0, [0.0]), [lambda r: r + 1.0]),
    ]
    report = []
    bad = 0
    for prop, make, mutants in tests:
        holds = caught = tried = 0
        for kind, pool in pools.items():
            for n in (0, 1, 2, 3, 5, 8, 17, 33):
                x = pool(n)
                args = list(make(x))
                if not PROPS[prop][0](*args):
                    bad += 1
                    report.append(f"self-test: the reference breaks `{prop}` on {kind} n={n}: {args}")
                    continue
                holds += 1
                for mut in mutants:
                    m = mut(args[0])
                    if m is None:
                        continue
                    tried += 1
                    if not PROPS[prop][0](m, *args[1:]):
                        caught += 1
        status = "ok" if tried == 0 or caught == tried else "MISSES MUTANTS"
        if tried and caught != tried:
            # a mutant that is still correct (e.g. swapping equal values) is not a miss; report it
            status = f"caught {caught} of {tried} mutants (the rest are still correct results)"
        report.append(f"  {prop:26s} reference holds on {holds} inputs; {status}")
        if tried == 0:
            report[-1] += " (no mutant applied)"
    return bad, report


def needles(x):
    v = sorted(x, key=key_numpy)
    return [q for q in v] + [q + 0.5 for q in v if not math.isnan(q)] + [-1e9, 1e9]


def partial(x, k):
    s = sorted(x, key=key_numpy)
    rest = list(x)
    for v in s[:k]:
        for i, w in enumerate(rest):
            if same(v, w):
                del rest[i]
                break
    return s[:k] + rest


def nth_ref(x, k):
    s = sorted(x, key=key_numpy)
    return s[:k][::-1] + [s[k]] + s[k + 1:][::-1]


def runs(x):
    out = []
    for v in x:
        if not out or before(out[-1], v) or before(v, out[-1]):
            out.append(v)
    return out


def halves(x):
    return sorted(x[::2], key=key_numpy), sorted(x[1::2], key=key_numpy)


# ================================================================ engines

def three_engines(dirs):
    lines, ok = [], True
    for d in dirs:
        p = subprocess.run([os.path.join(ROOT, "tools", "three-engines.sh"), d], capture_output=True, text=True, cwd=ROOT)
        lines.append(f"tools/three-engines.sh {d}")
        lines += ["  " + l for l in p.stdout.strip().split("\n") if l]
        if p.returncode != 0:
            ok = False
            lines.append(f"  FAILED (exit {p.returncode})")
            lines += ["  " + l for l in p.stderr.strip().split("\n")[-10:] if l]
    return ok, lines


# ================================================================ ledger

INPUTS = ["lib/algo/algo.spec", "lib/algo/props.viv", "lib/algo/cases.viv", "tools/algo/check.py", "tools/algo/legacy.py"]


def inputs_hash():
    h = hashlib.sha256()
    names = list(INPUTS) + sorted("lib/algo/legacy/" + f for f in os.listdir(os.path.join(ROOT, "lib", "algo", "legacy")) if f.endswith(".viv"))
    for rel in names:
        h.update(rel.encode())
        with open(os.path.join(ROOT, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def static_checks(blocks, errors):
    errs = list(errors) + check_spec(blocks)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "algo", "legacy.py"), "--check"], capture_output=True, text=True)
    if p.returncode != 0:
        errs.append(p.stdout.strip() or "legacy.py --check failed")
    for path, text_ in programs(blocks).items():
        rel = os.path.relpath(path, ROOT)
        if not os.path.exists(path) or open(path).read() != text_:
            errs.append(f"{rel} differs from what check.py writes; run tools/algo/check.py --write")
    return errs


# ================================================================ report

def fmt_kinds(per_kind):
    return ", ".join(f"{k} {p}/{p + f}" for k, (p, f) in per_kind.items())


def short(v, width=110):
    s = text(v) if not isinstance(v, str) else v
    return s if len(s) <= width else s[:width] + " ..."


def write_report(results, comp, selftest, engines, log):
    out = ["lib/algo harness report (tools/algo/check.py --run). Counts are cases passing / cases run.", ""]
    for r in results:
        out.append(f"impl {r['id']}  ({r['algo']})  {r['title']}")
        for label, per_kind in r["props"].items():
            fails = sum(f for _, f in per_kind.values())
            out.append(f"  {'ok  ' if not fails else 'FAIL'} {label}: {fmt_kinds(per_kind)}")
        for label, per_kind in r["oracles"].items():
            fails = sum(f for _, f in per_kind.values())
            out.append(f"  {'ok  ' if not fails else 'DIFF'} = {label}: {fmt_kinds(per_kind)}")
        for label, ex in r["examples"].items():
            for kind, vals in list(ex.items())[:2]:
                out.append(f"       smallest failing {kind} case of `{label}`: " + "; ".join(f"{k} = {short(v)}" for k, v in vals.items()))
        for d in r["disagree"]:
            out.append(f"  VIV AND PYTHON DISAGREE: {d}")
        for u in r["unexpected"]:
            out.append(f"  UNEXPECTED: {u}")
        for u in r["known_now_pass"]:
            out.append(f"  NOTE: {u}")
        if r["id"] in comp:
            c = comp[r["id"]]
            out.append(f"  comparisons ({' '.join(c['defs'])}), held to `{c['growth']}`:")
            for kind, row in c["kinds"].items():
                counts = " ".join(str(x) for x in row["count"])
                out.append(f"    {'ok  ' if row['ok'] else 'OVER'} {kind:9s} n={row['n'][0]}..{row['n'][-1]}: {counts}; exponent {row['exponent']}; {row['words']}")
        out.append("")
    out.append("self-test of the postconditions (Python references and mutants):")
    out += selftest
    out.append("")
    out += engines
    out += log
    return "\n".join(out) + "\n"


# ================================================================ main

def main():
    args = sys.argv[1:]
    only = args[args.index("--only") + 1] if "--only" in args else None
    blocks, errors = parse_spec()
    if "--write" in args or "--run" in args:
        errs = list(errors) + check_spec(blocks)
        if errs:
            print("\n".join(errs))
            sys.exit(1)
        for path, text_ in programs(blocks).items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text_)
        if "--run" not in args:
            print(f"{len(programs(blocks))} test programs written")
            return
    if "--run" not in args:
        errs = static_checks(blocks, errors)
        if os.path.exists(LEDGER):
            ledger = json.load(open(LEDGER))
            if ledger.get("inputs") != inputs_hash():
                errs.append("tools/algo/results/ledger.json is older than lib/algo or tools/algo; run tools/algo/check.py --run")
            if ledger.get("unexpected"):
                errs.append(f"the ledger records {len(ledger['unexpected'])} unexpected results")
        else:
            errs.append("no ledger; run tools/algo/check.py --run")
        if errs:
            print("\n".join(errs))
            sys.exit(1)
        print(f"lib/algo: spec, legacy copies, test programs and ledger are current ({sum(1 for b in blocks if b.kind == 'impl')} impls)")
        return

    algos = {b.id: b for b in blocks if b.kind == "algo"}
    impls = [b for b in blocks if b.kind == "impl" and (only is None or only in b.id)]
    results, comp, log = [], {}, []
    outputs = {}
    for path in sorted({program_path(module_of(i)) for i in impls}):
        print(f"running {os.path.relpath(path, ROOT)}", flush=True)
        outputs[path] = parse_cases(run_viv(path))
    for impl in impls:
        cases = outputs[program_path(module_of(impl))].get(impl.id, [])
        results.append(evaluate(impl, algos, cases))
    # inside the repository, so that imports of lib/ resolve from the programs written there
    with tempfile.TemporaryDirectory(prefix=".work-", dir=os.path.join(ROOT, "tools", "algo")) as tmp:
        for impl in impls:
            if impl.get("count"):
                print(f"counting {impl.id}", flush=True)
                comp[impl.id] = complexity(impl, algos, tmp, log)
    bad_self, selftest = self_test()
    engines_ok, engines = True, []
    if "--no-engines" not in args:
        print("three engines", flush=True)
        engines_ok, engines = three_engines(["lib/algo", "lib/algo/legacy", "lib/algo/tests", "lib/algo/legacy/tests"])
    unexpected = []
    for r in results:
        unexpected += [f"{r['id']}: {u}" for u in r["unexpected"] + r["disagree"]]
    for iid, c in comp.items():
        impl = next(b for b in impls if b.id == iid)
        for kind, row in c["kinds"].items():
            if not row["ok"] and (("growth", kind) not in impl_known(impl)):
                unexpected.append(f"{iid}: comparisons on {kind} exceed `{c['growth']}` ({row['words']})")
    if bad_self:
        unexpected.append(f"self-test: {bad_self} references break their own postconditions")
    if not engines_ok:
        unexpected.append("tools/three-engines.sh failed")
    report = write_report(results, comp, selftest, engines, log)
    os.makedirs(RESULTS, exist_ok=True)
    if only is None:
        with open(REPORT, "w") as f:
            f.write(report)
        with open(LEDGER, "w") as f:
            json.dump({"inputs": inputs_hash(), "unexpected": unexpected,
                       "impls": {r["id"]: {"props": r["props"], "oracles": r["oracles"]} for r in results},
                       "complexity": comp}, f, indent=1, sort_keys=True)
            f.write("\n")
    print(report)
    if unexpected:
        print("UNEXPECTED:\n  " + "\n  ".join(unexpected))
        sys.exit(1)


if __name__ == "__main__":
    main()

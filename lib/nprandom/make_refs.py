#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write lib/nprandom/selftest.viv: NumPy's own draws, and the Viv that must reproduce them.

    python3 lib/nprandom/make_refs.py            write the file (needs NumPy 1.24.4)
    python3 lib/nprandom/make_refs.py --check    fail if the committed file differs

Each case below is a pair: Python that draws with NumPy, and Viv that draws the same numbers
with lib/nprandom. The script runs the Python, writes its values into the Viv as the
expected array, and the Viv test compares them exactly (tolerance 0). Run the result in the
three engines with tools/three-engines.sh lib/nprandom selftest.

The normal draws are the exception: NumPy's ziggurat calls the platform's log1p in the
base strip's tail, which Viv cannot reproduce, and lib/nprandom returns NaN there. For
those cases the script also finds, by replaying the ziggurat on the raw PCG64 outputs
(`bit_generator.random_raw`, with this platform's math.exp in the wedge test, as NumPy
does), which draw first reaches the tail, and the test checks that every draw before it
agrees and that draw is NaN.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "selftest.viv")
sys.path.insert(0, HERE)
import make_tables  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tools/imports'))
from viv_imports import migrated  # noqa: E402  (tools/imports/viv_imports.py)

NUMPY = "1.24.4"


def legacy(seed):
    rs = np.random.RandomState()
    rs.seed(seed)
    return rs


def floats(xs):
    return [float(x) for x in np.asarray(xs).ravel()]


# (name, python returning a flat list of numbers, viv statements leaving the result in `got`)
CASES = [
    ("legacy random_sample after seed 0", lambda: floats(legacy(0).random_sample(6)), """
st = npr_mt_new()
npr_mt_seed(st, 0)
got = []
for i to len(REF) {
    got = append(got, npr_mt_double(st))
}"""),
    ("legacy seed 2**32 - 1, 1500 draws (three twists)", lambda: floats(legacy(2**32 - 1).random_sample(1500)[-5:]), """
n_draws = 1500
st = npr_mt_new()
npr_mt_seed(st, 4294967295)
got = []
for i to n_draws {
    v = npr_mt_double(st)
    if i >= n_draws - len(REF) {
        got = append(got, v)
    }
}"""),
    ("legacy seed [1, 2, 3] (init_by_array)", lambda: floats(legacy([1, 2, 3]).random_sample(4)), """
st = npr_mt_new()
npr_mt_seed_array(st, [1, 2, 3])
got = []
for i to len(REF) {
    got = append(got, npr_mt_double(st))
}"""),
    ("legacy seed of 700 words (a key longer than the state)", lambda: floats(legacy(np.arange(700) * 7919 % 2**32).random_sample(3)), """
key_len = 700
key_step = 7919
key = []
for i to key_len {
    key = append(key, mod(i * key_step, 4294967296.0))
}
st = npr_mt_new()
npr_mt_seed_array(st, key)
got = []
for i to len(REF) {
    got = append(got, npr_mt_double(st))
}"""),
    ("legacy uniform(-2, 3) and rand interleaved", lambda: (lambda rs: floats([rs.uniform(-2.0, 3.0), rs.rand(), rs.uniform(5.0, 1.0)]))(legacy(42)), """
st = npr_mt_new()
npr_mt_seed(st, 42)
got = [npr_lg_uniform(st, -2.0, 3.0 - -2.0), npr_mt_double(st), npr_lg_uniform(st, 5.0, 1.0 - 5.0)]"""),
    ("legacy randint over ranges 10, 0, 2**32 - 1, 2**32 + 5, 2**40, 7", lambda: (lambda rs: [int(v) for v in [
        *rs.randint(0, 10, size=8), rs.randint(5, 6), rs.randint(0, 2**32), rs.randint(-3, 2**32 + 3),
        rs.randint(0, 2**40), rs.randint(-3, 4)]])(legacy(1)), """
n_small = 8
st = npr_mt_new()
npr_mt_seed(st, 1)
got = []
for i to n_small {
    got = append(got, npr_lg_bounded(st, 10 - 1))
}
got = append(got, 5 + npr_lg_bounded(st, 0))
got = append(got, npr_lg_bounded(st, 4294967295))
got = append(got, -3 + npr_lg_bounded(st, 4294967296.0 + 3 - 1 + 3))
got = append(got, npr_lg_bounded(st, 1099511627776.0 - 1))
got = append(got, -3 + npr_lg_bounded(st, 4 - 1 + 3))"""),
    ("legacy shuffle, permutation, rows", lambda: (lambda rs: [*(lambda a: (rs.shuffle(a), a)[1])(np.arange(10.0)), *rs.permutation(7),
                                                               *(lambda A: (rs.shuffle(A), A)[1])(np.arange(15.0).reshape(5, 3)).ravel()])(legacy(3)), """
n_a = 10
n_p = 7
rows = 5
cols = 3
st = npr_mt_new()
npr_mt_seed(st, 3)
a = st_iota(n_a)
npr_lg_shuffle(st, a)
p = st_iota(n_p)
npr_lg_shuffle(st, p)
A = st_grid(rows, cols)
npr_lg_shuffle_rows(st, A)
got = concat(concat(a, p), st_flat(A))"""),
    ("legacy choice with replacement, with p, without replacement", lambda: (lambda rs: [int(v) for v in [
        *rs.choice(10, 5), *rs.choice(4, 6, p=[0.1, 0.2, 0.3, 0.4]), *rs.choice(10, 4, replace=False)]])(legacy(7)), """
pop = 10
n_pick = 5
n_p = 6
n_unique = 4
st = npr_mt_new()
npr_mt_seed(st, 7)
got = []
for i to n_pick {
    got = append(got, npr_lg_bounded(st, pop - 1))
}
cdf = npr_lg_cdf([0.1, 0.2, 0.3, 0.4])
u = []
for i to n_p {
    u = append(u, npr_mt_double(st))
}
for i to n_p {
    got = append(got, npr_lg_search(cdf, u[i]))
}
perm = st_iota(pop)
npr_lg_shuffle(st, perm)
for i to n_unique {
    got = append(got, perm[i])
}"""),
    ("the SeedSequence pools of 0, 42, 2**32 - 1, 2**40 + 3 and [1, 2, 3, 4, 5]", lambda: [int(v) for s in (0, 42, 2**32 - 1, 2**40 + 3, [1, 2, 3, 4, 5]) for v in np.random.SeedSequence(s).pool], """
seeds = [0, 42, 4294967295, 1099511627779.0]
got = []
for s to len(seeds) {
    got = concat(got, npr_ss_pool(npr_ss_words(seeds[s])))
}
got = concat(got, npr_ss_pool([1, 2, 3, 4, 5]))"""),
    ("default_rng random for seeds 0, 1, 2**32 - 1, 2**53 - 1", lambda: [float(v) for s in (0, 1, 2**32 - 1, 2**53 - 1) for v in np.random.default_rng(s).random(3)], """
seeds = [0, 1, 4294967295, 9007199254740991.0]
n_each = 3
got = []
for s to len(seeds) {
    rng = npr_pcg_from_entropy(npr_ss_words(seeds[s]))
    for i to n_each {
        got = append(got, npr_pcg_double(rng))
    }
}"""),
    ("the last three of 1000 draws from default_rng(0)", lambda: floats(np.random.default_rng(0).random(1000)[-3:]), """
n_draws = 1000
rng = npr_pcg_from_entropy([0])
got = []
for i to n_draws {
    v = npr_pcg_double(rng)
    if i >= n_draws - len(REF) {
        got = append(got, v)
    }
}"""),
    ("Generator.integers over ranges 10, 0, 2**32 - 1, 2**32 + 5, 2**40, 2**53 - 2, with random between", lambda: (lambda g: [float(v) for v in [
        *g.integers(0, 10, size=5), g.integers(5, 6), g.random(), g.integers(0, 2**32), g.integers(-3, 2**32 + 3),
        g.integers(0, 2**40), g.integers(0, 2**53 - 1), g.integers(1, 7, endpoint=True), g.random()]])(np.random.default_rng(5)), """
n_small = 5
rng = npr_pcg_from_entropy([5])
got = []
for i to n_small {
    got = append(got, npr_gn_bounded(rng, 10 - 1))
}
got = append(got, 5 + npr_gn_bounded(rng, 0))
got = append(got, npr_pcg_double(rng))
got = append(got, npr_gn_bounded(rng, 4294967295))
got = append(got, -3 + npr_gn_bounded(rng, 4294967296.0 + 3 - 1 + 3))
got = append(got, npr_gn_bounded(rng, 1099511627776.0 - 1))
got = append(got, npr_gn_bounded(rng, 9007199254740991.0 - 1))
got = append(got, 1 + npr_gn_bounded(rng, 7 - 1))
got = append(got, npr_pcg_double(rng))"""),
    ("Generator.uniform, shuffle, permutation, rows", lambda: (lambda g: floats([g.uniform(-1.0, 4.0), *(lambda a: (g.shuffle(a), a)[1])(np.arange(10.0)),
                                                                              *g.permutation(6), *(lambda A: (g.shuffle(A), A)[1])(np.arange(12.0).reshape(4, 3)).ravel()]))(np.random.default_rng(11)), """
n_a = 10
n_p = 6
rows = 4
cols = 3
rng = npr_pcg_from_entropy([11])
got = [npr_gn_uniform(rng, -1.0, 4.0 - -1.0)]
a = st_iota(n_a)
npr_gn_shuffle(rng, a)
p = st_iota(n_p)
npr_gn_shuffle(rng, p)
A = st_grid(rows, cols)
npr_gn_shuffle_rows(rng, A)
got = concat(concat(concat(got, a), p), st_flat(A))"""),
    ("Generator.choice with replacement, with p, without (Floyd, shuffled and not), and from 20000", lambda: (lambda g: [float(v) for v in [
        *g.choice(10, 5), *g.choice(4, 6, p=[0.1, 0.2, 0.3, 0.4]), *g.choice(10, 4, replace=False),
        *g.choice(100, 6, replace=False, shuffle=False), *g.choice(20000, 500, replace=False)[-4:], g.random()]])(np.random.default_rng(13)), """
pop = 10
n_pick = 5
n_p = 6
n_unique = 4
big_pop = 100
n_big = 6
huge_pop = 20000
n_huge = 500
n_tail = 4
rng = npr_pcg_from_entropy([13])
got = []
for i to n_pick {
    got = append(got, npr_gn_bounded(rng, pop - 1))
}
cdf = npr_lg_cdf([0.1, 0.2, 0.3, 0.4])
u = []
for i to n_p {
    u = append(u, npr_pcg_double(rng))
}
for i to n_p {
    got = append(got, npr_lg_search(cdf, u[i]))
}
got = concat(got, npr_gn_choice_noreplace(rng, pop, n_unique, 1))
got = concat(got, npr_gn_choice_noreplace(rng, big_pop, n_big, 0))
c = npr_gn_choice_noreplace(rng, huge_pop, n_huge, 1)
for i to n_tail {
    got = append(got, c[n_huge - n_tail + i])
}
got = append(got, npr_pcg_double(rng))"""),
]


def zig_first_tail(seed, n):
    """Replay random_standard_normal on the raw outputs of default_rng(seed): the index of
    the first of n draws that reaches the base strip's tail, or n."""
    ki, fi = ZIG["ki"], ZIG["fi"]
    raw = np.random.default_rng(seed).bit_generator.random_raw(20 * n + 100)
    pos = 0
    for k in range(n):
        while True:
            r = int(raw[pos])
            pos += 1
            idx = r & 0xFF
            r >>= 8
            sign = r & 1
            rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
            x = rabs * ZIG["wi"][idx]
            if sign:
                x = -x
            if rabs < ki[idx]:
                break
            if idx == 0:
                return k
            u = (int(raw[pos]) >> 11) * (1.0 / 9007199254740992.0)
            pos += 1
            if (fi[idx - 1] - fi[idx]) * u + fi[idx] < math.exp(-0.5 * x * x):
                break
    return n


ZIG = {}


def load_zig():
    import re
    text = open(make_tables.OUT).read()

    def arr(name):
        m = re.search(r"%s = \[(.*?)\]" % name, text, re.S)
        return [float(t) for t in m.group(1).replace("\n", " ").split(",") if t.strip()]
    ZIG["ki"] = [int(v) for v in arr("npr_zig_ki")]
    ZIG["wi"] = arr("npr_zig_wi")
    ZIG["fi"] = arr("npr_zig_fi")


def normal_cases():
    """(name, seed, n, first_tail, values before it) for the ziggurat's tests."""
    out = []
    for seed, n in ((0, 1000), (42, 3000), (2, 6000)):
        t = zig_first_tail(seed, n)
        vals = floats(np.random.default_rng(seed).standard_normal(n))
        out.append((seed, n, t, vals[:min(t, n)]))
    return out


def viv_list(vals):
    rows = [", ".join(repr(v) for v in vals[i:i + 4]) for i in range(0, len(vals), 4)]
    return "[\n        " + ",\n        ".join(rows) + "\n    ]"


def render():
    load_zig()
    parts = ["""###
post: lib/nprandom against NumPy's own draws
agent: claude

Generated by lib/nprandom/make_refs.py with NumPy %s; do not edit, rerun the
script. Each test draws with lib/nprandom what the script drew with NumPy and
compares every value exactly. Run it in the three engines with
`tools/three-engines.sh lib/nprandom selftest`.
###

#! use lib/nprandom/legacy
#! use lib/nprandom/generator

#| 0, 1, ..., n - 1.
def st_iota(n) {
    a = zeros(n)
    for i to n {
        a[i] = i
    }
    return a
}

#| A rows by cols matrix holding 0, 1, 2, ... row by row.
def st_grid(rows, cols) {
    A = zeros(rows, cols)
    for i to rows {
        for j to cols {
            A[i, j] = cols * i + j
        }
    }
    return A
}

#| A matrix's entries, row by row.
def st_flat(A: Arr) {
    out = []
    for i to shape(A)[0] {
        for j to shape(A)[1] {
            out = append(out, A[i, j])
        }
    }
    return out
}
""" % NUMPY]
    for name, py, viv in CASES:
        ref = py()
        body = "\n".join("    " + l if l else "" for l in viv.strip("\n").splitlines())
        parts.append('test "%s" {\n    REF = %s\n%s\n    assert_allclose(got, REF, 0.0)\n}\n' % (name, viv_list(ref), body))
    for seed, n, t, vals in normal_cases():
        name = "Generator.standard_normal from seed %d agrees on the %d draws before the first tail draw (draw %d)" % (seed, len(vals), t)
        nan_check = ""
        if t < n:
            nan_check = "\n    tail = npr_gn_normal(rng)\n    assert_true(tail != tail)"
        parts.append("""test "%s" {
    REF = %s
    rng = npr_pcg_from_entropy([%d])
    got = zeros(len(REF))
    for i to len(REF) {
        got[i] = npr_gn_normal(rng)
    }
    assert_allclose(got, REF, 0.0)%s
}
""" % (name, viv_list(vals), seed, nan_check))
    # normal(loc, scale) is loc + scale * z, no fused multiply-add in NumPy's build
    g = np.random.default_rng(9)
    ref = floats([g.normal(1.5, 2.0), g.normal(-3.0, 0.5), g.standard_normal()])
    parts.append("""test "Generator.normal(loc, scale) is loc + scale * z" {
    REF = %s
    rng = npr_pcg_from_entropy([9])
    got = [1.5 + 2.0 * npr_gn_normal(rng), -3.0 + 0.5 * npr_gn_normal(rng), npr_gn_normal(rng)]
    assert_allclose(got, REF, 0.0)
}
""" % viv_list(ref))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if np.__version__ != NUMPY:
        sys.exit("make_refs.py: NumPy %s here; the references are NumPy %s's" % (np.__version__, NUMPY))
    out = migrated(render(), OUT)
    if a.check:
        if open(OUT).read() != out:
            sys.exit("make_refs.py: %s differs from NumPy's draws" % OUT)
        print("selftest.viv: current")
        return
    with open(OUT, "w") as f:
        f.write(out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

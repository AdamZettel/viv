#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""probe_numpy_order.py: the order in which NumPy adds up an array, checked bit for bit.

    python3 tools/py2viv/probe_numpy_order.py            print the checks; exit 1 on a mismatch
    python3 tools/py2viv/probe_numpy_order.py --write    also write results/axis_order.txt

py2viv's reduction helpers (rules/helpers.toml: py_pw, py_blk, py_sum_fl, py_var_fl) claim to
add in NumPy's order. This script states that order as Python code (the same arithmetic,
one IEEE operation at a time) and compares it with NumPy on random arrays whose entries span
forty orders of magnitude, so that a different order changes the last bits:

  - a contiguous run of n entries is summed from 0.0, block by block of 8192 entries (NumPy's
    buffer), each block by NumPy's pairwise sum (numpy/core/src/umath/loops_utils.h);
  - a sum along the last axis (every later axis of length 1) is that, per output entry; along
    any other axis, the slabs are added in order from 0.0;
  - np.mean divides the sum; np.var subtracts the kept mean, squares, sums the same way;
  - a slice with positive steps sums as its copy does, along any axis and flat;
  - a transpose sums as the array it came from (memory order), which its row-order copy
    does not: py2viv sums the original instead, or refuses.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUF = 8192


def pw(f, lo, n):
    if n < 8:
        s = 0.0
        for i in range(n):
            s += f[lo + i]
        return s
    if n <= 128:
        r = [f[lo + j] for j in range(8)]
        i = 8
        while i < n - n % 8:
            for j in range(8):
                r[j] += f[lo + i + j]
            i += 8
        s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            s += f[lo + i]
            i += 1
        return s
    n2 = n // 2
    n2 -= n2 % 8
    return pw(f, lo, n2) + pw(f, lo + n2, n - n2)


def blk(f, lo, n):
    s = 0.0
    for b in range(0, n, BUF):
        s = s + pw(f, lo + b, min(BUF, n - b))
    return s


def sum_axis(a, k):
    """py_sum_fl: a C-ordered copy of a, summed along axis k."""
    d = a.shape
    before = int(np.prod(d[:k]))
    along = d[k]
    after = int(np.prod(d[k + 1:]))
    f = list(np.ascontiguousarray(a).ravel())
    r = []
    for o in range(before):
        for i in range(after):
            if after == 1:
                r.append(blk(f, o * along, along))
            else:
                t = 0.0
                for j in range(along):
                    t += f[(o * along + j) * after + i]
                r.append(t)
    return np.array(r).reshape(d[:k] + d[k + 1:])


def var_axis(a, k):
    mu = sum_axis(a, k) / a.shape[k]
    x = np.ascontiguousarray(a) - np.expand_dims(mu, k)
    return sum_axis(x * x, k) / a.shape[k]


def same(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


def main(argv):
    rng = np.random.default_rng(20260927)

    def rnd(shape):
        return rng.standard_normal(shape) * np.exp(rng.uniform(-20, 20, shape))

    checks = {}

    def check(tag, ok):
        c = checks.setdefault(tag, [0, 0])
        c[1] += 1
        c[0] += 0 if ok else 1

    for n in list(range(0, 200)) + [1000, 8191, 8192, 8193, 9000, 16385, 20000, 50001]:
        a = rnd(n)
        check("flat: 0.0 + blocks of 8192, each pairwise", same(np.sum(a), blk(list(a), 0, n)))
        check("flat mean", same(np.mean(a), blk(list(a), 0, n) / n) if n else True)
    for n in (1, 7, 8, 9, 200):
        check("n entries of -0.0 sum to 0.0", same(np.sum(np.full(n, -0.0)), blk([-0.0] * n, 0, n)))
    shapes = [(3, 5), (4, 9), (2, 130), (5, 300), (3, 8193), (1, 50), (7, 1), (3, 4, 5), (2, 20, 3), (2, 20, 1),
              (4, 1, 9), (1, 30, 1), (3, 17, 2), (2, 3, 4, 5)]
    for shp in shapes:
        for _ in range(2):
            a = rnd(shp)
            for k in range(len(shp)):
                check("sum along an axis", same(np.sum(a, axis=k), sum_axis(a, k)))
                check("mean along an axis", same(np.mean(a, axis=k), sum_axis(a, k) / shp[k]))
                check("var along an axis", same(np.var(a, axis=k), var_axis(a, k)))
                check("std along an axis", same(np.std(a, axis=k), np.sqrt(var_axis(a, k))))
    views = [((6, 10), (slice(1, 5), slice(2, 7))), ((6, 10), (slice(None), slice(None, None, 2))),
             ((6, 9), (slice(None), slice(None, None, 2))), ((40, 50), (slice(1, -1), slice(1, -1))),
             ((3, 30000), (slice(None), slice(0, 20000))), ((4, 5, 6), (slice(None), 2, slice(1, 5))),
             ((20, 30, 40), (slice(2, 18), slice(3, 27), slice(1, 39)))]
    for shp, items in views:
        a = rnd(shp)
        s = a[items]
        c = s.copy()
        check("a slice sums as its copy (flat)", same(np.sum(s), blk(list(c.ravel()), 0, c.size)))
        check("a slice's mean and var are its copy's", same(np.mean(s), np.mean(c)) and same(np.var(s), np.var(c)))
        for k in range(s.ndim):
            check("a slice sums as its copy (along an axis)", same(np.sum(s, axis=k), sum_axis(c, k)))
    a = rnd(50000)
    check("a strided vector sums as its copy", same(np.sum(a[::3]), blk(list(a[::3]), 0, len(a[::3]))))
    for shp in [(5, 7), (30, 40), (3, 9000)]:
        A = rnd(shp)
        check("a transpose sums as its array (memory order)", same(np.sum(A.T), np.sum(A)))
        check("a transpose's axes are its array's, swapped", same(np.sum(A.T, axis=0), np.sum(A, axis=1))
              and same(np.sum(A.T, axis=1), np.sum(A, axis=0)))
        check("var and prod of a transpose are its array's", same(np.var(A.T), np.var(A)) and same(np.prod(A.T), np.prod(A)))
        check("a row-order copy of a transpose does not sum as the transpose",
              not same(np.sum(A.T.copy()), np.sum(A)) or A.size < 9)
    lines = ["probe_numpy_order.py: NumPy %s, Python %s" % (np.__version__, sys.version.split()[0]), ""]
    bad = 0
    for tag, (nbad, n) in checks.items():
        lines.append("%-78s %5d checks, %d differ" % (tag, n, nbad))
        bad += nbad
    lines.append("")
    lines.append("every check agrees bit for bit" if not bad else "%d checks differ" % bad)
    print("\n".join(lines))
    if "--write" in argv:
        with open(os.path.join(HERE, "results", "axis_order.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

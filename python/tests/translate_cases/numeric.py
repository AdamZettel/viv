# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Typical numeric Python, as written without Viv in mind: vp.translate's
test and benchmark cases (python/tests/test_translate.py, bench_translate.py)."""
import math

import numpy as np

ALPHA = 0.25
MOD = 1000003


def ewma(x, a):
    """An exponentially weighted moving average: a per-row recurrence."""
    y = np.zeros(len(x))
    s = x[0]
    for i in range(len(x)):
        s = a * x[i] + (1.0 - a) * s
        y[i] = s
    return y


def count_above(x, t):
    """How many entries exceed t: an int result."""
    n = 0
    for v in x:
        if v > t:
            n += 1
    return n


def _blend(s, v):
    return ALPHA * v + (1.0 - ALPHA) * s


def smooth(x):
    """A helper function and a module constant, translated with it."""
    y = np.zeros(len(x))
    s = 0.0
    for i in range(len(x)):
        s = _blend(s, x[i])
        y[i] = s
    return y


def group_cumsum(keys, vals):
    """A running sum that restarts at each new key: a groupby().cumsum() on
    rows sorted by key."""
    out = np.zeros(len(vals))
    acc = 0.0
    for i in range(len(vals)):
        if i > 0 and keys[i] != keys[i - 1]:
            acc = 0.0
        acc += vals[i]
        out[i] = acc
    return out


def group_mean(keys, vals):
    """Each row's group mean (groupby().transform("mean")) on sorted keys:
    two passes over each run of equal keys."""
    out = np.zeros(len(vals))
    n = len(vals)
    start = 0
    while start < n:
        end = start
        total = 0.0
        while end < n and keys[end] == keys[start]:
            total += vals[end]
            end += 1
        mean = total / (end - start)
        for j in range(start, end):
            out[j] = mean
        start = end
    return out


def cube_roots(c):
    """Newton's method per element, stopped when the step is below 1e-12
    of the iterate: root finding in a loop."""
    out = np.zeros(len(c))
    for i in range(len(c)):
        x = max(c[i], 1.0)
        for _ in range(100):
            step = (x * x * x - c[i]) / (3.0 * x * x)
            x = x - step
            if abs(step) <= 1e-12 * x:
                break
        out[i] = x
    return out


def bisect_sqrt(c: float, lo: float, hi: float) -> float:
    """Bisection for sqrt(c) on [lo, hi]: annotated, so translated at once."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * mid > c:
            hi = mid
        else:
            lo = mid
    return lo


def poly_hash(k):
    """A rolling hash of an int array, all in integers: exact."""
    h = 7
    for i in range(len(k)):
        h = (h * 31 + int(k[i])) % MOD
    return h


def norm_and_max(x):
    """Two results; math.sqrt is correctly rounded on both sides."""
    s = 0.0
    m = -math.inf
    for v in x:
        s += v * v
        if v > m:
            m = v
    return math.sqrt(s), m


# ------------------------------------------------------------- refused

def uses_dict_of_lists(x):
    d = {}
    for v in x:
        d.setdefault(int(v), []).append(v)
    return len(d)


def uses_try(x):
    try:
        return x[0] / x[1]
    except ZeroDivisionError:
        return 0.0


def uses_with(x):
    with np.errstate(divide="ignore"):
        return x[0] / x[1]

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""numpy_cases.py: the array vocabulary (lang/viv/src/interp/arrays.rs) against NumPy.

    python3 tools/arrays/numpy_cases.py          write the case files
    python3 tools/arrays/numpy_cases.py --check  exit 1 if the committed files differ

Writes, under lang/viv/tests/arrays/:

  numpy_exact.viv      one `print("<id>", fmt(<expr>))` per case
  numpy_exact.txt      what NumPy computes for each, as Viv's fmt writes it
                       (Python's repr of the value, a list for an array)
  numpy_tol.viv        the cases where NumPy's order of operations is its own
                       (BLAS, LAPACK, einsum's loops): each a `test` that holds
                       Viv's result to NumPy's within a tolerance

lang/viv/tests/arrays.rs runs both programs in the three engines: every
engine must print numpy_exact.txt byte for byte, and every test in
numpy_tol.viv must pass.

The exact cases are the ones whose result NumPy computes in an order Viv
follows: shapes, gathers, masks, constructors, sorting, `cumsum`, `prod`,
axis sums in NumPy's pairwise order (tools/py2viv/probe_numpy_order.py), and
elementwise arithmetic under broadcasting. `dot` and `norm` are exact against
a left-to-right sum in Python (the order arrays.rs documents) and within
1e-15 of NumPy's BLAS.

Inputs: entries whose magnitudes span 40 orders (so a different order of a
sum changes the last bits), with ties, signed zeros, NaN and infinities where
the operation defines them. The generator is seeded; the files are the same on
every run.
"""

import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "lang", "viv", "tests", "arrays")

R = random.Random(20260930)


def lit(x):
    """A float as a Viv literal (NAN and INF are variables of the program)."""
    x = float(x)
    if math.isnan(x):
        return "NAN_"
    if math.isinf(x):
        return "INF_" if x > 0 else "-INF_"
    r = repr(x)
    return r


def flat_lit(a):
    return "[" + ", ".join(lit(x) for x in np.asarray(a, dtype=float).ravel()) + "]"


# An exact case's long array is a variable, defined once before the first case
# that uses it (the program is one sequence of top-level statements).
PENDING = []
NAMED = {}
INLINE = [False]


def arr_lit(a):
    """An array as Viv source: a vector literal, or reshape of one."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1 and a.size > 0:
        text = flat_lit(a)
    else:
        dims = ", ".join(str(d) for d in a.shape)
        text = f"zeros({dims})" if a.size == 0 else f"reshape({flat_lit(a)}, {dims})"
    if INLINE[0] or len(text) <= 48:
        return text
    key = (a.shape, a.tobytes())
    if key not in NAMED:
        NAMED[key] = f"X{len(NAMED) + 1}"
        PENDING.append(f"{NAMED[key]} = {text}")
    return NAMED[key]


def rep(v):
    """What Viv's fmt writes for a NumPy result."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    a = np.asarray(v)
    if a.ndim == 0:
        return repr(float(a))
    return repr(np.asarray(a, dtype=float).tolist())


def value(scale=20):
    """A number whose magnitude is anywhere in 10^-scale .. 10^scale."""
    return R.choice([-1.0, 1.0]) * R.random() * 10.0 ** R.randint(-scale, scale)


def small_int():
    return float(R.randint(-4, 4))


def array(shape, kind="wide"):
    n = int(np.prod(shape)) if len(shape) else 1
    if kind == "wide":
        d = [value() for _ in range(n)]
    elif kind == "int":
        d = [small_int() for _ in range(n)]
    elif kind == "unit":
        d = [R.uniform(-1.0, 1.0) for _ in range(n)]
    elif kind == "special":
        pool = [0.0, -0.0, 1.0, -1.0, 2.5, float("nan"), float("inf"), float("-inf")]
        d = [R.choice(pool) for _ in range(n)]
    elif kind == "ties":
        d = [R.choice([0.0, -0.0, 1.0, 1.0, 2.0, -3.5]) for _ in range(n)]
    else:
        raise ValueError(kind)
    return np.array(d, dtype=float).reshape(shape)


def shape(rank, lo=1, hi=5):
    return tuple(R.randint(lo, hi) for _ in range(rank))


class Cases:
    def __init__(self):
        self.exact = []  # (id, expr, expected text)
        self.tol = []  # (id, setup lines, expr, expected array, rtol, atol)
        self.n = 0

    def add(self, expr, expected, setup=()):
        self.n += 1
        cid = f"c{self.n}"
        self.exact.append((cid, PENDING[:] + list(setup), expr, rep(expected)))
        PENDING.clear()

    def near(self, expr, expected, rtol, atol, setup=(), name=None):
        self.n += 1
        cid = name or f"t{self.n}"
        self.tol.append((cid, list(setup), expr, np.asarray(expected, dtype=float), rtol, atol))


C = Cases()


def ax(k, rank):
    """An axis as a Viv literal; sometimes NumPy's negative form."""
    if R.random() < 0.25:
        return str(float(k - rank))
    return str(float(k))


# ---- shapes -------------------------------------------------------------------

# (moving entries is exact whatever they are: small whole numbers, and signed
# zeros and NaN in some, keep the files short)
for _ in range(40):
    r = R.randint(1, 4)
    a = array(shape(r), R.choice(["int", "int", "special"]))
    n = a.size
    # a target shape with the same size
    for _ in range(3):
        rt = R.randint(1, 4)
        dims = []
        rem = n
        for i in range(rt - 1):
            divs = [d for d in range(1, rem + 1) if rem % d == 0]
            d = R.choice(divs)
            dims.append(d)
            rem //= d
        dims.append(rem)
        R.shuffle(dims)
        spec = list(dims)
        if R.random() < 0.4:
            spec[R.randrange(rt)] = -1
        C.add(f"reshape({arr_lit(a)}, {', '.join(str(float(d)) for d in spec)})", a.reshape(spec))
    C.add(f"{R.choice(['ravel', 'flatten'])}({arr_lit(a)})", a.ravel())
    C.add(f"transpose({arr_lit(a)})", a.T)
    perm = list(range(r))
    R.shuffle(perm)
    C.add(f"transpose({arr_lit(a)}, axes=[{', '.join(str(float(p)) for p in perm)}])", np.transpose(a, perm))
    i, j = R.randrange(r), R.randrange(r)
    C.add(f"swapaxes({arr_lit(a)}, {ax(i, r)}, {ax(j, r)})", np.swapaxes(a, i, j))

# ---- constructors ---------------------------------------------------------------

for n in range(0, 6):
    C.add(f"eye({float(n)})", np.eye(n))
    for m in range(0, 5):
        for k in (-3, -1, 0, 1, 2, 5):
            C.add(f"eye({float(n)}, {float(m)}, {float(k)})", np.eye(n, m, k))
for _ in range(12):
    v = array((R.randint(1, 5),))
    for k in (-2, -1, 0, 1, 3):
        C.add(f"diag({arr_lit(v)}, k={float(k)})", np.diag(v, k))
    C.add(f"diag({arr_lit(v)})", np.diag(v))
    A = array(shape(2))
    for k in (-3, -1, 0, 1, 2, 4):
        C.add(f"diag({arr_lit(A)}, k={float(k)})", np.diag(A, k))
    C.add(f"diag({arr_lit(A)})", np.diag(A))
for dims in [(3,), (2, 3), (1, 2, 3), (2, 1, 2, 2), (0,), (2, 0)]:
    C.add(f"ones({', '.join(str(float(d)) for d in dims)})", np.ones(dims))
    a = array(dims) if 0 not in dims else np.zeros(dims)
    C.add(f"zeros_like({arr_lit(a)})", np.zeros_like(a))
    C.add(f"ones_like({arr_lit(a)})", np.ones_like(a))

aranges = [(5,), (0,), (-3,), (2.5,), (1, 7), (7, 1), (-2.5, 3.1), (0, 1, 0.1), (1, 2, 0.1), (10, 0, -3), (0.5, -2.25, -0.25),
           (0, 10, 3), (1e-3, 1e-2, 1e-3), (-1, 1, 0.2), (3, 3.0000001, 1), (0, 1, 0.3), (1e10, 1e10 + 10, 0.7), (0.1, 0.9, 0.1)]
for args in aranges:
    fa = ", ".join(lit(x) for x in args)
    C.add(f"arange({fa})", np.arange(*[float(x) for x in args]))
for lo, hi in [(0.0, 1.0), (-2.5, 7.25), (1e-3, 1e5), (3.0, 3.0), (5.0, -5.0), (0.1, 0.7)]:
    for n in (0, 1, 2, 3, 5, 7, 50):
        for ep in (True, False):
            C.add(f"linspace({lit(lo)}, {lit(hi)}, {float(n)}, endpoint={'true' if ep else 'false'})", np.linspace(lo, hi, n, endpoint=ep))

# ---- products ---------------------------------------------------------------------

for _ in range(20):
    u, v = array((R.randint(1, 6),)), array((R.randint(1, 6),))
    C.add(f"outer({arr_lit(u)}, {arr_lit(v)})", np.outer(u, v))
    A = array(shape(2, 1, 4))
    B = array(shape(1, 1, 3))
    C.add(f"outer({arr_lit(A)}, {arr_lit(B)})", np.outer(A, B))
    w = array(u.shape)

    def seq_dot(x, y):
        t = 0.0
        for p, q in zip(x, y):
            t += p * q
        return t

    C.add(f"dot({arr_lit(u)}, {arr_lit(w)})", seq_dot(u, w))
    C.add(f"norm({arr_lit(A)})", math.sqrt(seq_dot(A.ravel(), A.ravel())))
    INLINE[0] = True
    C.near(f"dot({arr_lit(u)}, {arr_lit(w)})", np.dot(u, w), 1e-15, 0.0)
    INLINE[0] = False
for n in (1, 2, 3, 7, 8, 9, 16, 17, 40, 129, 150):
    if n <= 17:
        A = array((n, n))
        src = arr_lit(A)
    else:
        # a long diagonal (NumPy's pairwise sum splits past 128) on diag(v)
        v = array((n,))
        A = np.diag(v)
        src = f"diag({arr_lit(v)})"
    for k in (0, 1, -2):
        C.add(f"trace({src}, offset={float(k)})", np.trace(A, k))
    C.add(f"trace({src})", np.trace(A))
for _ in range(10):
    A = array(shape(2))
    for k in (-2, -1, 0, 1, 3):
        C.add(f"triu({arr_lit(A)}, k={float(k)})", np.triu(A, k))
        C.add(f"tril({arr_lit(A)}, k={float(k)})", np.tril(A, k))
    C.add(f"triu({arr_lit(A)})", np.triu(A))
    C.add(f"tril({arr_lit(A)})", np.tril(A))

# ---- reductions -----------------------------------------------------------------

red_shapes = [(3, 4), (2, 3, 4), (5, 1), (1, 7), (2, 2, 2, 3), (4, 9), (3, 17), (2, 130), (130, 2), (2, 1, 3), (1, 1, 140), (6, 2, 1)]
for shp in red_shapes:
    big = int(np.prod(shp)) > 100
    for kind in (("wide", "special") if big else ("wide", "wide", "int", "special", "ties")):
        a = array(shp, kind)
        r = len(shp)
        L = arr_lit(a)
        for k in range(r):
            for keep in (False, True):
                kv = "axis=" + ax(k, r)
                ks = ", keepdims=true" if keep else ""
                C.add(f"sum({L}, {kv}{ks})", np.sum(a, axis=k, keepdims=keep))
                C.add(f"mean({L}, {kv}{ks})", np.mean(a, axis=k, keepdims=keep))
                if kind != "special":
                    C.add(f"var({L}, {kv}{ks})", np.var(a, axis=k, keepdims=keep))
                    C.add(f"std({L}, {kv}{ks})", np.std(a, axis=k, keepdims=keep))
                C.add(f"prod({L}, {kv}{ks})", np.prod(a, axis=k, keepdims=keep))
                C.add(f"amax({L}, {kv}{ks})", np.amax(a, axis=k, keepdims=keep))
                C.add(f"amin({L}, {kv}{ks})", np.amin(a, axis=k, keepdims=keep))
                C.add(f"any({L}, {kv}{ks})", np.any(a, axis=k, keepdims=keep))
                C.add(f"all({L}, {kv}{ks})", np.all(a, axis=k, keepdims=keep))
            C.add(f"argmax({L}, axis={ax(k, r)})", np.argmax(a, axis=k))
            C.add(f"argmin({L}, axis={ax(k, r)})", np.argmin(a, axis=k))
        C.add(f"mean({L})", np.mean(a))
        if kind != "special":
            C.add(f"var({L})", np.var(a))
            C.add(f"std({L})", np.std(a))
        C.add(f"prod({L})", np.prod(a))
        C.add(f"amax({L})", np.amax(a))
        C.add(f"amin({L})", np.amin(a))
        C.add(f"argmax({L})", np.argmax(a))
        C.add(f"argmin({L})", np.argmin(a))
        C.add(f"count_nonzero({L})", float(np.count_nonzero(a)))
for n in (1, 5, 8, 15, 16, 17, 100, 128, 129, 257, 1000):
    v = array((n,))
    C.add(f"sum({arr_lit(v)}, axis=0.0, keepdims=true)", np.sum(v, axis=0, keepdims=True))
    C.add(f"mean({arr_lit(v)})", np.mean(v))
    C.add(f"var({arr_lit(v)})", np.var(v))

# long sums (NumPy's 8192-entry blocks), built from rand() so the source stays small
M = 2147483647.0
state = 1.0


def srand(seed):
    global state
    state = math.floor(abs(seed)) % 2147483646.0 + 1.0


def rand():
    global state
    state = (16807.0 * state) % M
    return state / M


long_setup = ["srand(77.0)"]
srand(77.0)
longs = []
for name, n in [("LA", 8191), ("LB", 8192), ("LC", 8193), ("LD", 20000), ("LE", 16385)]:
    base = np.array([rand() for _ in range(n)])
    w = (base - 0.5) * (base * 1e8) * (base - 0.25)
    # the same arithmetic in Viv as in Python: draw, then the same products
    long_setup.append(f"{name}b = rand({float(n)})")
    long_setup.append(f"{name} = ({name}b - 0.5) * ({name}b * 1e8) * ({name}b - 0.25)")
    longs.append((name, n, w))
for i, (name, n, w) in enumerate(longs):
    C.add(f"sum({name}, axis=0.0, keepdims=true)", np.sum(w, axis=0, keepdims=True), setup=long_setup if i == 0 else ())
    C.add(f"mean({name})", np.mean(w))
    C.add(f"var({name})", np.var(w))
    C.add(f"std({name})", np.std(w))
    if n % 2 == 0:
        W = w.reshape(2, n // 2)
        C.add(f"sum(reshape({name}, 2.0, {float(n // 2)}), axis=1.0)", np.sum(W, axis=1))
        C.add(f"sum(reshape({name}, {float(n // 2)}, 2.0), axis=0.0)", np.sum(w.reshape(n // 2, 2), axis=0))
        C.add(f"mean(reshape({name}, 2.0, {float(n // 2)}), axis=1.0)", np.mean(W, axis=1))

# ---- order, gathers, masks ----------------------------------------------------------

for _ in range(30):
    kind = R.choice(["ties", "special", "wide", "int"])
    shp = R.choice([(R.randint(0, 12),), (R.randint(1, 4), R.randint(1, 9)), (2, 2, R.randint(1, 6))])
    a = array(shp, kind) if 0 not in shp else np.zeros(shp)
    C.add(f"argsort({arr_lit(a)})", np.argsort(a, kind="stable"))
for _ in range(30):
    r = R.randint(1, 3)
    a = array(shape(r), R.choice(["int", "special"]))
    k = R.randrange(r)
    ir = R.randint(1, 2) if r < 4 else 1
    if r - 1 + ir > 4:
        ir = 1
    idx = np.array([float(R.randrange(a.shape[k])) for _ in range(int(np.prod(shape(ir, 1, 3))))])
    ishape = shape(ir, 1, 3)
    idx = np.array([float(R.randrange(a.shape[k])) for _ in range(int(np.prod(ishape)))]).reshape(ishape)
    C.add(f"take({arr_lit(a)}, {arr_lit(idx)}, axis={ax(k, r)})", np.take(a, idx.astype(int), axis=k))
for _ in range(20):
    shp = shape(R.randint(1, 3))
    a = array(shp, "int")
    b = array(shp, R.choice(["int", "special"]))
    m = a > 0
    C.add(f"extract({arr_lit(a)} > 0.0, {arr_lit(b)})", np.extract(m, b))
    c = b.copy()
    np.putmask(c, m, 7.5)
    C.add("P", c, setup=[f"P = {arr_lit(b)}", f"putmask(P, {arr_lit(a)} > 0.0, 7.5)"])
    c2 = b.copy()
    vals = array(shp)
    np.putmask(c2, m, vals)
    C.add("P", c2, setup=[f"P = {arr_lit(b)}", f"putmask(P, {arr_lit(a)} > 0.0, {arr_lit(vals)})"])
    C.add(f"flatnonzero({arr_lit(a)})", np.flatnonzero(a))

# ---- scans ---------------------------------------------------------------------

for _ in range(25):
    r = R.randint(1, 3)
    kind = R.choice(["wide", "int", "special", "unit"])
    a = array(shape(r), kind)
    L = arr_lit(a)
    C.add(f"cumsum({L})", np.cumsum(a))
    C.add(f"cumprod({L})", np.cumprod(a))
    k = R.randrange(r)
    C.add(f"cumsum({L}, axis={ax(k, r)})", np.cumsum(a, axis=k))
    C.add(f"cumprod({L}, axis={ax(k, r)})", np.cumprod(a, axis=k))
    C.add(f"diff({L})", np.diff(a))
    n = R.randint(0, 4)
    C.add(f"diff({L}, n={float(n)})", np.diff(a, n))
    C.add(f"diff({L}, n={float(n)}, axis={ax(k, r)})", np.diff(a, n, axis=k))

# ---- clip, broadcasting -------------------------------------------------------------

for _ in range(20):
    a = array(shape(R.randint(1, 3)), R.choice(["special", "unit", "wide"]))
    lo, hi = sorted([R.choice([-1.0, -0.0, 0.0, 0.5, float("nan")]), R.choice([0.0, 1.0, 2.0])], key=lambda x: (math.isnan(x), x))
    C.add(f"clip({arr_lit(a)}, {lit(lo)}, {lit(hi)})", np.clip(a, lo, hi))
for x in (-2.0, -0.0, 0.0, 0.3, 5.0, float("nan"), float("inf")):
    C.add(f"clip({lit(x)}, 0.0, 1.0)", np.clip(x, 0.0, 1.0))

pairs = [((3, 4), (4,)), ((3, 4), (1, 4)), ((3, 4), (3, 1)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 4)), ((2, 3, 4), (3, 1)),
         ((2, 1, 4), (3, 1)), ((4,), (2, 3, 4)), ((1,), (5,)), ((2, 2, 1, 3), (2, 1, 3)), ((1, 1), (2, 3, 1, 1)), ((5, 1), (5,))]
ops = [("+", np.add), ("-", np.subtract), ("*", np.multiply), ("/", np.divide), ("<", np.less), ("<=", np.less_equal),
       (">", np.greater), (">=", np.greater_equal), ("==", np.equal), ("!=", np.not_equal)]
for sa, sb in pairs:
    for kind in ("wide", "int"):
        a, b = array(sa, kind), array(sb, kind)
        for sym, f in ops:
            with np.errstate(all="ignore"):
                C.add(f"{arr_lit(a)} {sym} {arr_lit(b)}", f(a, b).astype(float))
                C.add(f"{arr_lit(b)} {sym} {arr_lit(a)}", f(b, a).astype(float))
        C.add(f"max({arr_lit(a)}, {arr_lit(b)})", np.fmax(a, b))
        C.add(f"min({arr_lit(b)}, {arr_lit(a)})", np.fmin(b, a))

# ---- random draws ------------------------------------------------------------------

srand(12345.0)
draws = np.array([rand() for _ in range(24)])
C.add("rand(2.0, 3.0, 4.0)", draws.reshape(2, 3, 4), setup=["srand(12345.0)"])
C.add("rand(5.0)", np.array([rand() for _ in range(5)]))

# ---- the cases held to a tolerance ------------------------------------------------------

INLINE[0] = True

for spec, shapes in [("ij,jk->ik", [(3, 4), (4, 2)]), ("ij,ij->", [(3, 4), (3, 4)]), ("ij,jk", [(2, 5), (5, 3)]),
                     ("i,i", [(6,), (6,)]), ("ii->i", [(4, 4)]), ("ii", [(5, 5)]), ("ij->ji", [(3, 5)]),
                     ("ijkl,kl->ij", [(2, 3, 4, 5), (4, 5)]), ("ijkl,jl->ik", [(3, 3, 3, 3), (3, 3)]),
                     ("pq,pqrs,rs->", [(3, 3), (3, 3, 3, 3), (3, 3)]), ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),
                     ("bij,bjk->bik", [(2, 3, 4), (2, 4, 2)]), ("i,j->ij", [(3,), (4,)]), ("ijk->kji", [(2, 3, 4)]),
                     ("ij,kj->ik", [(3, 4), (5, 4)])]:
    ops_ = [array(s, "unit") for s in shapes]
    got = np.einsum(spec, *ops_)
    C.near(f'einsum("{spec}", {", ".join(arr_lit(o) for o in ops_)})', got, 1e-13, 1e-15)

for n in (1, 2, 3, 5, 8):
    A = array((n, n), "unit") + n * np.eye(n)
    S = A @ A.T + n * np.eye(n)
    b = array((n,), "unit")
    B = array((n, 3), "unit")
    C.near(f"lin_solve({arr_lit(A)}, {arr_lit(b)})", np.linalg.solve(A, b), 1e-12, 1e-13)
    C.near(f"lin_solve({arr_lit(A)}, {arr_lit(B)})", np.linalg.solve(A, B), 1e-12, 1e-13)
    C.near(f"inv({arr_lit(A)})", np.linalg.inv(A), 1e-12, 1e-13)
    C.near(f"cholesky({arr_lit(S)})", np.linalg.cholesky(S), 1e-12, 1e-13)
    w, V = np.linalg.eigh(S)
    C.near("EV_VALS", w, 1e-12, 1e-12, setup=[f"EV_VALS, EV_VECS = eig_sym({arr_lit(S)})"])
    # eigenvectors up to sign: compare |V|
    C.near("abs(EV_VECS)", np.abs(V), 1e-10, 1e-12, setup=[f"EV_VALS, EV_VECS = eig_sym({arr_lit(S)})"])
for shp in [(3, 3), (4, 2), (2, 4), (5, 3), (1, 4), (4, 1)]:
    A = array(shp, "unit")
    s = np.linalg.svd(A, compute_uv=False)
    C.near("SV_S", s, 1e-13, 1e-14, setup=[f"SV_U, SV_S, SV_VT = svd({arr_lit(A)})"])
    # A = U diag(s) Vt, and U, Vt with orthonormal columns and rows
    C.near("SV_U @ (diag(SV_S) @ SV_VT)", A, 1e-13, 1e-14, setup=[f"SV_U, SV_S, SV_VT = svd({arr_lit(A)})"])
    k = min(shp)
    C.near("transpose(SV_U) @ SV_U", np.eye(k), 1e-13, 1e-14, setup=[f"SV_U, SV_S, SV_VT = svd({arr_lit(A)})"])
    C.near("SV_VT @ transpose(SV_VT)", np.eye(k), 1e-13, 1e-14, setup=[f"SV_U, SV_S, SV_VT = svd({arr_lit(A)})"])


def exact_program():
    out = ["# Generated by tools/arrays/numpy_cases.py: do not edit.", "NAN_ = 0.0 / 0.0", "INF_ = 1.0 / 0.0"]
    for cid, setup, expr, _ in C.exact:
        out.extend(setup)
        out.append(f'print("{cid}", fmt({expr}))')
    return "\n".join(out) + "\n"


def exact_text():
    return "".join(f"{cid} {want}\n" for cid, _, _, want in C.exact)


def tol_program():
    out = ["# Generated by tools/arrays/numpy_cases.py: do not edit.",
           "# Each test holds a result to NumPy's within |viv - numpy| <= atol + rtol |numpy|."]
    for cid, setup, expr, want, rtol, atol in C.tol:
        out.append(f'test "{cid}" {{')
        for s in setup:
            out.append("    " + s)
        # the names of a test are the program's: each test has its own
        if want.ndim == 0:
            out.append(f"    got_{cid} = {expr}")
            out.append(f"    assert_near(got_{cid}, {lit(want)}, {lit(atol + rtol * abs(float(want)))})")
        else:
            out.append(f"    got_{cid} = ravel({expr})")
            w = want.ravel()
            out.append(f"    want_{cid} = {flat_lit(w)}")
            tol = atol + rtol * float(np.max(np.abs(w))) if w.size else atol
            out.append(f"    assert_allclose(got_{cid}, want_{cid}, {lit(tol)})")
        out.append("}")
    return "\n".join(out) + "\n"


def main():
    files = {
        "numpy_exact.viv": exact_program(),
        "numpy_exact.txt": exact_text(),
        "numpy_tol.viv": tol_program(),
    }
    if "--check" in sys.argv:
        bad = 0
        for name, text in files.items():
            with open(os.path.join(OUT, name)) as f:
                if f.read() != text:
                    print(f"{name} differs from what the generator writes")
                    bad = 1
        sys.exit(bad)
    os.makedirs(OUT, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text)
    print(f"{len(C.exact)} exact cases, {len(C.tol)} held to a tolerance, numpy {np.__version__}")


if __name__ == "__main__":
    main()

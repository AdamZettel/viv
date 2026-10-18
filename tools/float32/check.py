# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Decide the hard cases of exhaustive.c with mpmath.

    python3 check.py exp exp.hard          the inputs exhaustive.c wrote
    python3 check.py --bound log log.sample  the double function's error on
                                           exhaustive.c's sampled inputs

For each input the reference is the float32 nearest f(x) computed by mpmath
at 200 bits, ties to even, decided by exact comparison (never through a
double). It prints each input where Viv's Float32 result, the double result
rounded once, differs from the reference: those are the exceptions the
float32 function must fix. `--bound` measures the double function's error
in double ulps on random float32 inputs, the assumption exhaustive.c's K
rests on.
"""
import struct
import sys

import mpmath
import numpy as np

mpmath.mp.prec = 200
F = {"exp": mpmath.exp, "log": mpmath.log, "sin": mpmath.sin, "cos": mpmath.cos, "tanh": mpmath.tanh}


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def nearest32(v):
    """The float32 nearest the mpf v, ties to even."""
    c = np.float32(float(v))
    best = None
    for cand in (np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))):
        if not np.isfinite(cand):
            continue
        d = abs(mpmath.mpf(float(cand)) - v)
        key = (d, int(cand.view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


def hard(name, path):
    bad = 0
    n = 0
    for line in open(path):
        xb, yb = line.split()
        x = f32(int(xb, 16))
        y = f64(int(yb, 16))
        n += 1
        want = nearest32(F[name](mpmath.mpf(x)))
        got = np.float32(y)
        if got.view(np.uint32) != want.view(np.uint32) and not (got == 0 and want == 0 and np.signbit(got) == np.signbit(want)):
            bad += 1
            print(f"{name} {xb} {repr(x)}: rounded once {repr(got)}, correctly rounded {repr(want)} ({int(want.view(np.uint32)):08x})")
    print(f"{name}: {n} hard inputs checked, {bad} differ from the correctly rounded float32")


def bound(name, path):
    """The largest error of the double function on the sampled inputs, in
    ulps of the result (finite, nonzero results only)."""
    worst = mpmath.mpf(0)
    n = 0
    for line in open(path):
        xb, yb = line.split()
        x = f32(int(xb, 16))
        y = f64(int(yb, 16))
        if not np.isfinite(y) or y == 0:
            continue
        v = F[name](mpmath.mpf(x))
        u = np.spacing(abs(y))
        err = abs(mpmath.mpf(y) - v) / mpmath.mpf(float(u))
        n += 1
        if err > worst:
            worst = err
    print(f"{name}: {n} sampled inputs, largest error {mpmath.nstr(worst, 4)} ulp of the double result")


if __name__ == "__main__":
    if sys.argv[1] == "--bound":
        bound(sys.argv[2], sys.argv[3])
    else:
        hard(sys.argv[1], sys.argv[2])

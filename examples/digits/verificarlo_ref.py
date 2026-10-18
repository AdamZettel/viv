# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Verificarlo's Monte Carlo arithmetic, done exactly, beside `viv digits`.

Verificarlo (Denis, de Oliveira Castro and Petit, "Verificarlo: checking
floating point accuracy through Monte Carlo Arithmetic", ARITH 2016) replaces
each floating-point operation a op b of a C program by

    round(inexact(inexact(a) op inexact(b))),  inexact(x) = x + 2^(e_x - 52) * xi,

xi uniform in (-1/2, 1/2), e_x the exponent of x (x = 1.m * 2^e_x), the
operation done in higher precision (mode `mca`; `rr` drops the inner
noise, `pb` the outer). Over N runs the significant digits of a result are
s = -log10(sigma / |mu|).

This script runs the toy programs of examples/digits written as the C
program would be (indices are ints), with every operation in exact rational
arithmetic and one rounding to double, and prints its digit counts beside
the ones `viv digits --mca mca` measures for the .viv twins. Run it from the
repository root: python3 examples/digits/verificarlo_ref.py
"""
import json
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SAMPLES = 32
ROOT = Path(__file__).resolve().parents[2]
VIV = ROOT / "target" / "release" / "viv"


class Mca:
    def __init__(self, seed, mode="mca"):
        self.rng = random.Random(seed)
        self.inner = mode in ("mca", "pb")
        self.outer = mode in ("mca", "rr")

    def inexact(self, x):
        if x == 0:
            return Fraction(0)
        e = math.frexp(float(x))[1] - 1
        return Fraction(x) + Fraction(2) ** (e - 52) * Fraction(self.rng.random() - 0.5)

    def op(self, f, a, b):
        if self.inner:
            a, b = self.inexact(a), self.inexact(b)
        r = f(Fraction(a), Fraction(b))
        if self.outer:
            r = self.inexact(r)
        return float(r)

    def add(self, a, b):
        return self.op(lambda x, y: x + y, a, b)

    def sub(self, a, b):
        return self.op(lambda x, y: x - y, a, b)

    def mul(self, a, b):
        return self.op(lambda x, y: x * y, a, b)

    def div(self, a, b):
        return self.op(lambda x, y: x / y, a, b)


def cancel(m):
    out = []
    for k in range(4):
        x = 10.0 ** (-2.0 - 2.0 * k)   # pow: a library call, not instrumented
        out.append(x)
        out.append(m.sub(1.0, math.cos(x)))
        s = math.sin(m.div(x, 2.0))
        out.append(m.mul(m.mul(2.0, s), s))
    return out


def kahan(m, n=1000):
    naive = 0.0
    for _ in range(n):
        naive = m.add(naive, 0.1)
    total, carry = 0.0, 0.0
    for _ in range(n):
        y = m.sub(0.1, carry)
        t = m.add(total, y)
        carry = m.sub(m.sub(t, total), y)
        total = t
    return [naive, total]


def hilbert(m, n=6):
    A = [[m.div(1.0, float(i + j + 1)) for j in range(n)] for i in range(n)]
    b = []
    for i in range(n):      # sum(H[i, :]): fewer than 8 terms, left to right
        s = 0.0
        for j in range(n):
            s = m.add(s, A[i][j])
        b.append(s)
    for c in range(n):
        p = max(range(c, n), key=lambda r: (abs(A[r][c]), -r))
        A[c], A[p] = A[p], A[c]
        b[c], b[p] = b[p], b[c]
        for r in range(c + 1, n):
            f = m.div(A[r][c], A[c][c])
            for j in range(c, n):
                A[r][j] = m.sub(A[r][j], m.mul(f, A[c][j]))
            b[r] = m.sub(b[r], m.mul(f, b[c]))
    x = [0.0] * n
    for i in reversed(range(n)):
        s = b[i]
        for j in range(i + 1, n):
            s = m.sub(s, m.mul(A[i][j], x[j]))
        x[i] = m.div(s, A[i][i])
    return x


def digits(xs):
    if all(x == xs[0] for x in xs):
        return 15.95
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
    if sd == 0:
        return 15.95
    if mu == 0:
        return 0.0
    return max(0.0, min(15.95, -math.log10(sd / abs(mu))))


def viv_digits(prog, sets=()):
    args = [str(VIV), "digits", "--no-order", "--no-fma", "--attrib", "0", "--samples", str(SAMPLES), "--json"]
    for s in sets:
        args += ["--set", s]
    r = subprocess.run(args + [str(ROOT / "examples" / "digits" / prog)], capture_output=True, text=True, check=True)
    return [o["mca"] for o in json.loads(r.stdout)["outputs"]]


def main():
    cases = [
        ("cancel.viv", (), cancel, ["x=1e-2", "1-cos", "2sin^2", "x=1e-4", "1-cos", "2sin^2",
                                    "x=1e-6", "1-cos", "2sin^2", "x=1e-8", "1-cos", "2sin^2"]),
        ("kahan.viv", ("n=1000",), kahan, ["loop", "kahan"]),
        ("hilbert.viv", (), hilbert, ["x[%d]" % i for i in range(6)]),
    ]
    print("%-12s %-8s %10s %10s %6s" % ("program", "output", "reference", "viv", "diff"))
    worst = 0.0
    for prog, sets, fn, names in cases:
        runs = [fn(Mca(seed)) for seed in range(SAMPLES)]
        ref = [digits([r[j] for r in runs]) for j in range(len(names))]
        got = viv_digits(prog, sets)[: len(names)]
        for name, a, b in zip(names, ref, got):
            worst = max(worst, abs(a - b))
            print("%-12s %-8s %10.2f %10.2f %6.2f" % (prog, name, a, b, b - a))
    print("largest difference: %.2f digits" % worst)
    return 0 if worst < 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())

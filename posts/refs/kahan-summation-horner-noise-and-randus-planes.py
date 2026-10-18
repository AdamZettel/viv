# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-kahan-summation-horner-noise-and-randus-planes.viv.

1. Sum of 1/k (each term the double nearest 1/k) for k = 1..N, naive
   left-to-right and Kahan-compensated, against math.fsum (correctly rounded);
   fsum itself is checked against exact rational arithmetic for N <= 10^4.
2. (x - 1)^7 expanded, x^7 - 7x^6 + ... - 1, evaluated by Horner at 401 points
   of [0.98, 1.02], against the exact value in rationals; Higham's bound
   gamma_14 * sum |c_i| |x|^i with gamma_n = n u / (1 - n u), u = 2^-53; the factored form (x - 1)^7, by repeated products.
3. RANDU, x' = 65539 x mod 2^31 from x = 1: 9 x_k - 6 x_{k+1} + x_{k+2} is a
   multiple of 2^31; the values of (9 x_k - 6 x_{k+1} + x_{k+2}) / 2^31.
   MINSTD, x' = 16807 x mod (2^31 - 1), for comparison.
Run: python3 posts/refs/kahan-summation-horner-noise-and-randus-planes.py > posts/refs/kahan-summation-horner-noise-and-randus-planes.out
"""
import math
from fractions import Fraction as F

EPS = 2.0 ** -52
U = EPS / 2  # Higham's unit roundoff, 2^-53

print("1. sum of 1/k")
checkpoints = sorted({round(10 ** (k / 5)) for k in range(31)})
naive = kahan = comp = 0.0
exact_running = F(0)
terms = []
ci = 0
for k in range(1, checkpoints[-1] + 1):
    v = 1.0 / k
    terms.append(v)
    naive = naive + v
    y = v - comp
    t = kahan + y
    comp = (t - kahan) - y
    kahan = t
    if k <= 10**4:
        exact_running += F(v)
    if k == checkpoints[ci]:
        ex = math.fsum(terms)
        if k <= 10**4:
            assert float(exact_running) == ex
        ulp = math.ulp(ex)
        print(f"  N = {k:>7d}: fsum {ex!r:<22} naive {naive!r:<22} ({(naive - ex) / ulp:+.0f} ulp)  Kahan {kahan!r} ({(kahan - ex) / ulp:+.0f} ulp)")
        ci += 1
print(f"  fsum equals the exact rational sum, rounded, at every N <= 10^4 (asserted)")

print("2. (x - 1)^7 near x = 1")
C = [1, -7, 21, -35, 35, -21, 7, -1]


def horner(x):
    s = 1.0
    for c in C[1:]:
        s = s * x + c
    return s


def factored(x):
    d = x - 1.0
    p = d
    for _ in range(6):
        p = p * d
    return p


NP = 401
signs = 0
prev = None
worst_ratio = 0.0
worst_fact = 0.0
worst_err = 0.0
g14 = 14 * U / (1 - 14 * U)
for i in range(NP):
    x = 0.98 + (1.02 - 0.98) * i / (NP - 1)
    h = horner(x)
    ex = (F(x) - 1) ** 7
    bound = g14 * sum(abs(c) * abs(x) ** (7 - j) for j, c in enumerate(C))
    worst_ratio = max(worst_ratio, abs(F(h) - ex) / F(bound))
    worst_err = max(worst_err, float(abs(F(h) - ex)))
    if ex != 0:
        worst_fact = max(worst_fact, float(abs(F(factored(x)) - ex) / abs(ex)))
    if prev is not None and (h > 0) != (prev > 0):
        signs += 1
    prev = h
print(f"  401 points in [0.98, 1.02]: the Horner values change sign {signs} times; the exact polynomial once")
print(f"  largest |Horner - exact| {worst_err:.4e}; largest ratio to gamma_14 sum|c_i||x|^i: {float(worst_ratio):.4f}")
print(f"  largest relative error of the factored form: {worst_fact / EPS:.3f} eps")
x = 1.005
print(f"  x = 1.005: Horner {horner(x)!r}, factored {factored(x)!r}, exact {float((F(x) - 1) ** 7)!r}")
x = 1.2
print(f"  x = 1.2: Horner {horner(x)!r}, exact {float((F(x) - 1) ** 7)!r}, relative error {float(abs(F(horner(x)) - (F(x) - 1) ** 7) / (F(x) - 1) ** 7):.2e}")

print("3. RANDU")
M31 = 2**31
x = 1
xs = []
for _ in range(3002):
    x = (65539 * x) % M31
    xs.append(x)
vals = {}
for k in range(3000):
    c = 9 * xs[k] - 6 * xs[k + 1] + xs[k + 2]
    assert c % M31 == 0
    vals[c // M31] = vals.get(c // M31, 0) + 1
print(f"  3000 triples: 9 x_k - 6 x_{{k+1}} + x_{{k+2}} is a multiple of 2^31 every time; the multiples: {sorted(vals)} ({len(vals)} values)")
print(f"  counts: {[vals[v] for v in sorted(vals)]}")
print(f"  first draws: {xs[:5]}")
m = 2**31 - 1
x = 1
ys = []
for _ in range(3002):
    x = (16807 * x) % m
    ys.append(x)
whole = sum(1 for k in range(3000) if (9 * ys[k] - 6 * ys[k + 1] + ys[k + 2]) % m == 0)
print(f"  MINSTD: 9 y_k - 6 y_{{k+1}} + y_{{k+2}} a multiple of m in {whole} of 3000 triples; first draws {ys[:3]}")

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The constants of Viv's correctly rounded exp (vmath/mod.rs `exp`, its C twin
`viv_exp_cr` in vmath_tail.c), computed with Python's decimal module at 80
digits. Prints them as Rust (`--rust`) or C (`--c`); both twins paste the
same bits. `--cases` prints 4,000 inputs with their correctly rounded results
(lang/viv/tests/vmath_exp_cases.txt).

  T[j] = Th + Tl ~ 2^(j/128), j = 0..127   (Th the nearest double, Tl the rest)
  L1 + L2 + L3 ~ ln 2 / 128                (L1 with 36 significant bits, so
                                            k * L1 is exact for |k| < 2^17)
  INVL ~ 128 / ln 2
  C[n] = Ch + Cl ~ 1 / n!, n = 0..12        (the slow path's Taylor series)
"""
import struct
import sys
from decimal import Decimal, getcontext

getcontext().prec = 80
N = 128


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def split(d):
    hi = float(d)
    lo = float(d - Decimal(hi))
    return hi, lo


ln2 = Decimal(2).ln()
L = ln2 / N
l1 = struct.unpack('<d', struct.pack('<Q', bits(float(L)) & ~((1 << 17) - 1)))[0]
l2 = float(L - Decimal(l1))
l3 = float(L - Decimal(l1) - Decimal(l2))
invl = float(N / ln2)
T = [split((Decimal(j) * L).exp()) for j in range(N)]
C = []
f = Decimal(1)
for n in range(13):
    if n:
        f *= n
    C.append(split(1 / f))

if '--cases' in sys.argv:
    # inputs and the double nearest e^x, as bit patterns: the test
    # lang/viv/tests/vmath.rs `exp_is_correctly_rounded` reads them
    import random
    random.seed(20261006)
    xs = [1.0, -1.0, 0.5, 2.0, 10.0, -10.0, 700.0, -707.9, 708.9, 1e-10, 0.6931471805599453]
    while len(xs) < 4000:
        u = random.random()
        if u < 0.4:
            xs.append(random.uniform(-708, 709))
        elif u < 0.7:
            xs.append(random.uniform(-2, 2))
        elif u < 0.9:
            xs.append(random.uniform(-1e-3, 1e-3))
        else:
            xs.append(random.choice([-1, 1]) * 10 ** random.uniform(-17, -3))
    for x in xs:
        print(f'{bits(x):016x} {bits(float(Decimal(x).exp())):016x}')
    sys.exit(0)

rust = '--rust' in sys.argv
def h(x):
    return f'0x{bits(x):016x}'
if rust:
    print(f'const EXP_L1: u64 = {h(l1)};\nconst EXP_L2: u64 = {h(l2)};\nconst EXP_L3: u64 = {h(l3)};\nconst EXP_INVL: u64 = {h(invl)};')
    print('#[rustfmt::skip]\nconst EXP_T: [u64; 256] = [')
    for j in range(0, N, 2):
        print('    ' + ' '.join(f'{h(a)}, {h(b)},' for a, b in T[j:j + 2]))
    print('];\n#[rustfmt::skip]\nconst EXP_C: [u64; 26] = [')
    for n in range(0, 13, 2):
        print('    ' + ' '.join(f'{h(a)}, {h(b)},' for a, b in C[n:n + 2]))
    print('];')
else:
    print(f'#define VIV_EXP_L1 {h(l1)}ULL\n#define VIV_EXP_L2 {h(l2)}ULL\n#define VIV_EXP_L3 {h(l3)}ULL\n#define VIV_EXP_INVL {h(invl)}ULL')
    print('static const uint64_t viv_exp_T[256] = {')
    for j in range(0, N, 2):
        print('\t' + ' '.join(f'{h(a)}ULL, {h(b)}ULL,' for a, b in T[j:j + 2]))
    print('};\nstatic const uint64_t viv_exp_C[26] = {')
    for n in range(0, 13, 2):
        print('\t' + ' '.join(f'{h(a)}ULL, {h(b)}ULL,' for a, b in C[n:n + 2]))
    print('};')

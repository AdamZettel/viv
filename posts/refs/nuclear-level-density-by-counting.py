# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-nuclear-level-density-by-counting.viv.

Three independent computations, none of them the post's recursion:
  1. p(n), n <= 100, by Euler's pentagonal-number recurrence, in exact
     integers;
  2. a brute-force count of fermion states (every way to put N = 12
     particles in 30 equally spaced levels, by itertools.combinations),
     showing that the count at excitation energy E is p(E) for E <= N;
  3. the Hardy-Ramanujan asymptotic exp(pi sqrt(2E/3)) / (4 sqrt3 E), its
     ratio to p(E) at E = 100, and how (1 - ratio) sqrt(E) behaves at larger
     E (exact integers up to E = 6400).

    python3 posts/refs/nuclear-level-density-by-counting.py > posts/refs/nuclear-level-density-by-counting.out

Output recorded 2026-09-25 (Python 3, exact integer arithmetic; mpmath 1.3.0
for the logs of big integers) is in
posts/refs/nuclear-level-density-by-counting.out.
"""
import itertools
import math
from collections import Counter

import mpmath as mp

mp.mp.dps = 30


def partitions(nmax):
    p = [1] + [0] * nmax
    for n in range(1, nmax + 1):
        s, k = 0, 1
        while True:
            g1 = k * (3 * k - 1) // 2
            g2 = k * (3 * k + 1) // 2
            if g1 > n:
                break
            sign = 1 if k % 2 else -1
            s += sign * p[n - g1]
            if g2 <= n:
                s += sign * p[n - g2]
            k += 1
        p[n] = s
    return p


p = partitions(6400)
print("p(0..100) =", p[:101])

# brute force: N fermions, one per level, levels 0..M-1
N, M = 12, 30
ground = N * (N - 1) // 2
count = Counter(sum(c) - ground for c in itertools.combinations(range(M), N))
brute = [count[e] for e in range(N + 1)]
print(f"brute-force counts, N = {N}, E = 0..{N}:", brute)
print("equal to p(E):", brute == p[:N + 1])


def hardy_ramanujan(n):
    return mp.exp(mp.pi * mp.sqrt(mp.mpf(2) * n / 3)) / (4 * n * mp.sqrt(3))


# the next term of the asymptotic series, from expanding the first term of
# Rademacher's convergent series for p(n) in 1/sqrt(n):
#   p(n) = HR(n) (1 - c / sqrt(n) + O(1/n)),  c = sqrt(6) / (2 pi) + pi / (24 sqrt 6)
c_next = mp.sqrt(6) / (2 * mp.pi) + mp.pi / (24 * mp.sqrt(6))
print("c = sqrt(6)/(2 pi) + pi/(24 sqrt 6) =", mp.nstr(c_next, 17))

for n in [25, 100, 400, 1600, 6400]:
    r = mp.mpf(p[n]) / hardy_ramanujan(n)
    print(f"E = {n:5d}: p/HR = {mp.nstr(r, 17)}, (1 - p/HR) sqrt(E) = {mp.nstr((1 - r) * mp.sqrt(n), 6)}, "
          f"p/HR - (1 - c/sqrt(E)) = {mp.nstr(r - 1 + c_next / mp.sqrt(n), 4)}")

# the log-slope d ln rho / d sqrt(E) between E = 81 and E = 100, and what the
# asymptotic form predicts for it at the midpoint sqrt(E) = 9.5:
# d/d(sqrt E) [pi sqrt(2/3) sqrt(E) - ln(4 sqrt3 E)] = pi sqrt(2/3) - 2 / sqrt(E)
slope = (mp.log(p[100]) - mp.log(p[81])) / (10 - 9)
print("measured slope, E = 81 to 100:", mp.nstr(slope, 17))
print("2 sqrt(a) = pi sqrt(2/3):", mp.nstr(mp.pi * mp.sqrt(mp.mpf(2) / 3), 17))
print("pi sqrt(2/3) - 2/9.5:", mp.nstr(mp.pi * mp.sqrt(mp.mpf(2) / 3) - 2 / mp.mpf("9.5"), 17))

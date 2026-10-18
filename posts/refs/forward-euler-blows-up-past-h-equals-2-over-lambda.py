# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-forward-euler-blows-up-past-h-equals-2-over-lambda.viv.

y' = -lam (y - cos t) - sin t, lam = 1000, whose solution from y(0) = 1 is
y = cos t. Forward and backward Euler (the implicit equation is linear, so each
backward step is solved in closed form). The per-step amplification is
measured from the difference of two runs, one from y(0) = 2 and one from
y(0) = 1, over the first four steps, and compared with |1 - h lam| and
1/(1 + h lam).
Run: python3 posts/refs/forward-euler-blows-up-past-h-equals-2-over-lambda.py > posts/refs/forward-euler-blows-up-past-h-equals-2-over-lambda.out
"""
import math

LAM = 1000.0
T = 2.0


def f(t, y):
    return -LAM * (y - math.cos(t)) - math.sin(t)


def forward(h, y0, steps):
    y = y0
    out = [y]
    for n in range(steps):
        y = y + h * f(n * h, y)
        out.append(y)
    return out


def backward(h, y0, steps):
    y = y0
    out = [y]
    for n in range(steps):
        t = (n + 1) * h
        y = (y + h * LAM * math.cos(t) - h * math.sin(t)) / (1.0 + h * LAM)
        out.append(y)
    return out


for h in (0.0019, 0.0021):
    n = round(T / h)
    ys = forward(h, 1.0, n)
    e = [abs(ys[k] - math.cos(k * h)) for k in range(n + 1)]
    half = n // 2
    rate = (e[n] / e[half]) ** (1.0 / (n - half))
    first = next((k for k in range(n + 1) if e[k] > 1.0), None)
    leave = next((k for k in range(n + 1) if abs(ys[k]) > 1.5), None)
    print(f"forward Euler h = {h}: {n} steps to t = {n * h!r}; |y - cos t| at the end {e[n]!r}; "
          f"growth per step over the second half {rate!r}; |error| first above 1 at t = {first * h if first else None}; "
          f"|y| first above 1.5 at t = {leave * h if leave else None}")
for h in (0.01, 0.1):
    n = round(T / h)
    ys = backward(h, 1.0, n)
    print(f"backward Euler h = {h}: |y(2) - cos 2| = {abs(ys[-1] - math.cos(n * h))!r}")

M = 4
print("h, measured forward factor, |1 - h lam|, measured backward factor, 1/(1 + h lam)")
worst_f = worst_b = 0.0
for k in range(36):
    h = 10.0 ** (-4.0 + 3.0 * k / 35.0)
    df = forward(h, 2.0, M)[M] - forward(h, 1.0, M)[M]
    db = backward(h, 2.0, M)[M] - backward(h, 1.0, M)[M]
    gf, gb = abs(df) ** (1.0 / M), abs(db) ** (1.0 / M)
    pf, pb = abs(1.0 - h * LAM), 1.0 / (1.0 + h * LAM)
    worst_f = max(worst_f, abs(gf - pf) / max(pf, 1.0))
    worst_b = max(worst_b, abs(gb - pb) / pb)
    print(f"{h:.6e} {gf:.12f} {pf:.12f} {gb:.12f} {pb:.12f}")
print(f"largest gap: forward {worst_f:.2e} (relative to max(1, |1 - h lam|)), backward {worst_b:.2e} (relative)")

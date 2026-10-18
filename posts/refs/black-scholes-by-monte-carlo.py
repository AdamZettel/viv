# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-black-scholes-by-monte-carlo.viv.

1. The Black-Scholes price of the call and the put, with scipy.stats.norm,
   and the exact standard deviation of the discounted payoff (a closed form,
   below), which sets the Monte Carlo standard error sigma / sqrt(N).
2. The Viv run replayed in Python: Viv's rand() is Park-Miller MINSTD and
   lib/random's randn is Box-Muller's cosine branch (u1 then u2). log and cos
   may differ from Viv's own libm in the last bit, so the replay agrees to
   about 1e-12, not bit for bit.
3. 2000 independent Monte Carlo runs with numpy's generator: how much the
   price and the sample standard deviation vary from run to run.

    python3 posts/refs/black-scholes-by-monte-carlo.py > posts/refs/black-scholes-by-monte-carlo.out
"""
import math
import numpy as np
from scipy.stats import norm

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N_MAX = 2 ** 17
SEED = 1973.0

d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
C = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
P = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
print(f"d1 = {d1!r}, d2 = {d2!r}")
print(f"call C = {C!r}")
print(f"put  P = {P!r}")
print(f"C - P = {C - P!r}, S0 - K e^(-rT) = {S0 - K * math.exp(-r * T)!r}")

# second moment of the discounted payoff Y = e^{-rT} max(S_T - K, 0):
# E[S_T^2 1{S_T > K}] = S0^2 e^{(2r + sigma^2) T} N(d1 + sigma sqrt T)
# E[S_T 1{S_T > K}]   = S0 e^{rT} N(d1)
# P(S_T > K)          = N(d2)
e2 = S0 ** 2 * math.exp((2 * r + sigma ** 2) * T) * norm.cdf(d1 + sigma * math.sqrt(T)) \
    - 2 * K * S0 * math.exp(r * T) * norm.cdf(d1) + K ** 2 * norm.cdf(d2)
EY2 = math.exp(-2 * r * T) * e2
sd = math.sqrt(EY2 - C ** 2)
print(f"E[Y^2] = {EY2!r}; standard deviation of the discounted payoff = {sd!r}")
print(f"standard error at N = {N_MAX}: {sd / math.sqrt(N_MAX)!r}")

# ---- 2. replay ---------------------------------------------------------------
state = math.floor(abs(SEED)) % 2147483646.0 + 1.0


def rand():
    global state
    state = (16807.0 * state) % 2147483647.0
    return state / 2147483647.0


def randn():
    u1 = rand()
    u2 = rand()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * 3.141592653589793 * u2)


drift = (r - 0.5 * sigma * sigma) * T
vol = sigma * math.sqrt(T)
disc = math.exp(-r * T)
s1 = 0.0
s2 = 0.0
ys = np.zeros(N_MAX)
marks = {2 ** k: None for k in range(4, 18)}
for n in range(1, N_MAX + 1):
    ST = S0 * math.exp(drift + vol * randn())
    y = disc * max(ST - K, 0.0)
    ys[n - 1] = y
    s1 += y
    s2 += y * y
    if n in marks:
        m = s1 / n
        marks[n] = (m, math.sqrt((s2 / n - m * m) * n / (n - 1)))
for n, (m, s) in marks.items():
    print(f"replay N = {n:6d}: price {m:.12f}  sample sd {s:.12f}  (price - C) / (sd/sqrt N) = {(m - C) / (sd / math.sqrt(n)):+.3f}")

KS = list(range(4, 13))


def rms_slope(Y):
    """RMS error against C of the batch means of 2^k consecutive payoffs, k in KS,
    and the least-squares slope of log(RMS) against log(N)."""
    rms = []
    for k in KS:
        b = Y.reshape(-1, 2 ** k).mean(axis=1)
        rms.append(math.sqrt(np.mean((b - C) ** 2)))
    x = np.log(2.0 ** np.array(KS))
    yv = np.log(np.array(rms))
    slope = np.sum((x - x.mean()) * (yv - yv.mean())) / np.sum((x - x.mean()) ** 2)
    return rms, slope


rms_v, slope_v = rms_slope(ys)
print("replay: RMS batch error at N = 2^k, k = 4..12:", [f"{v:.10f}" for v in rms_v])
print("        sd / sqrt N:                          ", [f"{sd / math.sqrt(2 ** k):.10f}" for k in KS])
print(f"replay: slope of log RMS against log N = {slope_v:.12f}")

# ---- 3. run-to-run spread --------------------------------------------------------
rng = np.random.default_rng(99)
REPS = 2000
prices = np.zeros(REPS)
sds = np.zeros(REPS)
slopes = np.zeros(REPS)
rms16 = np.zeros(REPS)
for i in range(REPS):
    Z = rng.standard_normal(N_MAX)
    Y = disc * np.maximum(S0 * np.exp(drift + vol * Z) - K, 0.0)
    prices[i] = Y.mean()
    sds[i] = Y.std(ddof=1)
    rr, sl = rms_slope(Y)
    slopes[i] = sl
    rms16[i] = rr[0]
print(f"{REPS} numpy runs of N = {N_MAX}: price mean {prices.mean():.5f}, spread {prices.std(ddof=1):.5f} "
      f"(sd / sqrt N = {sd / math.sqrt(N_MAX):.5f}); sample sd mean {sds.mean():.5f}, spread {sds.std(ddof=1):.5f}")
print(f"  slope of log RMS against log N: mean {slopes.mean():.5f}, spread {slopes.std(ddof=1):.5f}")
print(f"  RMS batch error at N = 16: mean {rms16.mean():.5f}, spread {rms16.std(ddof=1):.5f} (sd / 4 = {sd / 4:.5f})")

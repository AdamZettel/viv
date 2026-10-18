# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Benchmarks of the option kernels: pure Python, NumPy, Numba, Viv exact,
Viv fast. Same algorithm in each, same inputs; the median of 5 warm runs.

    python examples/options/bench.py iv|crr|pde|asian [--out results.json]

Runs only while the 1-minute load average is below 4, one benchmark per
invocation. Pure Python runs a slice of the work (stated as `scale`) and its
time is multiplied up; every other column runs the full size."""

import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE))
import vipyr  # noqa: E402
import reference as ref  # noqa: E402

RUNS = 5


def load():
    return os.getloadavg()[0]


def median_time(f, runs=RUNS):
    f()                       # warm: caches, JIT, first-touch pages
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def viv(name):
    return vipyr.load(HERE / f"{name}.viv"), vipyr.load(HERE / f"{name}.viv", fast=True)

# ------------------------------------------------------------ implied vol


def iv_inputs(n):
    """n quotes whose time value is above 1e-5 of the price, so the price
    pins sigma (test_options.py, the round-trip test)."""
    rng = np.random.default_rng(0)
    m = 2 * n
    S = np.full(m, 100.0)
    K = 100.0 * np.exp(rng.uniform(-0.5, 0.5, m))
    T = rng.uniform(0.05, 2.0, m)
    sig = rng.uniform(0.1, 0.6, m)
    C = rng.integers(0, 2, m).astype(float)
    p = ref.bs_price(S, K, T, 0.02, 0.0, sig, C)
    F = S * np.exp(0.02 * T)
    tv = p - np.exp(-0.02 * T) * np.where(C > 0.5, np.maximum(F - K, 0), np.maximum(K - F, 0))
    keep = np.flatnonzero(tv > 1e-5 * p)[:n]
    return p[keep], S[keep], K[keep], T[keep], C[keep]


def _ncdf_py(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def iv_py(price, S, K, T, r, q, c):
    """bs.viv's implied_vol, line for line, with math.erfc for the CDF."""
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    o = price / df - (max(F - K, 0.0) if c else max(K - F, 0.0))
    bound = F if K >= F else K
    if not (o > 0.0) or o >= bound:
        return math.nan
    target = math.log(o)
    lo, hi = 0.0, 1e300
    s = max(math.sqrt(2.0 * abs(math.log(F / K))), 0.1)
    lfk = math.log(F / K)
    for _ in range(200):
        d1 = lfk / s + 0.5 * s
        d2 = d1 - s
        v = F * _ncdf_py(d1) - K * _ncdf_py(d2) if K >= F else K * _ncdf_py(-d2) - F * _ncdf_py(-d1)
        if v > 0.0:
            f = math.log(v) - target
            if f == 0.0:
                return s / math.sqrt(T)
            if f > 0.0:
                hi = s
            else:
                lo = s
            sn = s - f * v / (F * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi))
        else:
            lo = s
            sn = -1.0
        if not (lo < sn < hi):
            sn = 0.5 * (lo + hi) if hi < 1e299 else 2.0 * s
        if abs(sn - s) <= 4e-16 * s:
            return sn / math.sqrt(T)
        s = sn
    return s / math.sqrt(T)


_ncdf_nb = njit(_ncdf_py)
# the same source, compiled by Numba with the compiled CDF in its globals
iv_nb = njit(types.FunctionType(iv_py.__code__, {**iv_py.__globals__, "_ncdf_py": _ncdf_nb}, "iv_nb"))


@njit
def iv_nb_all(price, S, K, T, r, q, C):
    out = np.empty(len(price))
    for i in range(len(price)):
        out[i] = iv_nb(price[i], S[i], K[i], T[i], r, q, C[i] > 0.5)
    return out


def iv_np(price, S, K, T, r, q, C, iters=60):
    """The same safeguarded log-Newton, vectorized: every quote takes the
    same number of steps (a fixed count, enough for all to converge)."""
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    o = price / df - np.where(C > 0.5, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    target = np.log(o)
    lfk = np.log(F / K)
    otm_call = K >= F
    lo = np.zeros_like(o)
    hi = np.full_like(o, 1e300)
    s = np.maximum(np.sqrt(2.0 * np.abs(lfk)), 0.1)
    for _ in range(iters):
        d1 = lfk / s + 0.5 * s
        d2 = d1 - s
        v = np.where(otm_call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log(v) - target
            hi = np.where(f > 0, s, hi)
            lo = np.where(f <= 0, s, lo)
            sn = s - f * v / (F * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi))
        bad = ~((sn > lo) & (sn < hi))
        sn = np.where(bad, np.where(hi < 1e299, 0.5 * (lo + hi), 2.0 * s), sn)
        s = sn
    return s / np.sqrt(T)


def bench_iv():
    n = 1_000_000
    p, S, K, T, C = iv_inputs(n)
    exact, fast = viv("bs")
    m = 20_000
    out = {"size": "1e6 quotes", "scale": n // m}
    out["python"] = median_time(lambda: [iv_py(p[i], S[i], K[i], T[i], 0.02, 0.0, C[i] > 0.5) for i in range(m)]) * (n // m)
    out["numpy"] = median_time(lambda: iv_np(p, S, K, T, 0.02, 0.0, C))
    out["numba"] = median_time(lambda: iv_nb_all(p, S, K, T, 0.02, 0.0, C))
    out["viv_exact"] = median_time(lambda: exact.implied_vols(p, S, K, T, 0.02, 0.0, C))
    out["viv_fast"] = median_time(lambda: fast.implied_vols(p, S, K, T, 0.02, 0.0, C))
    sig = exact.implied_vols(p, S, K, T, 0.02, 0.0, C)
    out["agree"] = float(np.nanmax(np.abs(iv_np(p, S, K, T, 0.02, 0.0, C) / sig - 1)))
    return out

# ------------------------------------------------------------ CRR


def crr_py(S, K, T, r, q, sigma, N, is_call, american):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)
    sg = 1.0 if is_call else -1.0
    pw = [math.exp(sigma * math.sqrt(dt) * (k - N)) for k in range(2 * N + 1)]
    v = [max(sg * (S * pw[2 * j] - K), 0.0) for j in range(N + 1)]
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = pu * v[j + 1] + pd * v[j]
            v[j] = max(cont, sg * (S * pw[2 * j - i + N] - K)) if american else cont
    return v[0]


crr_nb = njit(crr_py)


@njit
def crr_nb_all(S, K, T, r, q, sig, C, N):
    out = np.empty(len(S))
    for i in range(len(S)):
        out[i] = crr_nb(S[i], K[i], T[i], r, q, sig[i], N, C[i] > 0.5, True)
    return out


def bench_crr():
    n, N = 1000, 1000
    rng = np.random.default_rng(1)
    S = rng.uniform(80, 120, n)
    K = np.full(n, 100.0)
    T = rng.uniform(0.25, 2.0, n)
    sig = rng.uniform(0.15, 0.5, n)
    C = np.zeros(n)
    exact, fast = viv("crr")
    m = 10
    out = {"size": "1e3 American puts, N = 1000", "scale": n // m}
    out["python"] = median_time(lambda: [crr_py(S[i], K[i], T[i], 0.05, 0.0, sig[i], N, False, True) for i in range(m)]) * (n // m)
    out["numpy"] = median_time(lambda: [ref.crr_price(S[i], K[i], T[i], 0.05, 0.0, sig[i], N, False, True) for i in range(n)])
    out["numba"] = median_time(lambda: crr_nb_all(S, K, T, 0.05, 0.0, sig, C, N))
    out["viv_exact"] = median_time(lambda: exact.crr_prices(S, K, T, 0.05, 0.0, sig, C, N, True))
    out["viv_fast"] = median_time(lambda: fast.crr_prices(S, K, T, 0.05, 0.0, sig, C, N, True))
    out["agree"] = float(np.max(np.abs(exact.crr_prices(S, K, T, 0.05, 0.0, sig, C, N, True) - crr_nb_all(S, K, T, 0.05, 0.0, sig, C, N))))
    return out

# ------------------------------------------------------------ CN PDE


def cn_py(S0, K, T, r, q, sigma, Smax, M, Nt, implicit, np_mod=None):
    """cnpde.viv's cn_american_put; with np_mod, the right side is built
    with whole-array NumPy operations (the Brennan-Schwartz sweep is a
    recurrence and stays a loop)."""
    dS = Smax / M
    i = np.arange(M + 1, dtype=float)
    pay = np.maximum(K - i * dS, 0.0)
    v = pay.copy()
    a = 0.5 * sigma * sigma * i * i - 0.5 * (r - q) * i
    b = -sigma * sigma * i * i - r
    c = 0.5 * sigma * sigma * i * i + 0.5 * (r - q) * i
    dt = T / Nt
    a, b, c, pay = list(a), list(b), list(c), list(pay)
    an, bn, cn = np.array(a), np.array(b), np.array(c)
    v = list(v)
    bp = [0.0] * (M + 1)
    dp = [0.0] * (M + 1)
    for n in range(Nt + implicit):
        theta, h = (1.0, 0.5 * dt) if n < 2 * implicit else (0.5, dt)
        if np_mod:
            vv = np.array(v)
            rhs = list(np.concatenate([[0.0], vv[1:M] + (1 - theta) * h * (an[1:M] * vv[:M - 1] + bn[1:M] * vv[1:M] + cn[1:M] * vv[2:]), [0.0]]))
        else:
            rhs = [0.0] * (M + 1)
            for k in range(1, M):
                rhs[k] = v[k] + (1 - theta) * h * (a[k] * v[k - 1] + b[k] * v[k] + c[k] * v[k + 1])
        rhs[1] += theta * h * a[1] * K
        bp[M - 1] = 1.0 - theta * h * b[M - 1]
        dp[M - 1] = rhs[M - 1]
        for k in range(M - 2, 0, -1):
            mlt = (-theta * h * c[k]) / bp[k + 1]
            bp[k] = 1.0 - theta * h * b[k] - mlt * (-theta * h * a[k + 1])
            dp[k] = rhs[k] - mlt * dp[k + 1]
        v[0] = K
        for k in range(1, M):
            v[k] = max((dp[k] - (-theta * h * a[k]) * v[k - 1]) / bp[k], pay[k])
        v[M] = 0.0
    x = S0 / dS
    j = min(max(int(math.floor(x + 0.5)), 1), M - 1)
    t = x - j
    return v[j - 1] * 0.5 * t * (t - 1) + v[j] * (1 - t * t) + v[j + 1] * 0.5 * t * (t + 1)


@njit
def cn_nb(S0, K, T, r, q, sigma, Smax, M, Nt, implicit):
    dS = Smax / M
    v = np.empty(M + 1)
    pay = np.empty(M + 1)
    a = np.empty(M + 1)
    b = np.empty(M + 1)
    c = np.empty(M + 1)
    for i in range(M + 1):
        pay[i] = max(K - i * dS, 0.0)
        v[i] = pay[i]
        a[i] = 0.5 * sigma * sigma * i * i - 0.5 * (r - q) * i
        b[i] = -sigma * sigma * i * i - r
        c[i] = 0.5 * sigma * sigma * i * i + 0.5 * (r - q) * i
    dt = T / Nt
    bp = np.zeros(M + 1)
    dp = np.zeros(M + 1)
    rhs = np.zeros(M + 1)
    for n in range(Nt + implicit):
        theta, h = (1.0, 0.5 * dt) if n < 2 * implicit else (0.5, dt)
        for k in range(1, M):
            rhs[k] = v[k] + (1 - theta) * h * (a[k] * v[k - 1] + b[k] * v[k] + c[k] * v[k + 1])
        rhs[1] += theta * h * a[1] * K
        bp[M - 1] = 1.0 - theta * h * b[M - 1]
        dp[M - 1] = rhs[M - 1]
        for k in range(M - 2, 0, -1):
            mlt = (-theta * h * c[k]) / bp[k + 1]
            bp[k] = 1.0 - theta * h * b[k] - mlt * (-theta * h * a[k + 1])
            dp[k] = rhs[k] - mlt * dp[k + 1]
        v[0] = K
        for k in range(1, M):
            v[k] = max((dp[k] - (-theta * h * a[k]) * v[k - 1]) / bp[k], pay[k])
        v[M] = 0.0
    x = S0 / dS
    j = min(max(int(math.floor(x + 0.5)), 1), M - 1)
    t = x - j
    return v[j - 1] * 0.5 * t * (t - 1) + v[j] * (1 - t * t) + v[j + 1] * 0.5 * t * (t + 1)


def bench_pde():
    args = (36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 200.0, 2000, 1000, 2)
    exact, fast = viv("cnpde")
    out = {"size": "American put, 2000 x 1000 grid", "scale": 1}
    out["python"] = median_time(lambda: cn_py(*args))
    out["numpy"] = median_time(lambda: cn_py(*args, np_mod=True))
    out["numba"] = median_time(lambda: cn_nb(*args))
    out["viv_exact"] = median_time(lambda: exact.cn_american_put(*args))
    out["viv_fast"] = median_time(lambda: fast.cn_american_put(*args))
    out["agree"] = abs(exact.cn_american_put(*args) - cn_nb(*args))
    return out

# ------------------------------------------------------------ MC Asian


def asian_py(S, K, T, r, q, sigma, n, npaths, seed):
    dt = T / n
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)
    ya = []
    yg = []
    for p in range(npaths):
        x = math.log(S)
        sa = sl = 0.0
        for b in range((n + 1) // 2):
            w = ref.philox4x32((p & 0xFFFFFFFF, p >> 32, b, 0), (seed, 0))
            u1 = ((w[0] >> 5) * 67108864.0 + (w[1] >> 6) + 0.5) / 9007199254740992.0
            u2 = ((w[2] >> 5) * 67108864.0 + (w[3] >> 6) + 0.5) / 9007199254740992.0
            rad = math.sqrt(-2.0 * math.log(u1))
            for k, g in enumerate((rad * math.cos(2 * math.pi * u2), rad * math.sin(2 * math.pi * u2))):
                if 2 * b + k < n:
                    x += drift + vol * g
                    sa += math.exp(x)
                    sl += x
        ya.append(disc * max(sa / n - K, 0.0))
        yg.append(disc * max(math.exp(sl / n) - K, 0.0))
    return ya, yg


@njit
def _philox_nb(c0, c1, c2, c3, k0, k1):
    mask = np.uint64(0xFFFFFFFF)
    x0, x1, x2, x3 = np.uint64(c0), np.uint64(c1), np.uint64(c2), np.uint64(c3)
    ka, kb = np.uint64(k0), np.uint64(k1)
    for r in range(10):
        if r > 0:
            ka = (ka + np.uint64(0x9E3779B9)) & mask
            kb = (kb + np.uint64(0xBB67AE85)) & mask
        p0 = np.uint64(0xD2511F53) * x0
        p1 = np.uint64(0xCD9E8D57) * x2
        x0, x1, x2, x3 = (p1 >> np.uint64(32)) ^ x1 ^ ka, p1 & mask, (p0 >> np.uint64(32)) ^ x3 ^ kb, p0 & mask
    return x0, x1, x2, x3


@njit
def asian_nb(S, K, T, r, q, sigma, n, npaths, seed):
    dt = T / n
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)
    ya = np.empty(npaths)
    yg = np.empty(npaths)
    for p in range(npaths):
        x = math.log(S)
        sa = 0.0
        sl = 0.0
        for b in range((n + 1) // 2):
            w0, w1, w2, w3 = _philox_nb(p & 0xFFFFFFFF, p >> 32, b, 0, seed, 0)
            u1 = (float(w0 >> np.uint64(5)) * 67108864.0 + float(w1 >> np.uint64(6)) + 0.5) / 9007199254740992.0
            u2 = (float(w2 >> np.uint64(5)) * 67108864.0 + float(w3 >> np.uint64(6)) + 0.5) / 9007199254740992.0
            rad = math.sqrt(-2.0 * math.log(u1))
            x += drift + vol * rad * math.cos(2 * math.pi * u2)
            sa += math.exp(x)
            sl += x
            if 2 * b + 1 < n:
                x += drift + vol * rad * math.sin(2 * math.pi * u2)
                sa += math.exp(x)
                sl += x
        ya[p] = disc * max(sa / n - K, 0.0)
        yg[p] = disc * max(math.exp(sl / n) - K, 0.0)
    return ya, yg


def bench_asian():
    args = (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 12)
    npaths = 1_000_000
    exact, fast = viv("mc")
    m = 20_000
    out = {"size": "1e6 paths x 12 dates", "scale": npaths // m}
    out["python"] = median_time(lambda: asian_py(*args, m, 7)) * (npaths // m)
    out["numpy"] = median_time(lambda: ref.asian_call_np(*args, npaths, 7))
    out["numba"] = median_time(lambda: asian_nb(*args, npaths, 7))
    out["viv_exact"] = median_time(lambda: exact.asian_call(*args, npaths, 7))
    out["viv_fast"] = median_time(lambda: fast.asian_call(*args, npaths, 7))
    ya, yg = asian_nb(*args, npaths, 7)
    out["agree"] = abs(exact.asian_call(*args, npaths, 7)[2] - float(np.mean(ya)))
    return out


BENCHES = {"iv": bench_iv, "crr": bench_crr, "pde": bench_pde, "asian": bench_asian}


def main():
    which = sys.argv[1]
    outp = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "bench_results.json"
    la = load()
    if la >= 4.0:
        sys.exit(f"bench: load average {la:.2f} is not below 4; not running")
    res = BENCHES[which]()
    res["load_before"] = la
    res["load_after"] = load()
    res["machine"] = f"{platform.machine()} {platform.mac_ver()[0] or platform.release()}, " + subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    res["versions"] = f"Python {platform.python_version()}, NumPy {np.__version__}, Numba {__import__('numba').__version__}"
    allres = json.loads(outp.read_text()) if outp.exists() else {}
    allres[which] = res
    outp.write_text(json.dumps(allres, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Independent references for the option kernels: closed forms and NumPy /
SciPy implementations written from the papers, sharing no code with the
.viv files. Used by test_options.py and bench.py."""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, log_ndtr

# ------------------------------------------------------------ Black-Scholes

def bs_price(S, K, T, r, q, sigma, is_call):
    """Black-Scholes with scipy's ndtr; the out-of-the-money side is taken
    from log_ndtr so deep tails keep relative accuracy."""
    S, K, T, sigma, is_call = np.broadcast_arrays(*map(np.asarray, (S, K, T, sigma, is_call)))
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    s = sigma * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    call_otm = F * np.exp(log_ndtr(d1)) - K * np.exp(log_ndtr(d2))
    put_otm = K * np.exp(log_ndtr(-d2)) - F * np.exp(log_ndtr(-d1))
    otm = np.where(K >= F, call_otm, put_otm)
    itm_add = np.where(is_call.astype(bool), np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return df * (otm + itm_add)


def bs_greeks(S, K, T, r, q, sigma, is_call):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    delta = np.exp(-q * T) * np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)
    gamma = np.exp(-q * T) * pdf / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf * np.sqrt(T)
    return delta, gamma, vega

# ------------------------------------------------------------ CRR

def crr_price(S, K, T, r, q, sigma, N, is_call, american):
    """The CRR tree, vectorized over the nodes of a step (NumPy)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sg = 1.0 if is_call else -1.0
    j = np.arange(N + 1)
    v = np.maximum(sg * (S * u ** (2.0 * j - N) - K), 0.0)
    for i in range(N - 1, -1, -1):
        v = disc * (p * v[1:i + 2] + (1 - p) * v[:i + 1])
        if american:
            v = np.maximum(v, sg * (S * u ** (2.0 * np.arange(i + 1) - i) - K))
    return float(v[0])

# ------------------------------------------------------------ Heston

def heston_cf(u, T, r, q, v0, kappa, theta, eta, rho):
    iu = 1j * u
    b = kappa - rho * eta * iu
    D = np.sqrt(b * b + eta * eta * (u * u + iu))
    G = (b - D) / (b + D)
    e = np.exp(-D * T)
    C = kappa * theta / eta**2 * ((b - D) * T - 2 * np.log((1 - G * e) / (1 - G)))
    B = (b - D) / eta**2 * (1 - e) / (1 - G * e)
    return np.exp(iu * (r - q) * T + C + B * v0)


def heston_call_quad(S, K, T, r, q, v0, kappa, theta, eta, rho):
    """Heston's two probabilities by Gil-Pelaez inversion, scipy.integrate.quad
    on (0, inf)."""
    x = np.log(S / K)
    p = (T, r, q, v0, kappa, theta, eta, rho)
    norm = heston_cf(-1j, *p)

    def f1(u):
        return (np.exp(1j * u * x) * heston_cf(u - 1j, *p) / norm / (1j * u)).real

    def f2(u):
        return (np.exp(1j * u * x) * heston_cf(u, *p) / (1j * u)).real

    kw = dict(limit=1000, epsabs=1e-14, epsrel=1e-13)
    P1 = 0.5 + quad(f1, 0, np.inf, **kw)[0] / np.pi
    P2 = 0.5 + quad(f2, 0, np.inf, **kw)[0] / np.pi
    return S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

# ------------------------------------------------------------ barrier

def down_out_call(S, K, H, T, r, q, sigma):
    """A down-and-out call, H <= K, no rebate, by the method of images, which
    is Reiner and Rubinstein's (1991) formula for this case:
    C_do = C(S) - (H / S)^(2 nu / sigma^2) C(H^2 / S), nu = r - q - sigma^2 / 2."""
    nu = r - q - 0.5 * sigma**2
    return float(bs_price(S, K, T, r, q, sigma, 1) - (H / S) ** (2 * nu / sigma**2) * bs_price(H * H / S, K, T, r, q, sigma, 1))

# ------------------------------------------------------------ Philox and Asian

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox4x32(c, k):
    """Random123's philox4x32-10 on Python ints."""
    x = list(c)
    k0, k1 = k
    for r in range(10):
        if r:
            k0 = (k0 + W0) & MASK
            k1 = (k1 + W1) & MASK
        p0 = M0 * x[0]
        p1 = M1 * x[2]
        x = [(p1 >> 32) ^ x[1] ^ k0, p1 & MASK, (p0 >> 32) ^ x[3] ^ k1, p0 & MASK]
    return x


def philox4x32_np(c0, c1, c2, c3, k0, k1):
    """philox4x32-10 vectorized over uint64 arrays of counters."""
    x0, x1, x2, x3 = (np.asarray(c, dtype=np.uint64) for c in (c0, c1, c2, c3))
    ka, kb = np.uint64(k0), np.uint64(k1)
    m = np.uint64(MASK)
    for r in range(10):
        if r:
            ka = (ka + np.uint64(W0)) & m
            kb = (kb + np.uint64(W1)) & m
        p0 = np.uint64(M0) * x0
        p1 = np.uint64(M1) * x2
        x0, x1, x2, x3 = (p1 >> np.uint64(32)) ^ x1 ^ ka, p1 & m, (p0 >> np.uint64(32)) ^ x3 ^ kb, p0 & m
    return x0, x1, x2, x3


def normals_np(npaths, nsteps, seed, stream):
    """The same normals as mc.viv's `normals`: path p, block b -> Box-Muller
    of two 53-bit uniforms from one Philox block."""
    nb = (nsteps + 1) // 2
    p = np.repeat(np.arange(npaths, dtype=np.uint64), nb)
    b = np.tile(np.arange(nb, dtype=np.uint64), npaths)
    w0, w1, w2, w3 = philox4x32_np(p & np.uint64(MASK), p >> np.uint64(32), b, np.zeros_like(p), seed, stream)
    u1 = ((w0 >> np.uint64(5)).astype(np.float64) * 67108864.0 + (w1 >> np.uint64(6)).astype(np.float64) + 0.5) / 9007199254740992.0
    u2 = ((w2 >> np.uint64(5)).astype(np.float64) * 67108864.0 + (w3 >> np.uint64(6)).astype(np.float64) + 0.5) / 9007199254740992.0
    rad = np.sqrt(-2.0 * np.log(u1))
    z = np.empty((npaths, 2 * nb))
    z[:, 0::2] = (rad * np.cos(2 * np.pi * u2)).reshape(npaths, nb)
    z[:, 1::2] = (rad * np.sin(2 * np.pi * u2)).reshape(npaths, nb)
    return z[:, :nsteps]


def geometric_asian_call(S, K, T, r, q, sigma, n):
    nu = r - q - 0.5 * sigma**2
    mu = np.log(S) + nu * T * (n + 1) / (2 * n)
    sg = sigma * np.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n * n))
    d2 = (mu - np.log(K)) / sg
    return np.exp(-r * T) * (np.exp(mu + 0.5 * sg * sg) * ndtr(d2 + sg) - K * ndtr(d2))


def asian_call_np(S, K, T, r, q, sigma, n, npaths, seed):
    """mc.viv's asian_call in NumPy: arithmetic Asian with the geometric control variate."""
    dt = T / n
    z = normals_np(npaths, n, seed, 0)
    x = np.log(S) + np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    disc = np.exp(-r * T)
    ya = disc * np.maximum(np.exp(x).mean(axis=1) - K, 0.0)
    yg = disc * np.maximum(np.exp(x.mean(axis=1)) - K, 0.0)
    c = np.cov(ya, yg)
    beta = c[0, 1] / c[1, 1]
    cv = ya.mean() - beta * (yg.mean() - geometric_asian_call(S, K, T, r, q, sigma, n))
    var = c[0, 0] - 2 * beta * c[0, 1] + beta**2 * c[1, 1]
    return cv, np.sqrt(var / npaths), ya.mean(), np.sqrt(c[0, 0] / npaths), yg.mean()

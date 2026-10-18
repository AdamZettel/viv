# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The option kernels against their references, with stated tolerances.

    python -m pytest examples/options/test_options.py

Each .viv module beside this file is imported through `import vipyr.auto`.
References: closed forms, published values (cited at each test) and the
independent NumPy / SciPy code in reference.py."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

import vipyr.auto  # noqa: E402,F401
import bs  # noqa: E402
import crr  # noqa: E402
import heston  # noqa: E402
import cnpde  # noqa: E402
import mc  # noqa: E402
import greeks  # noqa: E402
import reference as ref  # noqa: E402

HESTON_FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, eta=0.5751, rho=-0.5711)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))

# ------------------------------------------------------------ Black-Scholes


def test_norm_cdf_against_mpmath():
    """Relative error over the whole range, tails included, against mpmath
    at 40 digits. (scipy's ndtr is itself 2.3e-13 off near x = -35.6.)"""
    import mpmath as mp
    mp.mp.dps = 40
    x = np.concatenate([np.linspace(-37.5, 8.0, 4001), [-0.67448975, 0.67448975, -5.656854249492380195]])
    got = np.array([bs.norm_cdf(v) for v in x])
    assert rel(got, [float(mp.ncdf(v)) for v in x]) < 1e-15
    assert rel(got[x > -20], ndtr(x[x > -20])) < 1e-13


def test_bs_prices_against_the_closed_form():
    rng = np.random.default_rng(1)
    n = 2000
    S = rng.uniform(50, 150, n)
    K = S * np.exp(rng.uniform(-1.0, 1.0, n))
    T = rng.uniform(0.02, 3.0, n)
    sig = rng.uniform(0.05, 1.0, n)
    cp = rng.integers(0, 2, n).astype(float)
    got = bs.bs_prices(S, K, T, 0.03, 0.01, sig, cp)
    want = ref.bs_price(S, K, T, 0.03, 0.01, sig, cp)
    ok = want > 1e-8 * S
    assert ok.mean() > 0.8
    assert rel(got[ok], want[ok]) < 1e-12


def test_bs_deep_out_of_the_money_against_mpmath():
    """Far in the tail the two terms of the formula nearly cancel (d1, d2 near
    -27 for a price of 1e-158), so a few digits go; against mpmath at 40
    digits the worst relative error stays below 1e-10."""
    import mpmath as mp
    mp.mp.dps = 40

    def exact(S, K, T, r, q, s):
        S, K, T, s, r, q = map(mp.mpf, (S, K, T, s, r, q))
        F = S * mp.e ** ((r - q) * T)
        sq = s * mp.sqrt(T)
        d1 = mp.log(F / K) / sq + sq / 2
        return float(mp.e ** (-r * T) * (F * mp.ncdf(d1) - K * mp.ncdf(d1 - sq)))
    worst = 0.0
    for S, K, T, s in [(55.785312375657924, 150.65985391943852, 0.46365751730610233, 0.054086253565277666),
                       (100.0, 300.0, 0.1, 0.25), (100.0, 130.0, 0.05, 0.1), (100.0, 200.0, 1.0, 0.1)]:
        got = bs.bs_price(S, K, T, 0.03, 0.01, s, True)
        worst = max(worst, abs(got / exact(S, K, T, 0.03, 0.01, s) - 1))
    assert worst < 1e-10


def iv_grid():
    m = np.exp(np.linspace(-0.7, 0.7, 15))          # K / F
    T = np.array([0.02, 0.1, 0.5, 1.0, 2.0, 5.0])
    sig = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
    M, TT, SS, C = np.meshgrid(m, T, sig, [0.0, 1.0], indexing="ij")
    M, TT, SS, C = (a.ravel() for a in (M, TT, SS, C))
    S = np.full_like(M, 100.0)
    K = 100.0 * np.exp(0.02 * TT) * M
    return S, K, TT, SS, C


def test_implied_vol_round_trip_on_a_grid():
    """1e-12 relative on 15 moneyness x 6 expiries x 6 vols x {call, put}
    wherever the time value is above 1e-5 of the price. Below that the
    quoted price, rounded to a double, does not pin sigma to 1e-12: the
    error grows like 1e-16 * price / time value (1e-6 at 1e-10). A
    subnormal price (below 2.2e-308) has fewer than 53 bits to invert."""
    S, K, T, sig, C = iv_grid()
    p = bs.bs_prices(S, K, T, 0.02, 0.0, sig, C)
    iv = bs.implied_vols(p, S, K, T, 0.02, 0.0, C)
    F = S * np.exp(0.02 * T)
    tv = p - np.exp(-0.02 * T) * np.where(C > 0.5, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = (tv > 1e-5 * p) & (p > 1e-300)       # and not a subnormal price
    assert ok.mean() > 0.8
    assert rel(iv[ok], sig[ok]) < 1e-12


def test_implied_vol_edge_cases():
    r, q = 0.05, 0.0
    cases = [  # S, K, T, sigma, call, tolerance
        (100.0, 300.0, 0.1, 0.25, 1, 1e-12),      # deep OTM call, price ~1e-40
        (100.0, 20.0, 0.5, 0.3, 0, 1e-12),        # deep OTM put
        (100.0, 100.0, 1e-4, 0.2, 1, 1e-12),      # near expiry, at the money
        (100.0, 101.0, 1e-3, 0.2, 0, 1e-11),      # near expiry, slightly ITM put
        (100.0, 60.0, 0.25, 0.3, 1, 1e-9),        # deep ITM call: time value 1e-5 of the price
        (100.0, 100.0, 1.0, 3.0, 1, 1e-12),       # very high vol
    ]
    for S, K, T, sig, c, tol in cases:
        p = bs.bs_price(S, K, T, r, q, sig, bool(c))
        iv = bs.implied_vol(p, S, K, T, r, q, bool(c))
        assert abs(iv / sig - 1) < tol, (S, K, T, sig, c, p, iv)
    # below intrinsic or above the bound: NaN, not a number that prices it
    assert np.isnan(bs.implied_vol(0.5 * (100 - 80 * np.exp(-0.05)), 100.0, 80.0, 1.0, r, q, True))
    assert np.isnan(bs.implied_vol(101.0, 100.0, 80.0, 1.0, r, q, True))

# ------------------------------------------------------------ CRR


@pytest.mark.parametrize("N,want", [(5, 4.49), (30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)])
def test_crr_hull_american_put(N, want):
    """Hull, Options, Futures, and Other Derivatives: the American put with
    S0 = K = 50, r = 10%, sigma = 40%, T = 5 months; the 5-step tree gives
    4.49 and 30, 50, 100, 500 steps give 4.263, 4.272, 4.278, 4.283."""
    got = crr.crr_price(50.0, 50.0, 5 / 12, 0.1, 0.0, 0.4, N, False, True)
    assert abs(got - want) <= 0.5 * 10.0 ** -(len(str(want)) - 2)
    assert abs(got - ref.crr_price(50.0, 50.0, 5 / 12, 0.1, 0.0, 0.4, N, False, True)) < 1e-12


def test_crr_european_converges_to_black_scholes():
    errs = []
    for N in (250, 1000, 4000):
        got = crr.crr_price(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, N, True, False)
        errs.append(abs(got - float(ref.bs_price(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, 1))))
    assert errs[-1] < 2e-3 and errs[0] > errs[-1]


def test_crr_batch_matches_one_at_a_time():
    S = np.array([36.0, 40.0, 44.0])
    out = crr.crr_prices(S, np.full(3, 40.0), np.ones(3), 0.06, 0.0, np.full(3, 0.2), np.zeros(3), 400, True)
    for i in range(3):
        assert out[i] == crr.crr_price(S[i], 40.0, 1.0, 0.06, 0.0, 0.2, 400, False, True)

# ------------------------------------------------------------ Heston


def test_heston_cos_published_values():
    """Fang and Oosterlee (2008), section 5.3: the call with S0 = K = 100,
    r = q = 0, Heston parameters HESTON_FO, has the reference values
    5.785155450 (T = 1) and 22.318945791 (T = 10)."""
    p1 = heston.heston_cos(100.0, [100.0], 1.0, 0.0, 0.0, *HESTON_FO.values(), 1024, 32.0, [1.0])[0]
    p10 = heston.heston_cos(100.0, [100.0], 10.0, 0.0, 0.0, *HESTON_FO.values(), 1024, 32.0, [1.0])[0]
    assert abs(p1 - 5.785155450) < 2e-8         # the published 5.785155450 is itself 1.6e-8 from quad
    assert abs(p10 - 22.318945791) < 1e-9


def test_heston_cos_against_scipy_quad():
    K = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0])
    for T, r, q in [(0.25, 0.02, 0.0), (1.0, 0.0, 0.0), (2.0, 0.03, 0.01)]:
        got = heston.heston_cos(100.0, K, T, r, q, *HESTON_FO.values(), 1024, 32.0, np.ones(len(K)))
        want = np.array([ref.heston_call_quad(100.0, k, T, r, q, *HESTON_FO.values()) for k in K])
        assert np.max(np.abs(got - want)) < 1e-9, (T, got - want)

# ------------------------------------------------------------ PDE


def test_cn_american_put_against_binomial():
    """Longstaff and Schwartz's (2001) Table 1 first row, S = 36, K = 40,
    r = 6%, sigma = 20%, T = 1. Its finite-difference value 4.478 is coarse;
    the reference here is the CRR tree, the mean of N = 10000 and 10001
    (4.48669), which the tree's odd-even oscillation brackets."""
    want = 0.5 * (ref.crr_price(36, 40, 1, 0.06, 0, 0.2, 10000, False, True) + ref.crr_price(36, 40, 1, 0.06, 0, 0.2, 10001, False, True))
    got = cnpde.cn_american_put(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 200.0, 2000, 1000, 2)
    assert abs(got - want) < 2e-4, (got, want)


def test_cn_down_and_out_call_against_reiner_rubinstein():
    for S, K, H, T, r, q, sig in [(100, 100, 90, 1.0, 0.05, 0.0, 0.2), (100, 110, 95, 0.5, 0.03, 0.01, 0.3), (120, 100, 80, 2.0, 0.04, 0.02, 0.25)]:
        got = cnpde.cn_down_out_call(S, K, H, T, r, q, sig, 5.0 * K, 2000, 1000, 2)
        want = ref.down_out_call(S, K, H, T, r, q, sig)
        assert abs(got - want) < 2e-3, (S, K, H, got, want)

# ------------------------------------------------------------ Monte Carlo


def test_philox_matches_the_python_reference():
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = [int(v) for v in rng.integers(0, 2**32, 4)]
        k = [int(v) for v in rng.integers(0, 2**32, 2)]
        assert list(mc.philox4x32(*c, *k)) == ref.philox4x32(c, k)


def test_normals_match_numpy_twin():
    z = mc.normals(500, 7, 11, 1)
    assert np.max(np.abs(z - ref.normals_np(500, 7, 11, 1))) < 1e-13


def test_asian_geometric_and_control_variate():
    args = (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 12)
    cv, se, plain, pse, geo = mc.asian_call(*args, 200_000, 7)
    exact = ref.geometric_asian_call(*args)
    assert abs(geo - exact) < 4 * pse          # the geometric call's MC against its closed form
    assert abs(cv - plain) < 4 * pse           # the two estimators agree
    assert se < pse / 20                        # and the control variate pays
    want = ref.asian_call_np(*args, 200_000, 7)
    assert abs(cv - want[0]) < 1e-10           # the NumPy twin on the same Philox stream


def test_lsm_against_binomial():
    """LSM on Longstaff and Schwartz's first case (their 4.472, s.e. 0.010);
    the binomial value 4.48669. LSM's in-sample policy biases it low by a few
    thousandths, so the allowance is 3 s.e. + 0.01."""
    price, se = mc.lsm_american_put(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, 100_000, 5)
    assert abs(price - 4.48669) < 3 * se + 0.01, (price, se)


def digest(*xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes())
    return h.hexdigest()


def test_bit_identical_across_runs():
    a = digest(*mc.asian_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 12, 50_000, 3))
    b = digest(*mc.asian_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 12, 50_000, 3))
    assert a == b


LIBOMP = Path("/opt/homebrew/opt/libomp")


@pytest.mark.skipif(sys.platform != "darwin" or not LIBOMP.is_dir() or not shutil.which("cc"), reason="needs Homebrew libomp")
def test_bit_identical_across_thread_counts(tmp_path):
    """mc.viv exported and built with OpenMP: the `parallel for` loops run on
    1, 2, 4 and 8 threads, and the results hash the same."""
    from vipyr import _core
    out = tmp_path / "x"
    subprocess.run([_core.compiler(), "export", "--infer", str(HERE / "mc.viv"), "-o", str(out)], check=True, capture_output=True,
                   env=dict(os.environ, VIV_ROOT=str(ROOT)))
    lib = out / "libmc.dylib"
    subprocess.run(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-Xpreprocessor", "-fopenmp", f"-I{LIBOMP}/include",
                    str(out / "mc.c"), "-o", str(lib), f"-L{LIBOMP}/lib", "-lomp", "-lm"], check=True)
    code = ("import sys, hashlib, numpy as np; sys.path.insert(0, %r); import mc\n"
            "r = mc.asian_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 12, 100000, 3) + mc.lsm_american_put(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, 50, 20000, 5)\n"
            "print(hashlib.sha256(np.array(r).tobytes()).hexdigest())" % str(out))
    seen = set()
    for t in (1, 2, 4, 8):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=str(t)))
        assert r.returncode == 0, r.stderr
        seen.add(r.stdout.strip())
    assert len(seen) == 1, seen

# ------------------------------------------------------------ Greeks


def test_bs_greeks_by_autodiff():
    rng = np.random.default_rng(4)
    n = 300
    S = rng.uniform(70, 130, n)
    K = rng.uniform(70, 130, n)
    T = rng.uniform(0.1, 2.0, n)
    sig = rng.uniform(0.1, 0.6, n)
    cp = rng.integers(0, 2, n).astype(float)
    d, g, v = greeks.bs_greeks_arr(S, K, T, 0.03, 0.01, sig, cp)
    dw, gw, vw = ref.bs_greeks(S, K, T, 0.03, 0.01, sig, cp > 0.5)
    assert np.max(np.abs(d - dw)) < 1e-13
    assert rel(g, gw) < 1e-12
    assert rel(v, vw) < 1e-12


def test_heston_greeks_against_finite_differences():
    """Delta and dV/dv0 by autodiff against central differences of the
    SciPy-quad price (h = 1e-3 S, 1e-4 v0); gamma, itself a difference of
    autodiff deltas, against the second difference."""
    p = HESTON_FO
    for K, T in [(90.0, 0.5), (100.0, 1.0), (115.0, 2.0)]:
        v, d, g, dv = greeks.heston_greeks(100.0, K, T, 0.02, 0.0, *p.values(), 4096, 8.0, True)

        def price(S=100.0, v0=p["v0"]):
            return ref.heston_call_quad(S, K, T, 0.02, 0.0, v0, p["kappa"], p["theta"], p["eta"], p["rho"])
        h, hv = 0.02, 1e-4
        assert abs(v - price()) < 1e-8
        assert abs(d - (price(100 + h) - price(100 - h)) / (2 * h)) < 1e-6
        assert abs(g - (price(100 + 5 * h) - 2 * price() + price(100 - 5 * h)) / (5 * h) ** 2) < 1e-5
        assert abs(dv - (price(v0=p["v0"] + hv) - price(v0=p["v0"] - hv)) / (2 * hv)) < 1e-5

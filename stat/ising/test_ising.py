# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Tests of the exported Ising program (stat/ising/ising_mc.export.toml).

    python3 stat/ising/test_ising.py        (or python3 -m pytest stat/ising/test_ising.py)

It loads stat/ising/ising_mc.viv through vipyr (python/vipyr), which
exports it by the manifest beside it and builds the library, and calls
`run(**kwargs)`. $VIV names the compiler (default target/release/viv).

Every run has a fixed seed, so every number below is reproducible to the
bit. The statistical tolerances are stated beside each check. Most are 4
standard errors: with about 40 such comparisons, a correct program would
fail one of them with probability about 40 * 6.3e-5 = 0.25 percent for a
random seed. The standard errors come from binning, and each check first
asserts that a bin holds at least 10 autocorrelation times, so the bins are
independent enough for the error bars to mean something.

References:
  - exact enumeration of the 4 by 4 lattice: all 2^16 states, summed here in
    numpy, independent of the Viv code;
  - Kaufman's exact partition function of the finite L by L periodic lattice
    (lib/ising.viv's kaufman_energy_heat, which its own test checks against
    the same enumeration), returned by the program as exact_energy and
    exact_heat;
  - Onsager's infinite-lattice energy and T_c = 2 / ln(1 + sqrt 2), and
    Yang's spontaneous magnetization;
  - Binder's cumulant at the crossing, U* = 0.61069 (Kamieniarz and Blote,
    J. Phys. A 26, 201 (1993)).
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("VIV", os.path.join(ROOT, "target", "release", "viv"))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools", "export"))
import vipyr  # noqa: E402
import check  # noqa: E402  (tools/export/check.py: result_hash)

ising = vipyr.load(os.path.join(HERE, "ising_mc.viv"))
TC = 2.0 / math.log(1.0 + math.sqrt(2.0))
NSIG = 4.0          # the statistical tolerance, in standard errors
MIN_BIN_TAUS = 10.0  # a bin must hold this many autocorrelation times


def run(**kw):
    r = ising.run(**kw)
    bins = kw.get("bins", 20)
    blen = kw["sweeps"] // bins
    for tau in (r["tau_energy"], r["tau_mag"]):
        assert blen >= MIN_BIN_TAUS * tau, "bins of %d sweeps are too short for tau = %.2f (%s)" % (blen, tau, kw)
    return r


def near(name, got, err, want, tol):
    """Asserts |got - want| <= tol and prints the comparison."""
    dev = (got - want) / err if err > 0 else float("inf")
    print("  %-34s %.6f +- %.6f   against %.6f   %+.2f sigma   tolerance %.2e" % (name, got, err, want, dev, tol))
    assert abs(got - want) <= tol, (name, got, want, tol)


# ------------------------------------------------------------ exact references

def enumerate_lattice(L, T, J=1.0, h=0.0):
    """Exact averages over all 2^(L*L) states, defined as lib/ising.viv's
    `derived` defines them: energy and specific heat per site, <|m|>,
    beta N (<m^2> - <|m|>^2) and 1 - <m^4> / (3 <m^2>^2)."""
    n = L * L
    s = np.array(list(itertools.product([1.0, -1.0], repeat=n))).reshape(-1, L, L)
    bonds = np.sum(s * np.roll(s, 1, 1), axis=(1, 2)) + np.sum(s * np.roll(s, 1, 2), axis=(1, 2))
    m = s.sum(axis=(1, 2)) / n
    e = (-J * bonds - h * m * n) / n
    beta = 1.0 / T
    w = np.exp(-beta * n * (e - e.min()))
    w /= w.sum()
    avg = lambda x: float(np.sum(w * x))
    return {"energy": avg(e), "heat": beta ** 2 * n * (avg(e * e) - avg(e) ** 2), "mag_abs": avg(abs(m)),
            "chi": beta * n * (avg(m * m) - avg(abs(m)) ** 2), "binder": 1.0 - avg(m ** 4) / (3.0 * avg(m * m) ** 2)}


# ------------------------------------------------------------------- the tests

def test_kaufman_against_enumeration():
    """The program's exact_energy and exact_heat (Kaufman, by five-point
    differences of log Z) against the 4 by 4 enumeration. Tolerance 1e-9
    for the energy and 1e-6 for the specific heat: the differences' rounding
    error is about 1e-12 and 1e-9."""
    print("Kaufman's 4 x 4 lattice against the sum over 65536 states")
    for T in (1.5, 2.0, TC + 0.1, 3.0):
        r = ising.run(size=4, temperature=T, sweeps=2, thermalization=0, bins=2)
        x = enumerate_lattice(4, T)
        print("  T = %.4f  energy %.12f (sum %.12f)  heat %.10f (sum %.10f)" % (T, r["exact_energy"], x["energy"], r["exact_heat"], x["heat"]))
        assert abs(r["exact_energy"] - x["energy"]) < 1e-9
        assert abs(r["exact_heat"] - x["heat"]) < 1e-6


def test_4x4_against_enumeration():
    """Monte Carlo on the 4 by 4 lattice against the exact sums, with and
    without a field, both algorithms. Tolerance: 4 standard errors."""
    runs = [("metropolis", 2.0, 0.0), ("wolff", 2.0, 0.0), ("metropolis", 2.5, 0.1), ("wolff", 2.5, 0.1), ("metropolis", 3.0, 0.0)]
    for alg, T, h in runs:
        r = run(size=4, temperature=T, field=h, algorithm=alg, sweeps=200000, thermalization=1000, bins=50, seed=4)
        x = enumerate_lattice(4, T, h=h)
        print("4 x 4, %s, T = %.1f, h = %.1f" % (alg, T, h))
        for q in ("energy", "heat", "mag_abs", "chi", "binder"):
            near(q, r[q], r[q + "_err"], x[q], NSIG * r[q + "_err"])


def test_energy_against_kaufman_and_onsager():
    """The energy per site at L = 32 against Kaufman's exact value for the
    same lattice (4 standard errors), below, near and above T_c; and
    Kaufman's L = 32 value against Onsager's infinite lattice, which it
    approaches as exp(-L / xi): within 1e-6 away from T_c."""
    for alg, T in (("metropolis", 2.0), ("wolff", 2.25), ("metropolis", 3.0)):
        r = run(size=32, temperature=T, algorithm=alg, sweeps=20000, thermalization=2000, bins=20, seed=32)
        print("L = 32, %s, T = %.2f" % (alg, T))
        near("energy vs Kaufman (L = 32)", r["energy"], r["energy_err"], r["exact_energy"], NSIG * r["energy_err"])
        near("heat vs Kaufman (L = 32)", r["heat"], r["heat_err"], r["exact_heat"], NSIG * r["heat_err"])
        gap = r["exact_energy"] - r["energy_infinite"]
        print("  Kaufman L = 32 minus Onsager infinite: %.3e" % gap)
        if abs(T - TC) > 0.5:
            assert abs(gap) < 1e-6


def test_magnetization_against_yang():
    """<|m|> at L = 64 against Yang's infinite-lattice magnetization, below
    T_c. A finite lattice's <|m|> sits above Yang's value: by 2.1e-3 at
    L = 16 and 6e-4 at L = 32 at T = 2.0, and by 7.4e-5 +- 7.5e-5 at L = 64
    (Wolff, 40,000 sweeps, seed 1). The tolerance is 4 standard errors plus
    an allowance of 2e-4 for that excess at L = 64."""
    for T in (1.8, 2.0):
        r = run(size=64, temperature=T, algorithm="wolff", sweeps=10000, thermalization=500, bins=20, seed=64)
        print("L = 64, Wolff, T = %.1f" % T)
        near("|m| vs Yang (infinite lattice)", r["mag_abs"], r["mag_abs_err"], r["mag_infinite"], NSIG * r["mag_abs_err"] + 2e-4)
    # above T_c Yang's value is 0, and a finite lattice's is not: |m| falls
    # as L^(-1) at T = 3 (patches of size xi, N / xi^2 of them)
    r = ising.run(size=64, temperature=3.0, algorithm="wolff", sweeps=4000, thermalization=200)
    assert r["mag_infinite"] == 0.0 and 0.0 < r["mag_abs"] < 0.05


def test_binder_crossing():
    """Binder's cumulant for L = 8, 16 and 32 (Wolff, 20,000 sweeps per
    point) on a grid of temperatures around T_c. Below T_c it grows with L,
    above it falls, and the curves of L and 2L cross near T_c: each crossing,
    by linear interpolation, is within 0.01 + 3 sigma of T_c (0.01 is 0.44
    percent of T_c, an allowance for the crossing's drift with L), and U
    there is U* = 0.61069 within 0.02 + 3 sigma."""
    temps = [2.20, 2.24, 2.27, 2.30, 2.34]
    U, dU = {}, {}
    for L in (8, 16, 32):
        rs = [run(size=L, temperature=T, algorithm="wolff", sweeps=20000, thermalization=1000, bins=20, seed=100 + L) for T in temps]
        U[L] = np.array([r["binder"] for r in rs])
        dU[L] = np.array([r["binder_err"] for r in rs])
        print("U_%-2d " % L + "  ".join("%.4f(%.4f)" % (u, d) for u, d in zip(U[L], dU[L])))
    for small, large in ((8, 16), (16, 32)):
        d = U[small] - U[large]
        # below T_c the larger lattice is more ordered, above it less
        assert d[0] < 0.0 and d[-1] > 0.0
        k = int(np.nonzero(np.diff(np.sign(d)))[0][0])
        f = d[k] / (d[k] - d[k + 1])
        Tx = temps[k] + f * (temps[k + 1] - temps[k])
        Ux = U[large][k] + f * (U[large][k + 1] - U[large][k])
        # the crossing's error: the difference's error over its slope
        slope = (d[k + 1] - d[k]) / (temps[k + 1] - temps[k])
        sd = math.hypot(dU[small][k], dU[large][k]) * (1 - f) + math.hypot(dU[small][k + 1], dU[large][k + 1]) * f
        sT = sd / abs(slope)
        sU = dU[large][k] * (1 - f) + dU[large][k + 1] * f
        print("  L = %d and %d cross at T = %.4f +- %.4f (T_c %.4f), U = %.4f +- %.4f (U* 0.61069)" % (small, large, Tx, sT, TC, Ux, sU))
        assert abs(Tx - TC) < 0.01 + 3 * sT
        assert abs(Ux - 0.61069) < 0.02 + 3 * sU


def test_wolff_agrees_with_metropolis():
    """The two algorithms at L = 16 near T_c (T = 2.3), where Metropolis is
    slowest: every measurement within 4 combined standard errors."""
    a = run(size=16, temperature=2.3, algorithm="metropolis", sweeps=200000, thermalization=2000, bins=40, seed=9)
    b = run(size=16, temperature=2.3, algorithm="wolff", sweeps=40000, thermalization=1000, bins=40, seed=9)
    print("L = 16, T = 2.3: Metropolis (tau_|m| %.1f sweeps) against Wolff (tau_|m| %.2f)" % (a["tau_mag"], b["tau_mag"]))
    for q in ("energy", "mag_abs", "heat", "chi", "binder"):
        err = math.hypot(a[q + "_err"], b[q + "_err"])
        near(q + " (Metropolis - Wolff)", a[q], err, b[q], NSIG * err)
    near("energy vs Kaufman (Metropolis)", a["energy"], a["energy_err"], a["exact_energy"], NSIG * a["energy_err"])
    near("energy vs Kaufman (Wolff)", b["energy"], b["energy_err"], b["exact_energy"], NSIG * b["energy_err"])


def test_error_bars_match_the_autocorrelation_time():
    """The binning error and the autocorrelation time measure the same
    thing: for n sweeps, err^2 = 2 tau var / n. With 100 bins the binned
    variance scatters by sqrt(2 / 99) = 14 percent, so the ratio
    err^2 n / (2 tau var) must lie in [0.55, 1.45] (3 times that scatter).

    One exception, measured: Wolff's series have a slow tail in their
    autocorrelation. For the energy at L = 16, T = 2.4, rho(t) is about
    0.005 out to t = 30 sweeps and beyond, which the window (it stops at
    t = 9) leaves out, so the window's tau is 1.37 where the bins say 2.0
    (ratio 1.45); |m| gives 1.42. Wolff's upper bound is 2.0.
    The bins are the error bars the program reports; the window's tau only
    sizes them, and the bins of these tests are 100 to 1000 times longer."""
    for alg in ("metropolis", "wolff"):
        r = run(size=16, temperature=2.4, algorithm=alg, sweeps=100000, thermalization=1000, bins=100, seed=21)
        n = len(r["energy_series"])
        for q, series, tau in (("energy", r["energy_series"], r["tau_energy"]), ("mag_abs", np.abs(r["mag_series"]), r["tau_mag"])):
            ratio = r[q + "_err"] ** 2 * n / (2 * tau * np.var(series))
            top = 2.0 if alg == "wolff" else 1.45
            print("  %-10s %-8s tau %.2f  err^2 n / (2 tau var) = %.3f   allowed [0.55, %.2f]" % (alg, q, tau, ratio, top))
            assert 0.55 < ratio < top


def test_bits_and_boundary():
    """The default run gives the recorded result hash (tools/export/check.py
    --record wrote it once the library, the native build and the
    interpreter agreed); a repeat gives the same bits and another seed
    different ones; bad settings are refused with the parameter named."""
    rec = json.load(open(os.path.join(HERE, "ising.exact_hashes.json")))
    h = check.result_hash(ising.run())
    assert h == rec["default"], (h, rec["default"])
    assert check.result_hash(ising.run()) == h
    assert check.result_hash(ising.run(seed=2)) != h
    for kw, word in (({"size": 15}, "even"), ({"algorithm": "wolff", "coupling": -1.0}, "ferromagnetic"),
                     ({"sweeps": 10, "bins": 20}, "bins"), ({"temperature": 0.0}, "temperature"),
                     ({"algorithm": "heatbath"}, "algorithm"), ({"size": 2}, "size")):
        try:
            ising.run(**kw)
        except Exception as e:  # noqa: BLE001  (a ValueError in Python, VivStop from the program)
            assert word in str(e), (kw, str(e))
        else:
            raise AssertionError("not refused: %s" % kw)
    assert check.result_hash(ising.run()) == h


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        print("==", n)
        globals()[n]()
    print("all %d passed" % len(names))

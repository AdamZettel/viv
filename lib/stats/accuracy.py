# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The accuracy map of lib/stats/special.viv (and the distributions built on
it): each function on a grid that reaches into its tails, against mpmath at
50 digits, beside SciPy's and R's values on the same grid.

    python3 lib/stats/accuracy.py            # writes lib/stats/accuracy.json
    python3 lib/stats/accuracy.py --only gammainc,betainc

The error of a value v against the reference r is |v - r| / spacing(r), in
units in the last place of r rounded to double (the reference is rounded
once); a reference that underflows below 1e-300 is left out. An inverse is
held to mpmath's root of the forward function at the same (double) argument,
found by mpmath.findroot at 50 digits from vipyr's value.

Measured with SciPy 1.12, R 4.3.3 (Rscript), mpmath 1.3. R has no erf
family; its column for those functions is empty, and erfc/erfcx/ndtr are
compared through pnorm where R has them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(v, "1")

import mpmath as mp
import numpy as np
import scipy.special as sc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
mp.mp.dps = 50
UNDERFLOW = 1e-300

# the code of each function in lib/stats/api.viv's special_eval
CODES = dict(
    lgamma=0,
    digamma=1,
    erf=2,
    erfc=3,
    erfcx=4,
    erfinv=5,
    erfcinv=6,
    ndtr=7,
    ndtri=8,
    gammainc=9,
    gammaincc=10,
    gammaincinv=11,
    gammainccinv=12,
    betainc=13,
    betaincc=14,
    betaincinv=15,
    betainccinv=16,
    lbeta=17,
)

A_GAMMA = [1e-2, 0.1, 0.5, 1.0, 2.5, 10.0, 50.0, 200.0, 1e3, 1e4]
AB_BETA = [0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 1e3, 1e4]


def grid(name):
    """(a, b, x) arrays for one function."""
    z = np.zeros
    if name == "lgamma":
        x = np.concatenate(
            [np.logspace(-3, 5, 300), np.linspace(0.5, 3.0, 200)]
        )
    elif name == "digamma":
        x = np.concatenate(
            [
                np.logspace(-3, 5, 300),
                np.linspace(0.5, 3.0, 200),
                np.linspace(1.4, 1.52, 60),
            ]
        )
    elif name in ("erf", "erfc"):
        x = np.concatenate(
            [np.linspace(-6, 6, 241), np.linspace(6, 26.5, 120)]
        )
    elif name == "erfcx":
        x = np.concatenate([np.linspace(-3, 6, 181), np.logspace(0.8, 5, 120)])
    elif name == "ndtr":
        x = np.concatenate([np.linspace(-38, 8, 461)])
    elif name in ("erfinv",):
        t = np.logspace(-16, -0.01, 150)
        x = np.concatenate([t, -t, 1.0 - t[t > 1.2e-16]])
    elif name == "erfcinv":
        x = np.concatenate(
            [np.logspace(-300, -0.01, 300), 2.0 - np.logspace(-15, -0.01, 80)]
        )
    elif name == "ndtri":
        t = np.logspace(-300, np.log10(0.499), 300)
        x = np.concatenate(
            [t, 1.0 - t[t > 1.2e-16], [0.5 + 1e-10, 0.5 - 1e-10]]
        )
    elif name in ("gammainc", "gammaincc"):
        a, x = [], []
        for av in A_GAMMA:
            s = np.sqrt(av)
            xs = np.concatenate(
                [
                    av * np.logspace(-3, 0, 12)[:-1],
                    av + s * np.linspace(-6, 6, 25),
                    av + s * np.logspace(0.9, 2.5, 14),
                    av * np.logspace(0.3, 2, 6),
                ]
            )
            xs = xs[xs > 0]
            a += [av] * len(xs)
            x += list(xs)
        a, x = np.array(a), np.array(x)
        return a, z(len(x)), x
    elif name in ("gammaincinv", "gammainccinv"):
        a, x = [], []
        for av in A_GAMMA:
            ps = np.concatenate(
                [
                    np.logspace(-250, -1, 20),
                    np.linspace(0.15, 0.85, 8),
                    1.0 - np.logspace(-1, -15, 10),
                ]
            )
            a += [av] * len(ps)
            x += list(ps)
        return np.array(a), z(len(x)), np.array(x)
    elif name in ("betainc", "betaincc"):
        a, b, x = [], [], []
        for av in AB_BETA:
            for bv in AB_BETA:
                m = av / (av + bv)
                sd = np.sqrt(av * bv / ((av + bv) ** 2 * (av + bv + 1)))
                xs = np.concatenate(
                    [
                        np.logspace(-6, 0, 8)[:-1] * m,
                        m + sd * np.linspace(-5, 5, 11),
                        1.0 - np.logspace(-6, 0, 8)[:-1] * (1 - m),
                    ]
                )
                xs = xs[(xs > 0) & (xs < 1)]
                a += [av] * len(xs)
                b += [bv] * len(xs)
                x += list(xs)
        return np.array(a), np.array(b), np.array(x)
    elif name in ("betaincinv", "betainccinv"):
        a, b, x = [], [], []
        for av in AB_BETA:
            for bv in AB_BETA:
                ps = np.concatenate(
                    [
                        np.logspace(-100, -1, 8),
                        [0.3, 0.5, 0.7],
                        1.0 - np.logspace(-1, -12, 5),
                    ]
                )
                a += [av] * len(ps)
                b += [bv] * len(ps)
                x += list(ps)
        return np.array(a), np.array(b), np.array(x)
    elif name == "lbeta":
        aa, bb = np.meshgrid(np.logspace(-2, 5, 20), np.logspace(-2, 5, 20))
        return aa.ravel(), bb.ravel(), z(aa.size)
    else:
        raise KeyError(name)
    return z(len(x)), z(len(x)), x


def mp_forward(name, a, b, x):
    """The function at mpmath precision (an mpf)."""
    a, b, x = mp.mpf(a), mp.mpf(b), mp.mpf(x)
    if name == "lgamma":
        return mp.log(abs(mp.gamma(x)))
    if name == "digamma":
        return mp.digamma(x)
    if name == "erf":
        return mp.erf(x)
    if name == "erfc":
        return mp.erfc(x)
    if name == "erfcx":
        return mp.erfc(x) * mp.exp(x * x)
    if name == "ndtr":
        return mp.ncdf(x)
    if name == "gammainc":
        return mp.gammainc(a, 0, x, regularized=True)
    if name == "gammaincc":
        return mp.gammainc(a, x, mp.inf, regularized=True)
    if name in ("betainc", "betaincc"):
        lower = name == "betainc"
        try:
            if lower:
                return mp.betainc(a, b, 0, x, regularized=True)
            # 1 - I_x(a, b) = I_(1-x)(b, a), with 1 - x exact in mpmath
            return mp.betainc(b, a, 0, 1 - x, regularized=True)
        except ValueError:
            return mp_beta_cf(a, b, x, lower)
    if name == "lbeta":
        return mp.log(mp.beta(a, b))
    raise KeyError(name)


def mp_beta_cf(a, b, x, lower):
    """I_x(a, b) (lower) or 1 - I_x(a, b) by the continued fraction at
    mpmath precision, on the side of the mean where it converges, for the
    parameters where mpmath.betainc's hypergeometric series gives up."""

    def direct(a, b, x):
        # x^a (1-x)^b / (a B(a, b)) / (1 + d1 / (1 + d2 / (1 + ...))) (DLMF 8.17.22),
        # evaluated from the bottom with N and 2N terms until the two agree
        def d(j):
            m = j // 2
            if j % 2 == 0:
                return m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            return -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))

        def cf(n):
            t = mp.mpf(1)
            for j in range(n, 0, -1):
                t = 1 + d(j) / t
            return 1 / t

        n, prev = 64, cf(32)
        while True:
            f = cf(n)
            if abs(f - prev) < mp.mpf(10) ** -48 * abs(f):
                break
            if n > 4000000:
                raise ValueError("continued fraction did not converge")
            n, prev = 2 * n, f
        pre = mp.exp(
            a * mp.log(x)
            + b * mp.log1p(-x)
            - mp.log(a)
            - mp.log(mp.beta(a, b))
        )
        return pre * f

    if x < (a + 1) / (a + b + 2):
        v = direct(a, b, x)
        return v if lower else 1 - v
    v = direct(b, a, 1 - x)
    return 1 - v if lower else v


# an inverse: (the forward function, whether the root's forward value is p)
INVERSES = {
    "erfinv": "erf",
    "erfcinv": "erfc",
    "ndtri": "ndtr",
    "gammaincinv": "gammainc",
    "gammainccinv": "gammaincc",
    "betaincinv": "betainc",
    "betainccinv": "betaincc",
}


def mp_inverse(name, a, b, p, start):
    fwd = INVERSES[name]
    p = mp.mpf(p)
    if (
        not np.isfinite(start)
        or start <= 0
        and name not in ("erfinv", "erfcinv", "ndtri")
    ):
        return None
    f = lambda t: mp_forward(fwd, a, b, t) - p
    try:
        # the log of the root for the positive ones, so that the steps keep it positive
        if name in ("betaincinv", "betainccinv"):
            # x = 1 / (1 + e^-u) keeps every step inside (0, 1)
            sig = lambda u: 1 / (1 + mp.exp(-u))
            g = lambda u: mp_forward(fwd, a, b, sig(u)) - p
            s0 = mp.mpf(start)
            r = sig(
                mp.findroot(
                    g,
                    mp.log(s0 / (1 - s0)),
                    tol=mp.mpf(10) ** -45,
                    maxsteps=200,
                )
            )
        elif name in ("gammaincinv", "gammainccinv"):
            g = lambda u: mp_forward(fwd, a, b, mp.exp(u)) - p
            r = mp.exp(
                mp.findroot(
                    g,
                    mp.log(mp.mpf(start)),
                    tol=mp.mpf(10) ** -45,
                    maxsteps=200,
                )
            )
        else:
            r = mp.findroot(
                f, mp.mpf(start), tol=mp.mpf(10) ** -45, maxsteps=200
            )
    except Exception:
        return None
    # a root is accepted when the forward function there is p to 40 digits
    if abs(f(r)) > mp.mpf(10) ** -40 * max(abs(p), mp.mpf(10) ** -300):
        return None
    return r


def ulps(v, r):
    """|v - r| in units in the last place of r rounded to double; inf when
    v is not finite."""
    v = np.asarray(v, dtype=float)
    rd = np.array([float(t) for t in r])
    out = np.full(v.shape, np.inf)
    ok = np.isfinite(v)
    sp = np.spacing(np.abs(rd))
    err = np.array(
        [
            float(abs(mp.mpf(float(vi)) - ri)) if f else np.inf
            for vi, ri, f in zip(v, r, ok)
        ]
    )
    out[ok] = err[ok] / sp[ok]
    return out


def scipy_values(name, a, b, x):
    f = {
        "lgamma": lambda: sc.gammaln(x),
        "digamma": lambda: sc.psi(x),
        "erf": lambda: sc.erf(x),
        "erfc": lambda: sc.erfc(x),
        "erfcx": lambda: sc.erfcx(x),
        "erfinv": lambda: sc.erfinv(x),
        "erfcinv": lambda: sc.erfcinv(x),
        "ndtr": lambda: sc.ndtr(x),
        "ndtri": lambda: sc.ndtri(x),
        "gammainc": lambda: sc.gammainc(a, x),
        "gammaincc": lambda: sc.gammaincc(a, x),
        "gammaincinv": lambda: sc.gammaincinv(a, x),
        "gammainccinv": lambda: sc.gammainccinv(a, x),
        "betainc": lambda: sc.betainc(a, b, x),
        "betaincinv": lambda: sc.betaincinv(a, b, x),
        "lbeta": lambda: sc.betaln(a, b),
    }.get(name)
    return None if f is None else f()


R_EXPR = {
    "lgamma": "lgamma(x)",
    "digamma": "digamma(x)",
    "ndtr": "pnorm(x)",
    "ndtri": "qnorm(x)",
    "erfc": "2 * pnorm(-sqrt(2) * x)",
    "gammainc": "pgamma(x, a)",
    "gammaincc": "pgamma(x, a, lower.tail = FALSE)",
    "gammaincinv": "qgamma(x, a)",
    "gammainccinv": "qgamma(x, a, lower.tail = FALSE)",
    "betainc": "pbeta(x, a, b)",
    "betaincc": "pbeta(x, a, b, lower.tail = FALSE)",
    "betaincinv": "qbeta(x, a, b)",
    "betainccinv": "qbeta(x, a, b, lower.tail = FALSE)",
    "lbeta": "lbeta(a, b)",
}


def r_values(name, a, b, x):
    # The inputs and outputs cross as raw little-endian doubles (writeBin /
    # readBin), not text: R's reading of a 17-digit numeral is not correctly
    # rounded (read.table turns 0.99999999999999956 into 0.99999999999999967),
    # and near p = 1 an inverse such as qnorm turns that one-ulp shift of the
    # input into 1e13 ulps of output.
    expr = R_EXPR.get(name)
    if expr is None:
        return None
    n = len(x)
    with tempfile.TemporaryDirectory() as d:
        np.column_stack([a, b, x]).astype("<f8").T.tofile(os.path.join(d, "in.bin"))
        code = (
            "v <- readBin('%s/in.bin', 'double', n = %d, size = 8, endian = 'little'); "
            "a <- v[1:%d]; b <- v[%d + 1:%d]; x <- v[2 * %d + 1:%d]; "
            "v <- suppressWarnings(%s); "
            "writeBin(as.double(v), '%s/out.bin', size = 8, endian = 'little')"
            % (d, 3 * n, n, n, n, n, n, expr, d)
        )
        subprocess.run(
            ["Rscript", "-e", code], check=True, capture_output=True
        )
        return np.fromfile(os.path.join(d, "out.bin"), dtype="<f8")


def run(name, mod):
    a, b, x = grid(name)
    t0 = time.time()
    viv = mod.special_eval(float(CODES[name]), a, b, x)
    if name in INVERSES:
        refs = [
            mp_inverse(name, ai, bi, xi, vi)
            for ai, bi, xi, vi in zip(a, b, x, viv)
        ]
    else:
        refs = [mp_forward(name, ai, bi, xi) for ai, bi, xi in zip(a, b, x)]
    keep = np.array(
        [r is not None and abs(r) > UNDERFLOW and mp.isfinite(r) for r in refs]
    )
    a, b, x, viv = a[keep], b[keep], x[keep], viv[keep]
    refs = [r for r, k in zip(refs, keep) if k]
    out = dict(
        points=int(keep.sum()), seconds_reference=round(time.time() - t0, 1)
    )
    cols = {
        "vipyr": viv,
        "scipy": scipy_values(name, a, b, x),
        "R": r_values(name, a, b, x),
    }
    for who, vals in cols.items():
        if vals is None:
            out[who] = None
            continue
        e = ulps(vals, refs)
        i = int(np.argmax(e))
        out[who] = dict(
            max_ulps=float(e[i]),
            median_ulps=float(np.median(e)),
            frac_le_4=float(np.mean(e <= 4.0)),
            worst_at=dict(a=float(a[i]), b=float(b[i]), x=float(x[i])),
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--out", default=str(Path(__file__).with_name("accuracy.json"))
    )
    args = ap.parse_args(argv)
    from vipyr import _core

    mod = _core.load(str(Path(__file__).with_name("api.viv")))
    names = [n for n in CODES if not args.only or n in args.only.split(",")]
    path = Path(args.out)
    result = json.loads(path.read_text()) if path.exists() else {}
    result.setdefault(
        "about",
        "lib/stats/accuracy.py: error in ulps against mpmath 1.3 at 50 digits; "
        "SciPy 1.12, R 4.3.3",
    )
    fns = result.setdefault("functions", {})
    for n in names:
        fns[n] = run(n, mod)
        r = fns[n]
        print(
            "%-13s %5d pts  vipyr %9.3g  scipy %9s  R %9s"
            % (
                n,
                r["points"],
                r["vipyr"]["max_ulps"],
                "%.3g" % r["scipy"]["max_ulps"] if r["scipy"] else "-",
                "%.3g" % r["R"]["max_ulps"] if r["R"] else "-",
            ),
            flush=True,
        )
        path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

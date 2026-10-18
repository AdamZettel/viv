# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.stats against SciPy, mpmath, statsmodels and R's recorded values.

The full accuracy map (lib/stats/accuracy.py), NIST's certified regressions
(lib/stats/nist_lls.py) and R's glm (lib/stats/glm_check.py) are scripts
with committed results; this file checks a sample of each fast enough for
every run.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import vipyr as vp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sp = pytest.importorskip("scipy.stats")
mp = pytest.importorskip("mpmath")

CONTINUOUS = [
    ("norm", ()),
    ("lognorm", (0.7,)),
    ("t", (4.5,)),
    ("chi2", (3.0,)),
    ("f", (5.0, 12.0)),
    ("gamma", (2.5,)),
    ("beta", (2.0, 5.0)),
    ("expon", ()),
    ("weibull_min", (1.7,)),
    ("cauchy", ()),
    ("logistic", ()),
    ("uniform", ()),
    ("laplace", ()),
    ("pareto", (2.5,)),
]
DISCRETE = [
    ("binom", (20, 0.3)),
    ("poisson", (4.5,)),
    ("nbinom", (5, 0.4)),
    ("geom", (0.25,)),
    ("hypergeom", (40, 12, 9)),
]


@pytest.mark.parametrize("name,shapes", CONTINUOUS)
def test_continuous_against_scipy(name, shapes):
    ours, ref = getattr(vp.stats, name), getattr(sp, name)
    q = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6])
    x = ref.ppf(q, *shapes, loc=0.5, scale=2.0)
    kw = dict(loc=0.5, scale=2.0)
    for m in ("pdf", "cdf", "sf", "logpdf"):
        np.testing.assert_allclose(
            getattr(ours, m)(x, *shapes, **kw),
            getattr(ref, m)(x, *shapes, **kw),
            rtol=1e-9,
            err_msg=m,
        )
    # SciPy 1.12's t quantile is off by up to 1e-9 in the tails (mpmath: ours
    # is right to the last digit there, test_tails_against_mpmath); the
    # round trip through our own cdf holds to rounding
    np.testing.assert_allclose(ours.ppf(q, *shapes, **kw), x, rtol=1e-8)
    np.testing.assert_allclose(
        ours.isf(1 - q, *shapes, **kw),
        ref.isf(1 - q, *shapes, **kw),
        rtol=1e-8,
    )
    # (without loc, which would round away the digits of a small quantile)
    np.testing.assert_allclose(
        ours.cdf(ours.ppf(q, *shapes), *shapes), q, rtol=1e-9
    )


@pytest.mark.parametrize("name,shapes", DISCRETE)
def test_discrete_against_scipy(name, shapes):
    ours, ref = getattr(vp.stats, name), getattr(sp, name)
    k = np.arange(0, 25)
    np.testing.assert_allclose(
        ours.pmf(k, *shapes), ref.pmf(k, *shapes), rtol=1e-12, atol=1e-300
    )
    np.testing.assert_allclose(
        ours.cdf(k, *shapes), ref.cdf(k, *shapes), rtol=1e-11
    )
    np.testing.assert_allclose(
        ours.sf(k, *shapes), ref.sf(k, *shapes), rtol=1e-10, atol=1e-300
    )
    q = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_array_equal(ours.ppf(q, *shapes), ref.ppf(q, *shapes))


def test_tails_against_mpmath():
    mp.mp.dps = 50
    # deep tails SciPy itself gets right only to a few digits
    v = vp.stats.norm.cdf(-37.5)
    assert abs(v / float(mp.ncdf(-37.5)) - 1) < 1e-14
    v = vp.stats.t.ppf(0.975, 10)
    ref = mp.findroot(
        lambda t: mp.betainc(5, 0.5, 0, 10 / (10 + t * t), regularized=True)
        / 2
        - mp.mpf(0.025),
        2.2,
    )
    assert abs(v / float(ref) - 1) < 1e-15
    v = vp.stats.gammaincc(50.0, 120.0)
    assert (
        abs(v / float(mp.gammainc(50, 120, mp.inf, regularized=True)) - 1)
        < 1e-12
    )


def test_draws_are_numpy_uniforms_inverted():
    for name, shapes in [("norm", ()), ("gamma", (2.5,)), ("poisson", (3.0,))]:
        d = getattr(vp.stats, name)
        u = np.random.default_rng(7).random(200)
        np.testing.assert_array_equal(
            d.rvs(*shapes, size=200, seed=7), d.ppf(u, *shapes)
        )


def test_describe_keeps_the_digits_of_a_shifted_sample():
    x = 1e9 + np.array([4.0, 7.0, 13.0, 16.0])
    for method in vp.stats.VARIANCE_METHODS:
        r = vp.stats.describe(x, method=method)
        assert r.mean == 1e9 + 10 and abs(r.variance - 30.0) < 1e-6
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, size=500)
    r = vp.stats.describe(y)
    s = sp.describe(y)
    np.testing.assert_allclose(
        [r.mean, r.variance, r.skewness, r.kurtosis],
        [s.mean, s.variance, s.skewness, s.kurtosis],
        rtol=1e-13,
    )
    assert r.check()


def test_quantile_types_against_r():
    x = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0]
    # R 4.3.3: quantile(x, c(0.1, 0.25, 0.5, 0.9), type = t)
    r_values = {
        1: [1, 1, 2, 8],
        2: [1, 1.5, 4.5, 8],
        3: [1, 1, 2, 8],
        4: [1, 1, 2, 8],
        5: [1, 1.5, 4.5, 8],
        6: [1, 1.25, 4.5, 8],
        7: [1, 1.75, 4.5, 8],
        8: [1, 1.4166666666666665, 4.4999999999999956, 8],
        9: [1, 1.4375, 4.5, 8],
    }
    for t, want in r_values.items():
        np.testing.assert_allclose(
            vp.stats.quantile(x, [0.1, 0.25, 0.5, 0.9], type=t).values,
            want,
            rtol=1e-15,
        )
    np.testing.assert_allclose(
        vp.stats.quantile(np.arange(11.0), [0.3, 0.77]).values,
        np.quantile(np.arange(11.0), [0.3, 0.77]),
        rtol=1e-15,
    )


def test_ols_against_statsmodels_and_nist():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(60), rng.normal(size=(60, 3))])
    y = X @ [1.0, 2.0, -1.0, 0.5] + rng.normal(size=60)
    w = rng.uniform(0.5, 2.0, 60)
    for weights in (None, w):
        r = vp.stats.ols(y, X, weights=weights)
        f = (
            sm.OLS(y, X) if weights is None else sm.WLS(y, X, weights=w)
        ).fit()
        np.testing.assert_allclose(r.coef, f.params, rtol=1e-12)
        np.testing.assert_allclose(r.bse, f.bse, rtol=1e-10)
        np.testing.assert_allclose(
            [r.rsquared, r.rsquared_adj, r.fvalue, r.llf, r.aic, r.bic],
            [f.rsquared, f.rsquared_adj, f.fvalue, f.llf, f.aic, f.bic],
            rtol=1e-10,
        )
        np.testing.assert_allclose(r.pvalues, f.pvalues, rtol=1e-8)
    f0 = sm.OLS(y, X).fit()
    for solver in vp.stats.LSTSQ_SOLVERS:
        np.testing.assert_allclose(
            vp.stats.ols(y, X, solver=solver).coef, f0.params, rtol=1e-12
        )
    sys.path.insert(0, str(ROOT / "lib" / "stats"))
    import nist_lls

    y, X, cert = nist_lls.read("Longley")
    r = vp.stats.ols(y, X)
    assert nist_lls.lre(r.coef, cert["coef"]).min() > 12.5
    assert nist_lls.lre(r.bse, cert["se"]).min() > 12.5
    with pytest.raises(ValueError):
        vp.stats.ols(y, X, solver="lu")


def test_rank_deficient_columns_are_dropped():
    x = np.arange(10.0)
    X = np.column_stack([np.ones(10), x, 2 * x])
    r = vp.stats.ols(x**2, X)
    assert r.rank == 2 and np.isnan(r.coef).sum() == 1


def test_glm_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    yb = (rng.random(200) < 1 / (1 + np.exp(-(X @ [0.3, 1.0, -0.7])))).astype(
        float
    )
    yp = rng.poisson(np.exp(X @ [0.5, 0.3, -0.2])).astype(float)
    for y, fam, link, smfam in [
        (yb, "binomial", "logit", sm.families.Binomial()),
        (
            yb,
            "binomial",
            "probit",
            sm.families.Binomial(sm.families.links.Probit()),
        ),
        (yp, "poisson", "log", sm.families.Poisson()),
    ]:
        g = vp.stats.glm(y, X, family=fam, link=link, epsilon=1e-14, maxit=100)
        f = sm.GLM(y, X, family=smfam).fit(tol=1e-14)
        # statsmodels stops IRLS on its own rule; the deviances agree to rounding
        np.testing.assert_allclose(g.coef, f.params, rtol=1e-7)
        np.testing.assert_allclose(g.deviance, f.deviance, rtol=1e-11)
        assert g.check()
    # Newton reaches the same probit fit; its observed-information SEs match
    # statsmodels' observed Hessian
    g = vp.stats.glm(
        yb,
        X,
        family="binomial",
        link="probit",
        fit="newton",
        cov="observed",
        epsilon=1e-14,
    )
    f = sm.GLM(
        yb, X, family=sm.families.Binomial(sm.families.links.Probit())
    ).fit(method="newton", tol=1e-14)
    np.testing.assert_allclose(g.coef, f.params, rtol=1e-8)
    np.testing.assert_allclose(g.bse, f.bse, rtol=1e-6)


def test_separation_warns_as_r():
    X = np.column_stack([np.ones(6), np.arange(1.0, 7.0)])
    y = np.array([0, 0, 0, 1, 1, 1.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = vp.stats.glm(y, X, family="binomial")
    assert (
        "glm.fit: fitted probabilities numerically 0 or 1 occurred"
        in g.warnings
    )
    assert any(issubclass(x.category, vp.stats.GLMWarning) for x in w)
    assert g.iterations == 25 or g.converged

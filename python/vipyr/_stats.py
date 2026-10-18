# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.stats: distributions, special functions, descriptive statistics,
linear models and GLMs, computed by lib/stats (Viv).

lib/stats/api.viv is compiled once through vipyr's cache; every function
here broadcasts its arguments with NumPy, calls one of its entry points on
flat float64 arrays, and names what comes back.

    vp.stats.norm.cdf(x)                    SciPy's names and parameters
    vp.stats.t.ppf(0.975, 10)
    vp.stats.binom.pmf(3, 10, 0.5)
    vp.stats.gammainc(a, x)                 the special functions underneath
    vp.stats.describe(x), vp.stats.quantile(x, q, type=7)
    vp.stats.ols(y, X).summary()            statsmodels' table
    vp.stats.glm(y, X, family="binomial", link="probit").summary()

Components that can be swapped are named slots, each with one canonical
choice and the alternatives registered beside it:

    LSTSQ_SOLVERS   the least-squares solver: "qr" (pivoted Householder QR,
                    the default), "cholesky" (normal equations), "svd"
    GLM_FITTERS     the GLM fitting method: "irls" (Fisher scoring, R's
                    glm.fit, the default), "newton" (observed information)
    VARIANCE_METHODS  "two_pass" (corrected two-pass, the default), "welford"
"""
from __future__ import annotations

import math
import threading
import warnings

import numpy as np

from . import _core, _spec

API = "lib/stats/api.viv"

_lock = threading.Lock()
_mod: dict = {}


def _api(fast=False):
    with _lock:
        if fast not in _mod:
            _mod[fast] = _core.load(_spec.source(API), fast=fast)
        return _mod[fast]


def _f(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


# ------------------------------------------------------------ special functions

# name -> (code in lib/stats/api.viv's special_eval, the arguments it takes)
SPECIAL = {
    "lgamma": (0, "x"),
    "digamma": (1, "x"),
    "erf": (2, "x"),
    "erfc": (3, "x"),
    "erfcx": (4, "x"),
    "erfinv": (5, "x"),
    "erfcinv": (6, "x"),
    "ndtr": (7, "x"),
    "ndtri": (8, "x"),
    "gammainc": (9, "ax"),
    "gammaincc": (10, "ax"),
    "gammaincinv": (11, "ax"),
    "gammainccinv": (12, "ax"),
    "betainc": (13, "abx"),
    "betaincc": (14, "abx"),
    "betaincinv": (15, "abx"),
    "betainccinv": (16, "abx"),
    "lbeta": (17, "ab"),
    "log1p": (18, "x"),
    "expm1": (19, "x"),
}


def _special(name, *args):
    code, takes = SPECIAL[name]
    arrs = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in args])
    shape = arrs[0].shape
    named = dict(zip(takes, arrs))
    z = np.zeros(shape)
    a, b, x = (_f(named.get(k, z)) for k in "abx")
    out = _api().special_eval(float(code), a, b, x).reshape(shape)
    return out[()] if out.ndim == 0 else out


def _make_special(name):
    takes = SPECIAL[name][1]

    def fn(*args):
        if len(args) != len(takes):
            raise TypeError(
                "vp.stats.%s takes %d arguments (%s)"
                % (name, len(takes), ", ".join(takes))
            )
        return _special(name, *args)

    fn.__name__ = name
    fn.__doc__ = (
        "%s(%s), lib/stats/special.viv's `%s`; accuracy against mpmath in lib/stats/accuracy.json."
        % (name, ", ".join(takes), name)
    )
    return fn


SPECIAL_FUNCTIONS = {n: _make_special(n) for n in SPECIAL}

# ------------------------------------------------------------ distributions

PDF, LOGPDF, CDF, SF, PPF, ISF = range(6)


class Distribution:
    """One family, with SciPy's method names and parameters:
    pdf/logpdf (pmf/logpmf for a discrete family), cdf, sf, ppf, isf and
    rvs, each taking the shape parameters (positionally or by name), then
    loc and, for a continuous family, scale."""

    def __init__(self, name, code, shapes, discrete, valid):
        self.name, self.code, self.shapes, self.discrete, self._valid = (
            name,
            code,
            shapes,
            discrete,
            valid,
        )

    def __repr__(self):
        return "<vp.stats.%s: %s, shapes (%s)>" % (
            self.name,
            "discrete" if self.discrete else "continuous",
            ", ".join(self.shapes),
        )

    def _args(self, args, kw):
        args = list(args)
        vals = []
        for s in self.shapes:
            if args:
                vals.append(args.pop(0))
            elif s in kw:
                vals.append(kw.pop(s))
            else:
                raise TypeError(
                    "vp.stats.%s: missing shape parameter %r" % (self.name, s)
                )
        loc = kw.pop("loc", args.pop(0) if args else 0.0)
        scale = (
            1.0
            if self.discrete
            else kw.pop("scale", args.pop(0) if args else 1.0)
        )
        if args or kw:
            raise TypeError(
                "vp.stats.%s: unexpected arguments %r %r"
                % (self.name, args, list(kw))
            )
        return vals, loc, scale

    def _eval(self, what, x, args, kw):
        vals, loc, scale = self._args(args, kw)
        arrs = np.broadcast_arrays(
            *[np.asarray(v, dtype=float) for v in [x, loc, scale] + vals]
        )
        shape = arrs[0].shape
        x, loc, scale = arrs[:3]
        s = list(arrs[3:]) + [np.zeros(shape)] * (3 - len(vals))
        z = x if what in (PPF, ISF) else (x - loc) / scale
        out = (
            _api()
            .dist_eval(
                float(self.code),
                float(what),
                _f(z),
                _f(s[0]),
                _f(s[1]),
                _f(s[2]),
            )
            .reshape(shape)
        )
        if what in (PPF, ISF):
            out = loc + scale * out
        elif what == PDF:
            out = out / scale
        elif what == LOGPDF:
            out = out - np.log(scale)
        bad = ~self._valid(*arrs[3:]) | (scale <= 0)
        if np.any(bad):
            out = np.where(bad, np.nan, out)
        return out[()] if out.ndim == 0 else out

    def pdf(self, x, *args, **kw):
        return self._eval(PDF, x, args, kw)

    def logpdf(self, x, *args, **kw):
        return self._eval(LOGPDF, x, args, kw)

    pmf = pdf
    logpmf = logpdf

    def cdf(self, x, *args, **kw):
        return self._eval(CDF, x, args, kw)

    def sf(self, x, *args, **kw):
        return self._eval(SF, x, args, kw)

    def ppf(self, q, *args, **kw):
        return self._eval(PPF, q, args, kw)

    def isf(self, q, *args, **kw):
        return self._eval(ISF, q, args, kw)

    def rvs(self, *args, size=1, seed=0, **kw):
        """Draws by inversion: ppf of `np.random.default_rng(seed).random(size)`,
        bit for bit (lib/nprandom's PCG64 in Viv)."""
        vals, loc, scale = self._args(args, kw)
        n = int(np.prod(size))
        s = [float(v) for v in vals] + [0.0] * (3 - len(vals))
        out = _api().dist_rvs(
            float(self.code), s[0], s[1], s[2], float(n), float(seed)
        )
        return (loc + scale * out).reshape(size)


def _pos(*xs):
    return (
        np.logical_and.reduce([np.asarray(x) > 0 for x in xs])
        if xs
        else np.True_
    )


def _prob(p):
    return (np.asarray(p) >= 0) & (np.asarray(p) <= 1)


DISTRIBUTIONS = {
    "norm": (0, (), False, lambda: np.True_),
    "lognorm": (1, ("s",), False, _pos),
    "t": (2, ("df",), False, _pos),
    "chi2": (3, ("df",), False, _pos),
    "f": (4, ("dfn", "dfd"), False, _pos),
    "gamma": (5, ("a",), False, _pos),
    "beta": (6, ("a", "b"), False, _pos),
    "expon": (7, (), False, lambda: np.True_),
    "weibull_min": (8, ("c",), False, _pos),
    "cauchy": (9, (), False, lambda: np.True_),
    "logistic": (10, (), False, lambda: np.True_),
    "uniform": (11, (), False, lambda: np.True_),
    "laplace": (12, (), False, lambda: np.True_),
    "pareto": (13, ("b",), False, _pos),
    "binom": (
        20,
        ("n", "p"),
        True,
        lambda n, p: (np.asarray(n) >= 0) & _prob(p),
    ),
    "poisson": (21, ("mu",), True, lambda mu: np.asarray(mu) >= 0),
    "nbinom": (
        22,
        ("n", "p"),
        True,
        lambda n, p: (np.asarray(n) > 0)
        & (np.asarray(p) > 0)
        & (np.asarray(p) <= 1),
    ),
    "geom": (
        23,
        ("p",),
        True,
        lambda p: (np.asarray(p) > 0) & (np.asarray(p) <= 1),
    ),
    "hypergeom": (
        24,
        ("M", "n", "N"),
        True,
        lambda M, n, N: (np.asarray(n) <= np.asarray(M))
        & (np.asarray(N) <= np.asarray(M))
        & (np.asarray(n) >= 0)
        & (np.asarray(N) >= 0),
    ),
}

FAMILY_OBJECTS = {
    n: Distribution(n, c, s, d, v) for n, (c, s, d, v) in DISTRIBUTIONS.items()
}

# ------------------------------------------------------------ descriptive statistics

VARIANCE_METHODS = {"two_pass": 0.0, "welford": 1.0}


def _describe(x, ddof=1, method="two_pass"):
    if method not in VARIANCE_METHODS:
        raise ValueError(
            "vp.stats.describe: method must be one of %s"
            % ", ".join(VARIANCE_METHODS)
        )
    x = _f(x)
    if x.size < 2:
        raise ValueError("vp.stats.describe: x needs at least two values")
    n, m, v, g1, g2 = _api().describe(x, float(ddof), VARIANCE_METHODS[method])
    return dict(
        nobs=int(n),
        minmax=(float(x.min()), float(x.max())),
        mean=m,
        variance=v,
        skewness=g1,
        kurtosis=g2,
        _problem="describe:%s:%d" % (method, ddof),
    )


def _quantile(x, q, type=7):
    if type not in range(1, 10):
        raise ValueError(
            "vp.stats.quantile: type must be 1 to 9 (Hyndman and Fan)"
        )
    qa = np.asarray(q, dtype=float)
    if np.any((qa < 0) | (qa > 1)):
        raise ValueError("vp.stats.quantile: q must lie in [0, 1]")
    out = _api().quantiles(_f(x), _f(qa), float(type)).reshape(qa.shape)
    return dict(
        values=out[()] if out.ndim == 0 else out,
        type=int(type),
        _problem="quantile:%d" % type,
    )


# ------------------------------------------------------------ linear models

LSTSQ_SOLVERS = {"qr": 0.0, "cholesky": 1.0, "svd": 2.0}


def _names(X, names, k):
    if names is not None:
        return list(names)
    cols = getattr(X, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]
    return ["x%d" % i for i in range(k)]


def _has_const(X):
    return bool(
        np.any(
            [np.ptp(X[:, j]) == 0 and X[0, j] != 0 for j in range(X.shape[1])]
        )
    )


def _ols(y, X, weights=None, solver="qr", refine=True, tol=1e-10, names=None):
    if solver not in LSTSQ_SOLVERS:
        raise ValueError(
            "vp.stats.ols: solver must be one of %s" % ", ".join(LSTSQ_SOLVERS)
        )
    nm = _names(X, names, np.asarray(X).shape[1] if np.ndim(X) == 2 else 1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = _f(y)
    n, k = X.shape
    if len(y) != n:
        raise ValueError("vp.stats.ols: y has %d rows, X has %d" % (len(y), n))
    w = np.ones(n) if weights is None else _f(weights)
    sw = np.sqrt(w)
    Xw = np.ascontiguousarray(X * sw[:, None])
    yw = y * sw
    b, rank, piv, C, rw = _api().ols_fit(
        Xw, yw, LSTSQ_SOLVERS[solver], float(tol), 1.0 if refine else 0.0
    )
    rank = int(rank)
    # a dropped column's coefficient is NaN, as R's NA
    b = np.where(np.isnan(np.diag(C)), np.nan, b)
    kc = 1 if _has_const(X) else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return _ols_stats(y, X, w, b, rank, C, rw, kc, nm, solver, weights)


def _ols_stats(y, X, w, b, rank, C, rw, kc, nm, solver, weights):
    n, k = X.shape
    nobs = n
    df_resid = nobs - rank
    df_model = rank - kc
    ssr = float(np.sum(rw * rw))
    if kc:
        ybar = np.sum(w * y) / np.sum(w)
        tss = float(np.sum(w * (y - ybar) ** 2))
    else:
        tss = float(np.sum(w * y * y))
    scale = np.float64(ssr) / df_resid
    bse = np.sqrt(scale * np.diag(C))
    tv = b / bse
    pv = 2.0 * FAMILY_OBJECTS["t"].sf(np.abs(tv), df_resid)
    q = FAMILY_OBJECTS["t"].ppf(0.975, df_resid)
    r2 = 1.0 - ssr / tss
    llf = -nobs / 2.0 * (
        np.log(2 * np.pi) + np.log(np.float64(ssr) / nobs) + 1.0
    ) + 0.5 * float(np.sum(np.log(w)))
    fval = np.float64(tss - ssr) / df_model / scale if df_model > 0 else np.nan
    return dict(
        coef=b,
        bse=bse,
        tvalues=tv,
        pvalues=pv,
        conf_int=np.column_stack([b - q * bse, b + q * bse]),
        rsquared=r2,
        rsquared_adj=1.0 - (nobs - kc) / df_resid * (1.0 - r2),
        fvalue=fval,
        f_pvalue=FAMILY_OBJECTS["f"].sf(fval, df_model, df_resid)
        if df_model > 0
        else np.nan,
        ssr=ssr,
        centered_tss=tss,
        scale=scale,
        resid_sd=float(np.sqrt(scale)),
        df_model=df_model,
        df_resid=df_resid,
        nobs=nobs,
        rank=rank,
        llf=llf,
        aic=-2 * llf + 2 * (df_model + kc),
        bic=-2 * llf + np.log(nobs) * (df_model + kc),
        resid=y - X @ np.nan_to_num(b),
        fittedvalues=X @ np.nan_to_num(b),
        cov_params=C * scale,
        names=nm,
        solver=solver,
        weighted=weights is not None,
        k_constant=kc,
        _problem="ols:%s:%d:%d" % (solver, n, k),
    )


# ------------------------------------------------------------ GLMs

FAMILIES = {"gaussian": 0, "binomial": 1, "poisson": 2, "gamma": 3}
LINKS = {"identity": 0, "logit": 1, "probit": 2, "log": 3, "inverse": 4}
CANONICAL = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
}
ALLOWED = {
    "gaussian": {"identity"},
    "binomial": {"logit", "probit"},
    "poisson": {"log"},
    "gamma": {"inverse", "log"},
}
GLM_FITTERS = {"irls": "glm_fit", "newton": "glm_fit_newton"}
EPS = np.finfo(float).eps


class GLMWarning(UserWarning):
    """R's glm.fit warnings: no convergence, fitted values numerically 0 or 1."""


def _glm(
    y,
    X,
    family="binomial",
    link=None,
    weights=None,
    offset=None,
    fit="irls",
    cov="expected",
    epsilon=1e-8,
    maxit=25,
    names=None,
):
    family = family.lower()
    if family not in FAMILIES:
        raise ValueError(
            "vp.stats.glm: family must be one of %s" % ", ".join(FAMILIES)
        )
    link = link or CANONICAL[family]
    if link not in ALLOWED[family]:
        raise ValueError(
            "vp.stats.glm: link %r is not one of %s for %s"
            % (link, sorted(ALLOWED[family]), family)
        )
    if fit not in GLM_FITTERS:
        raise ValueError(
            "vp.stats.glm: fit must be one of %s" % ", ".join(GLM_FITTERS)
        )
    if cov not in ("expected", "observed"):
        raise ValueError("vp.stats.glm: cov must be 'expected' or 'observed'")
    nm = _names(X, names, np.asarray(X).shape[1])
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = _f(y)
    n, k = X.shape
    w = np.ones(n) if weights is None else _f(weights)
    off = np.zeros(n) if offset is None else _f(offset)
    kc = 1 if _has_const(X) else 0
    fam, lk = float(FAMILIES[family]), float(LINKS[link])
    out = getattr(_api(), GLM_FITTERS[fit])(
        y, X, w, off, fam, lk, float(kc), float(epsilon), float(maxit)
    )
    b, C, dev, nulldev, iters, conv, mu, eta, wts, rank = out
    rank = int(rank)
    b = np.where(np.isnan(np.diag(C)), np.nan, b)
    if cov == "observed":
        C = _api().glm_observed(y, X, w, np.ascontiguousarray(eta), fam, lk)
    df_resid = n - rank
    msgs = []
    if not conv:
        msgs.append("glm.fit: algorithm did not converge")
    if family == "binomial" and np.any((mu > 1 - 10 * EPS) | (mu < 10 * EPS)):
        msgs.append(
            "glm.fit: fitted probabilities numerically 0 or 1 occurred"
        )
    if family == "poisson" and np.any(mu < 10 * EPS):
        msgs.append("glm.fit: fitted rates numerically 0 occurred")
    for m in msgs:
        warnings.warn(m, GLMWarning, stacklevel=3)
    # R's summary.glm: dispersion 1 for binomial and Poisson (z tests), else
    # the Pearson chi-squared over the residual degrees of freedom (t tests)
    var = {
        "gaussian": np.ones(n),
        "binomial": mu * (1 - mu),
        "poisson": mu,
        "gamma": mu * mu,
    }[family]
    pearson = (y - mu) * np.sqrt(w / var)
    est = family in ("gaussian", "gamma")
    disp = float(np.sum(pearson**2) / df_resid) if est else 1.0
    bse = np.sqrt(disp * np.diag(C))
    zv = b / bse
    if est:
        pv = 2.0 * FAMILY_OBJECTS["t"].sf(np.abs(zv), df_resid)
        q = FAMILY_OBJECTS["t"].ppf(0.975, df_resid)
    else:
        pv = 2.0 * FAMILY_OBJECTS["norm"].sf(np.abs(zv))
        q = FAMILY_OBJECTS["norm"].ppf(0.975)
    dres = np.sign(y - mu) * np.sqrt(
        np.maximum(_unit_dev(family, y, mu, w), 0.0)
    )
    aic = _aic(family, y, mu, w, dev) + 2 * rank
    return dict(
        coef=b,
        bse=bse,
        tvalues=zv,
        pvalues=pv,
        conf_int=np.column_stack([b - q * bse, b + q * bse]),
        deviance=dev,
        null_deviance=nulldev,
        df_resid=df_resid,
        df_null=n - kc,
        dispersion=disp,
        aic=aic,
        iterations=int(iters),
        converged=bool(conv),
        mu=mu,
        eta=eta,
        resid_deviance=dres,
        resid_pearson=pearson,
        cov_params=C * disp,
        warnings=tuple(msgs),
        family=family,
        link=link,
        fit=fit,
        cov=cov,
        names=nm,
        nobs=n,
        rank=rank,
        _problem="glm:%s:%s:%s:%d:%d" % (family, link, fit, n, k),
    )


def _unit_dev(family, y, mu, w):
    with np.errstate(divide="ignore", invalid="ignore"):
        if family == "binomial":

            def ylogy(a, m):
                return np.where(a > 0, a * np.log(a / m), 0.0)

            return 2 * w * (ylogy(y, mu) + ylogy(1 - y, 1 - mu))
        if family == "poisson":
            return (
                2 * w * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
            )
        if family == "gamma":
            return (
                -2
                * w
                * (np.log(np.where(y == 0, 1.0, y / mu)) - (y - mu) / mu)
            )
        return w * (y - mu) ** 2


def _aic(family, y, mu, w, dev):
    """R's family()$aic (before + 2 rank), with the densities of vp.stats."""
    S = FAMILY_OBJECTS
    if family == "binomial":
        m = w
        return -2 * np.sum(
            np.where(m > 0, w / m, 0)
            * S["binom"].logpmf(np.round(m * y), np.round(m), mu)
        )
    if family == "poisson":
        return -2 * np.sum(S["poisson"].logpmf(y, mu) * w)
    if family == "gamma":
        n = np.sum(w)
        disp = dev / n
        return (
            -2 * np.sum(S["gamma"].logpdf(y, 1 / disp, scale=mu * disp) * w)
            + 2
        )
    nobs = len(y)
    return nobs * (np.log(2 * np.pi * dev / nobs) + 1) + 2


# ------------------------------------------------------------ the tables


def _stars(p):
    return (
        "***"
        if p < 0.001
        else "**"
        if p < 0.01
        else "*"
        if p < 0.05
        else "."
        if p < 0.1
        else ""
    )


def _coef_table(names, b, se, t, p, ci, stat):
    w = max([len(n) for n in names] + [10])
    head = "%-*s %11s %10s %9s %9s %11s %11s" % (
        w,
        "",
        "coef",
        "std err",
        stat,
        "P>|%s|" % stat.strip(),
        "[0.025",
        "0.975]",
    )
    rows = [head, "-" * len(head)]
    for i, n in enumerate(names):
        rows.append(
            "%-*s %11.4g %10.4g %9.3f %9.3g %11.4g %11.4g %s"
            % (w, n, b[i], se[i], t[i], p[i], ci[i, 0], ci[i, 1], _stars(p[i]))
        )
    return rows


class OLSMixin:
    """The table of statsmodels' OLS summary: the fit statistics, then one
    row per coefficient."""

    def summary(self):
        lines = [
            "%s regression results (vp.stats.ols, solver %r)"
            % ("WLS" if self.weighted else "OLS", self.solver),
            "=" * 78,
            "No. Observations: %8d     R-squared:      %10.4f     F-statistic:  %10.4g"
            % (self.nobs, self.rsquared, self.fvalue),
            "Df Residuals:     %8d     Adj. R-squared: %10.4f     Prob (F):     %10.3g"
            % (self.df_resid, self.rsquared_adj, self.f_pvalue),
            "Df Model:         %8d     Log-Likelihood: %10.4g     AIC:          %10.4g"
            % (self.df_model, self.llf, self.aic),
            "Residual SE:      %8.4g     Rank:           %10d     BIC:          %10.4g"
            % (self.resid_sd, self.rank, self.bic),
            "=" * 78,
        ]
        lines += _coef_table(
            self.names,
            self.coef,
            self.bse,
            self.tvalues,
            self.pvalues,
            self.conf_int,
            "t",
        )
        return "\n".join(lines)


class GLMMixin:
    """R's summary.glm in statsmodels' layout."""

    def summary(self):
        stat = "t" if self.family in ("gaussian", "gamma") else "z"
        lines = [
            "Generalized linear model (vp.stats.glm: %s, link %s, fit %s, cov %s)"
            % (self.family, self.link, self.fit, self.cov),
            "=" * 78,
            "No. Observations: %8d     Deviance:       %12.6g   Null deviance: %10.6g"
            % (self.nobs, self.deviance, self.null_deviance),
            "Df Residuals:     %8d     Dispersion:     %12.6g   AIC:           %10.6g"
            % (self.df_resid, self.dispersion, self.aic),
            "Iterations:       %8d     Converged:      %12s"
            % (self.iterations, self.converged),
            "=" * 78,
        ]
        lines += _coef_table(
            self.names,
            self.coef,
            self.bse,
            self.tvalues,
            self.pvalues,
            self.conf_int,
            stat,
        )
        lines += ["Warning: " + m for m in self.warnings]
        return "\n".join(lines)


# ------------------------------------------------------------ checks


def _check_ols(res):
    ok = bool(np.isfinite(res.coef[~np.isnan(res.coef)]).all())
    return [("the coefficients are finite", ok, "")]


def _check_glm(res):
    return [
        (
            "IRLS stopped by R's deviance rule"
            if res.fit == "irls"
            else "Newton stopped by R's deviance rule",
            res.converged,
            "; ".join(res.warnings),
        )
    ]


def _check_describe(res):
    return [("the variance is not negative", res.variance >= 0, "")]


def _check_quantile(res):
    v = np.atleast_1d(res.values)
    return [("the quantiles are finite", bool(np.isfinite(v).all()), "")]

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.optimize: nonlinear least squares, curve fitting and root finding by
MINPACK's Levenberg-Marquardt (lib/minpack/lmder.viv, a port of netlib's
lmder that matches a gfortran build of the original step for step).

The function being fitted is Viv, compiled together with lmder:

    fun="model.viv:residuals"        a def of your own .viv file
    fun=kernels.residuals            a def of a module from vp.load
    fun=residuals                    a Python function in py2viv's numeric
                                     subset, translated to Viv once

least_squares and root call `fun(x, *args)`, which returns the m residuals as
an array; each of `args` is a number or a 1-D array. curve_fit calls the
model at one data point at a time, `f(xi, b1, ..., bk)` (SciPy's form) or
`f(xi, p)` with `p` an array, and fits `(f(xi) - yi) / sigma_i`.

The Jacobian is exact by default: the driver differentiates the residuals
with reverse-mode `grad` (one row per residual for least_squares, one point
at a time for curve_fit). `jac="fd"` takes MINPACK's forward differences
(fdjac2, as lmdif does: what SciPy uses without a Jacobian); `jac="name"`
uses a def of the same file returning the m x n Jacobian.

For each distinct problem the module writes a small driver .viv that imports
it and lib/minpack/fit.viv, compiles it through vipyr's cache, and calls it.
"""
from __future__ import annotations

import hashlib
import inspect
import re
import threading
import warnings
from pathlib import Path

import numpy as np

from . import _core, _ode

EPS = np.finfo(float).eps
LEASTSQ_TOL = 1.49012e-08          # scipy.optimize.leastsq's ftol and xtol
FACTOR = 100.0                     # MINPACK's initial step bound factor
MODE_JAC, MODE_DIAG = 1.0, 2.0     # lmder's mode: scale by the Jacobian, or by diag

# MINPACK's info, in leastsq's words
INFO = {
    0: "Improper input parameters.",
    1: "Both actual and predicted relative reductions in the sum of squares are at most ftol.",
    2: "The relative error between two consecutive iterates is at most xtol.",
    3: "Both actual and predicted relative reductions in the sum of squares are at most ftol, and the "
       "relative error between two consecutive iterates is at most xtol.",
    4: "The cosine of the angle between the residuals and any column of the Jacobian is at most gtol "
       "in absolute value.",
    5: "The number of residual evaluations has reached max_nfev.",
    6: "ftol is too small: no further reduction in the sum of squares is possible.",
    7: "xtol is too small: no further improvement in the approximate solution is possible.",
    8: "gtol is too small: the residuals are orthogonal to the columns of the Jacobian to machine precision.",
}
# least_squares' status from MINPACK's info (SciPy's FROM_MINPACK_TO_COMMON; SciPy keeps 6-8 from
# happening by refusing tolerances below eps, and so does this module)
STATUS = {0: -1, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0, 6: 2, 7: 3, 8: 1}
STATUS_TEXT = {
    -1: "Improper input parameters.",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}

_lock = threading.Lock()
_drivers: dict = {}


class OptimizeProblemError(ValueError):
    """The function or Jacobian given cannot be made into a Viv problem."""


class OptimizeWarning(UserWarning):
    """The covariance of the parameters could not be estimated."""


def _driver_dir() -> Path:
    d = _ode._driver_dir().parent / "optimize"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _viv_params(text, name):
    """The parameters of `def name(...)` in a .viv file's text: [(name, annotation or None)]."""
    m = re.search(r"^def\s+%s\s*\(([^)]*)\)" % re.escape(name), text, re.M)
    if m is None:
        return None
    out = []
    for p in m.group(1).split(","):
        p = p.strip()
        if p:
            n, _, ann = p.partition(":")
            out.append((n.strip(), ann.strip() or None))
    return out


def _source(fun, jac, who):
    """(module name, its text, the def's Viv name, its parameters, the jac def's
    Viv name or None) for fun given as 'file.viv:def', a vp.load def, or a
    Python function."""
    pyjac = callable(jac) and _ode._vipyr_def(jac) is None
    if callable(fun) and _ode._vipyr_def(fun) is None:
        funcs = [fun] + ([jac] if pyjac else [])
        try:
            text = _ode._py2viv(funcs)
        except _ode.OdeProblemError as e:
            raise OptimizeProblemError(str(e).replace("vp.ode", who).replace("rhs", "function")) from None
        mod = "optpy_" + hashlib.sha256(text.encode()).hexdigest()[:12]

        def viv_name(n):
            # py2viv keeps a name, or prefixes one Viv reserves
            m = re.search(r"^def (\w*?%s)\(" % re.escape(n), text, re.M)
            if m is None:
                raise OptimizeProblemError("%s: py2viv's translation has no def for %s" % (who, n))
            return m.group(1)
        name = viv_name(fun.__name__)
        params = [(p, None) for p in inspect.signature(fun).parameters]
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in inspect.signature(fun).parameters.values()):
            raise OptimizeProblemError("%s: %s takes *args or **kwargs; name each parameter" % (who, fun.__name__))
        jname = viv_name(jac.__name__) if pyjac else (jac if isinstance(jac, str) and jac not in ("exact", "fd",
                                                                                                "2-point") else None)
        return mod, text, name, params, jname
    if callable(fun):
        src, name = _ode._vipyr_def(fun)
    else:
        path, _, name = str(fun).rpartition(":")
        if not path or not name:
            raise OptimizeProblemError("%s: give the function as 'file.viv:def', a def of a vp.load module, or a "
                                       "Python function; got %r" % (who, fun))
        src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise OptimizeProblemError("%s: no file %s" % (who, src))
    text = src.read_text(encoding="utf-8")
    params = _viv_params(text, name)
    if params is None:
        raise OptimizeProblemError("%s: %s has no top-level def %s" % (who, src, name))
    jname = None
    if callable(jac):
        jname = getattr(getattr(jac, "__vipyr_raw__", jac), "__name__", None)
    elif isinstance(jac, str) and jac not in ("exact", "fd", "2-point"):
        jname = jac
    if jname and _viv_params(text, jname) is None:
        raise OptimizeProblemError("%s: %s has no top-level def %s" % (who, src, jname))
    return "optuser_" + re.sub(r"\W", "_", src.stem), text, name, params, jname


DRIVER = '''###
post: vp.optimize driver (written by vipyr/_optimize.py)
agent: vipyr
###

from {mod} import {names}
from minpack.fit import lm_fit, lm_fd_jac

{resdefs}
def _opt_into(data: Arr, x: Arr, out: Arr) {{
    out[:] = _opt_res(x, data)
}}

{jacdef}
def _opt_fcn(data: Arr, x: Arr, fvec: Arr, fjac: Arr, iflag) {{
    if iflag == 1 {{
        _opt_into(data, x, fvec)
    }}
    if iflag == 2 {{
        _opt_jac(data, x, fvec, fjac)
    }}
    return iflag
}}

def fit(x0: Arr, data: Arr, m: Int, ftol: Num, xtol: Num, gtol: Num, maxfev: Num, diag: Arr, mode: Num,
        factor: Num) {{
    return lm_fit(_opt_fcn, data, x0, m, ftol, xtol, gtol, maxfev, diag, mode, factor)
}}

def residuals(x: Arr, data: Arr) -> Arr {{
    return _opt_res(x, data)
}}
'''

# data[0] is epsfcn (forward differences); then the problem's own layout
HEAD = 1

# least_squares: data = [epsfcn, o_0 .. o_k, args...]; argument i is data[o_i:o_{i+1}] (an array) or
# data[o_i] (a number)
LS_RES = '''def _opt_res(x: Arr, data: Arr) -> Arr {{
    return {fun}(x{args})
}}
'''
LS_ROW = '''def _opt_row(x: Arr, data: Arr, i) -> Num {{
    return _opt_res(x, data)[i]
}}

def _opt_jac(data: Arr, x: Arr, fvec: Arr, fjac: Arr) {{
    for i to len(fvec) {{
        fjac[i, :] = grad(_opt_row, x, data, i)
    }}
}}
'''
# curve_fit: data = [epsfcn, xdata (m), ydata (m), 1 / sigma (m)]
CF_RES = '''def _opt_point(p: Arr, xi: Num, yi: Num, wi: Num) -> Num {{
    return ({call} - yi) * wi
}}

def _opt_m(data: Arr) {{
    return int((len(data) - 1) / 3)
}}

def _opt_res(x: Arr, data: Arr) -> Arr {{
    m = _opt_m(data)
    out = zeros(m)
    for i to m {{
        out[i] = _opt_point(x, data[1 + i], data[1 + m + i], data[1 + m + m + i])
    }}
    return out
}}
'''
CF_GRAD = '''def _opt_jac(data: Arr, x: Arr, fvec: Arr, fjac: Arr) {{
    m = _opt_m(data)
    for i to m {{
        fjac[i, :] = grad(_opt_point, x, data[1 + i], data[1 + m + i], data[1 + m + m + i])
    }}
}}
'''
FD_JAC = '''def _opt_jac(data: Arr, x: Arr, fvec: Arr, fjac: Arr) {{
    lm_fd_jac(_opt_into, data, x, fvec, fjac, data[0])
}}
'''
USER_JAC = '''def _opt_jac(data: Arr, x: Arr, fvec: Arr, fjac: Arr) {{
    J = {jac}(x{args})
    for i to len(fvec) {{
        fjac[i, :] = J[i, :]
    }}
}}
'''


def _arg_text(kinds):
    """The Viv text of the extra arguments, read from data's offsets."""
    out = []
    for i, k in enumerate(kinds):
        lo = "int(data[%d])" % (HEAD + i)
        if k == "num":
            out.append(", data[%s]" % lo)
        else:
            out.append(", data[%s:int(data[%d])]" % (lo, HEAD + i + 1))
    return "".join(out)


def _pack_args(args, who):
    """(kinds, the data array) for least_squares' extra arguments."""
    kinds, parts = [], []
    for a in args:
        v = np.asarray(a, dtype=float)
        if v.ndim == 0:
            kinds.append("num")
            parts.append(v.reshape(1))
        elif v.ndim == 1:
            kinds.append("arr")
            parts.append(v)
        else:
            raise ValueError("%s: each of args must be a number or a 1-D array; got shape %s" % (who, v.shape))
    k = len(parts)
    offsets, at = [], HEAD + k + 1
    for p in parts:
        offsets.append(at)
        at += len(p)
    offsets.append(at)
    return kinds, offsets, parts


def driver(form, fun, jac, nargs_kinds=(), nparams=None, fast=False):
    """The compiled driver for one problem: (module, key, mode name, number of
    parameters the model takes or None). form is "ls" or "cf"."""
    who = "vp.optimize." + ("curve_fit" if form == "cf" else "least_squares")
    mode = "fd" if jac in ("fd", "2-point") else ("exact" if jac in (None, "exact") else "user")
    if form == "cf" and mode == "user":
        raise OptimizeProblemError("%s: jac is None (grad, exact) or 'fd'; a Jacobian def of your own goes to "
                                   "vp.optimize.least_squares" % who)
    mod, text, name, params, jname = _source(fun, jac if mode == "user" else None, who)
    names = [name] + ([jname] if jname else [])
    k_model = None
    if form == "ls":
        if len(params) != 1 + len(nargs_kinds):
            raise OptimizeProblemError("%s: %s takes %d parameters; it is called as %s(x%s) with args of length %d"
                                       % (who, name, len(params), name, ", ..." if nargs_kinds else "",
                                          len(nargs_kinds)))
        args = _arg_text(nargs_kinds)
        resdefs = LS_RES.format(fun=name, args=args)
        if mode == "exact":
            jacdef = LS_ROW.format()
        elif mode == "fd":
            jacdef = FD_JAC.format()
        else:
            jacdef = USER_JAC.format(jac=jname, args=args)
    else:
        if len(params) < 2:
            raise OptimizeProblemError("%s: the model %s must take x and the parameters: f(x, b1, ..., bk) or "
                                       "f(x, p)" % (who, name))
        if len(params) == 2 and params[1][1] == "Arr":
            call = "%s(xi, p)" % name
        else:
            k_model = len(params) - 1
            call = "%s(xi, %s)" % (name, ", ".join("p[%d]" % i for i in range(k_model)))
        resdefs = CF_RES.format(call=call)
        jacdef = (CF_GRAD if mode == "exact" else FD_JAC).format()
    drv = DRIVER.format(mod=mod, names=", ".join(dict.fromkeys(names)), resdefs=resdefs, jacdef=jacdef)
    key = hashlib.sha256((drv + "\0" + text).encode()).hexdigest()[:16]
    with _lock:
        hit = _drivers.get((key, fast))
        if hit is None:
            d = _driver_dir() / key
            d.mkdir(parents=True, exist_ok=True)
            p = d / (mod + ".viv")
            if not p.is_file() or p.read_text(encoding="utf-8") != text:
                p.write_text(text, encoding="utf-8")
            dp = d / ("optdrv_%s.viv" % key)
            if not dp.is_file() or dp.read_text(encoding="utf-8") != drv:
                dp.write_text(drv, encoding="utf-8")
            hit = _core.load(dp, fast=fast)
            _drivers[(key, fast)] = hit
    return hit, key, mode, k_model


def _raw(mod, name):
    f = getattr(mod, name)
    return getattr(f, "__vipyr_raw__", f)


def _tol(name, v, who):
    v = float(v)
    if not v >= EPS:
        if v == 0 and name == "gtol":
            return v
        raise ValueError("%s: %s must be at least machine epsilon (%.3g); got %r" % (who, name, EPS, v))
    return v


def _run(mod, mode, x0, data, m, ftol, xtol, gtol, maxfev, diag, epsfcn):
    n = len(x0)
    data = np.concatenate([[float(epsfcn)], data])
    if diag is None:
        d, md = np.zeros(n), MODE_JAC
    else:
        d = np.broadcast_to(np.asarray(diag, dtype=float), (n,)).copy()
        if np.any(d <= 0):
            raise ValueError("vp.optimize: x_scale (diag) must be positive")
        md = MODE_DIAG
    x, f, J, info, nfev, njev = _raw(mod, "fit")(np.ascontiguousarray(x0), data, m, ftol, xtol, gtol,
                                                   float(maxfev), d, md, FACTOR)
    nfev, njev, info = int(nfev), int(njev), int(info)
    if mode == "fd":
        nfev += n * njev                  # lmdif counts the difference evaluations
    return np.asarray(x), np.asarray(f), np.asarray(J).reshape(m, n), info, nfev, njev


def _least_squares(fun, x0, *, jac=None, args=(), ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, max_nfev=None,
                   diff_step=None, fast=False):
    """vp.optimize.least_squares, before the result object is made."""
    who = "vp.optimize.least_squares"
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    if x0.ndim != 1:
        raise ValueError("%s: x0 must be 1-D" % who)
    n = len(x0)
    kinds, offsets, parts = _pack_args(tuple(args) if isinstance(args, (tuple, list)) else (args,), who)
    mod, key, mode, _ = driver("ls", fun, jac, tuple(kinds), fast=fast)
    payload = np.concatenate([np.asarray(offsets, dtype=float)] + parts)
    f0 = np.asarray(_raw(mod, "residuals")(x0, np.concatenate([[0.0], payload])))
    m = len(f0)
    if m < n:
        raise ValueError("%s: method 'lm' needs at least as many residuals (%d) as variables (%d)" % (who, m, n))
    ftol, xtol, gtol = _tol("ftol", ftol, who), _tol("xtol", xtol, who), _tol("gtol", gtol, who)
    if max_nfev is None:
        max_nfev = 100 * n * (n + 1) if mode == "fd" else 100 * n     # as SciPy's least_squares
    epsfcn = EPS if diff_step is None else float(diff_step) ** 2
    diag = None if isinstance(x_scale, str) and x_scale == "jac" else 1.0 / np.asarray(x_scale, dtype=float)
    x, f, J, info, nfev, njev = _run(mod, mode, x0, payload, m, ftol, xtol, gtol, max_nfev, diag, epsfcn)
    g = J.T @ f
    status = STATUS.get(info, -1)
    return {"x": x, "cost": 0.5 * float(f @ f), "fun": f, "jac": J, "grad": g,
            "optimality": float(np.max(np.abs(g))) if n else 0.0, "active_mask": np.zeros(n, dtype=int),
            "nfev": nfev, "njev": njev, "status": status, "message": STATUS_TEXT[status], "success": status > 0,
            "info": info, "_problem": key}


def covariance(J, rss, m, n, absolute_sigma):
    """SciPy's curve_fit covariance from the Jacobian of the weighted residuals:
    the pseudo-inverse of J^T J by an SVD of J, infinite if J is rank deficient,
    scaled by rss / (m - n) unless sigma is absolute."""
    _, s, VT = np.linalg.svd(J, full_matrices=False)
    threshold = EPS * max(J.shape) * s[0] if s.size else 0.0
    if s.size < n or np.any(s <= threshold):
        warnings.warn("Covariance of the parameters could not be estimated", OptimizeWarning, stacklevel=4)
        return np.full((n, n), np.inf)
    pcov = (VT.T / s ** 2) @ VT
    if not absolute_sigma:
        if m > n:
            pcov = pcov * (rss / (m - n))
        else:
            warnings.warn("Covariance of the parameters could not be estimated", OptimizeWarning, stacklevel=4)
            pcov = np.full((n, n), np.inf)
    return pcov


def _curve_fit(f, xdata, ydata, p0=None, *, sigma=None, absolute_sigma=False, jac=None, ftol=LEASTSQ_TOL,
               xtol=LEASTSQ_TOL, gtol=0.0, maxfev=None, epsfcn=None, diag=None, fast=False):
    """vp.optimize.curve_fit, before the result object is made."""
    who = "vp.optimize.curve_fit"
    xd = np.asarray(xdata, dtype=float)
    yd = np.asarray(ydata, dtype=float)
    if xd.ndim != 1 or yd.shape != xd.shape:
        raise ValueError("%s: xdata and ydata must be 1-D arrays of the same length (one x per point); got %s "
                         "and %s. For several x per point use vp.optimize.least_squares with args"
                         % (who, xd.shape, yd.shape))
    if not (np.all(np.isfinite(xd)) and np.all(np.isfinite(yd))):
        raise ValueError("%s: xdata and ydata must be finite" % who)
    m = len(yd)
    if sigma is None:
        w = np.ones(m)
    else:
        s = np.asarray(sigma, dtype=float)
        if s.shape != (m,):
            raise ValueError("%s: sigma must be 1-D with one entry per point (a covariance matrix is not taken)"
                             % who)
        if np.any(s <= 0):
            raise ValueError("%s: sigma must be positive" % who)
        w = 1.0 / s
    mod, key, mode, k_model = driver("cf", f, jac, fast=fast)
    if p0 is None:
        if k_model is None:
            raise ValueError("%s: the model takes its parameters as one array; give p0" % who)
        p0 = np.ones(k_model)
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    n = len(p0)
    if k_model is not None and k_model != n:
        raise ValueError("%s: the model takes %d parameters and p0 has %d" % (who, k_model, n))
    if n > m:
        raise TypeError("%s: the number of parameters (%d) must not exceed the number of data points (%d)"
                        % (who, n, m))
    ftol, xtol = _tol("ftol", ftol, who), _tol("xtol", xtol, who)
    gtol = float(gtol)
    if maxfev is None or maxfev == 0:
        maxfev = 200 * (n + 1) if mode == "fd" else 100 * (n + 1)      # as SciPy's leastsq
    x, fv, J, info, nfev, njev = _run(mod, mode, p0, np.concatenate([xd, yd, w]), m, ftol, xtol, gtol, maxfev,
                                      diag, EPS if epsfcn is None else epsfcn)
    if info not in (1, 2, 3, 4):
        raise RuntimeError("Optimal parameters not found: " + INFO.get(info, "info %d" % info))
    rss = float(fv @ fv)
    pcov = covariance(J, rss, m, n, absolute_sigma)
    return {"popt": x, "pcov": pcov, "perr": np.sqrt(np.diag(pcov)), "fun": fv, "rss": rss, "nfev": nfev,
            "njev": njev, "info": info, "message": INFO[info], "success": True, "_problem": key}


def _root(fun, x0, *, args=(), jac=None, method="lm", xtol=LEASTSQ_TOL, ftol=LEASTSQ_TOL, gtol=0.0, maxfev=None,
          eps=None, diag=None, fast=False):
    """vp.optimize.root, before the result object is made."""
    who = "vp.optimize.root"
    if method != "lm":
        raise ValueError("%s: method must be 'lm' (Levenberg-Marquardt on the n x n system, as SciPy's "
                         "root(method='lm')); MINPACK's hybrd/hybrj (Powell's hybrid method, SciPy's default 'hybr') is not ported" % who)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    kinds, offsets, parts = _pack_args(tuple(args) if isinstance(args, (tuple, list)) else (args,), who)
    mod, key, mode, _ = driver("ls", fun, jac, tuple(kinds), fast=fast)
    payload = np.concatenate([np.asarray(offsets, dtype=float)] + parts)
    m = len(np.asarray(_raw(mod, "residuals")(x0, np.concatenate([[0.0], payload]))))
    if m != n:
        raise ValueError("%s: fun must return as many values as x has (%d); it returns %d" % (who, n, m))
    ftol, xtol = _tol("ftol", ftol, who), _tol("xtol", xtol, who)
    if maxfev is None or maxfev == 0:
        maxfev = 200 * (n + 1) if mode == "fd" else 100 * (n + 1)
    x, f, J, info, nfev, njev = _run(mod, mode, x0, payload, n, ftol, xtol, float(gtol), maxfev, diag,
                                     EPS if eps is None else eps)
    return {"x": x, "fun": f, "jac": J, "success": info in (1, 2, 3, 4), "status": info,
            "message": INFO.get(info, "info %d" % info), "nfev": nfev, "njev": njev, "_problem": key}

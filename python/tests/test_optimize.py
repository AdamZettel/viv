# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.optimize: least_squares, curve_fit and root against SciPy (iterates and
evaluation counts) and NIST's certified values, the forms a function can
take, and the refusals.

    python -m pytest python/tests/test_optimize.py

The full NIST and MGH comparison is lib/minpack/bench_optimize.py; Misra1a
is also held by lib/minpack/fit.viv's own tests (run here through r.check()).
"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "lib" / "minpack"))

import vipyr as vp  # noqa: E402
from vipyr import _optimize  # noqa: E402
import nist_strd as N  # noqa: E402

so = pytest.importorskip("scipy.optimize")


def rosen(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def decay(b, t, y):
    return b[0] * np.exp(-b[1] * t) + b[2] - y


T = np.linspace(0.0, 4.0, 40)
Y = 3.0 * np.exp(-1.3 * T) + 0.5 + 0.01 * np.sin(7.0 * T)


def test_least_squares_rosenbrock_takes_scipys_steps():
    r = vp.optimize.least_squares(rosen, [-1.2, 1.0])
    s = so.least_squares(rosen, [-1.2, 1.0], jac=lambda x: np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]]),
                         method="lm")
    assert r.success and r.status == s.status
    assert (r.nfev, r.njev) == (s.nfev, s.njev)
    np.testing.assert_allclose(r.x, s.x, rtol=1e-14)
    np.testing.assert_allclose(r.jac, s.jac, rtol=1e-14)
    assert r.check().passed


# Forward differences divide a residual's rounding by a step of 1.5e-8 relative: the last bits of
# Viv's exp and NumPy's differ, so iterates by differences agree with SciPy's to about 1e-9, not
# to the bit (the counts are the same).
FD_RTOL = 1e-7


def test_forward_differences_are_lmdif():
    """jac='fd' is MINPACK's fdjac2 inside lmder: SciPy's lmdif path, evaluation for evaluation."""
    r = vp.optimize.least_squares(decay, [1.0, 1.0, 0.0], args=(T, Y), jac="fd")
    s = so.least_squares(decay, [1.0, 1.0, 0.0], args=(T, Y), method="lm")
    assert r.nfev == s.nfev and r.status == s.status
    np.testing.assert_allclose(r.x, s.x, rtol=FD_RTOL)


def test_curve_fit_misra1a_certified_values_and_scipys_counts():
    p = N.load("Misra1a")
    popt, pcov = vp.optimize.curve_fit(N.misra1a, p.x, p.y, p.starts[0])
    assert N.lre(popt, p.certified).min() >= 9
    assert N.lre(np.sqrt(np.diag(pcov)), p.cert_sd).min() >= 9
    r = vp.optimize.curve_fit(N.misra1a, p.x, p.y, p.starts[0], jac="fd")
    popt_s, pcov_s, info, msg, ier = so.curve_fit(N.misra1a, p.x, p.y, p.starts[0], full_output=True)
    assert r.nfev == info["nfev"] and r.info == ier
    np.testing.assert_allclose(r.popt, popt_s, rtol=FD_RTOL)
    np.testing.assert_allclose(r.pcov, pcov_s, rtol=1e-5)


def test_curve_fit_sigma_and_absolute_sigma_as_scipy():
    sigma = 0.01 + 0.02 * T
    for absolute in (False, True):
        r = vp.optimize.curve_fit("%s:model" % _viv_model(), T, Y, [1.0, 1.0, 0.0], sigma=sigma,
                                  absolute_sigma=absolute, jac="fd")
        popt, pcov = so.curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, T, Y, [1.0, 1.0, 0.0], sigma=sigma,
                                  absolute_sigma=absolute)
        np.testing.assert_allclose(r.popt, popt, rtol=FD_RTOL)
        np.testing.assert_allclose(r.pcov, pcov, rtol=1e-5)


def _viv_model(tmp=[]):
    if not tmp:
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "expmodel.viv").write_text(textwrap.dedent("""
            def model(t: Num, p: Arr) -> Num {
                return p[0] * exp(-p[1] * t) + p[2]
            }
            def scipy_form(t, a, b, c) {
                return a * exp(-b * t) + c
            }
            def res(b: Arr, t: Arr, y: Arr) -> Arr {
                return b[0] * exp(-b[1] * t) + b[2] - y
            }
            def res_jac(b: Arr, t: Arr, y: Arr) -> Arr {
                J = zeros(len(t), len(b))
                for i to len(t) {
                    e = exp(-b[1] * t[i])
                    J[i, 0] = e
                    J[i, 1] = -b[0] * t[i] * e
                    J[i, 2] = 1.0
                }
                return J
            }
            """))
        tmp.append(d / "expmodel.viv")
    return tmp[0]


def test_function_forms_agree():
    """A Viv def (array or SciPy-form parameters), a vp.load def and a Python function: one answer."""
    src = _viv_model()
    a = vp.optimize.curve_fit("%s:model" % src, T, Y, [1.0, 1.0, 0.0])
    b = vp.optimize.curve_fit("%s:scipy_form" % src, T, Y, [1.0, 1.0, 0.0])
    c = vp.optimize.curve_fit(lambda_free_model, T, Y, [1.0, 1.0, 0.0])
    k = vp.load(src)
    d = vp.optimize.least_squares(k.res, [1.0, 1.0, 0.0], args=(T, Y))
    e = vp.optimize.least_squares("%s:res" % src, [1.0, 1.0, 0.0], args=(T, Y), jac="res_jac")
    for r in (b, c):
        np.testing.assert_allclose(r.popt, a.popt, rtol=1e-10)
    for r in (d, e):
        np.testing.assert_allclose(r.x, a.popt, rtol=1e-8)


def lambda_free_model(t, a, b, c):
    return a * np.exp(-b * t) + c


def test_root_is_scipys_lm():
    def powell(x):
        return np.array([x[0] + 10.0 * x[1], np.sqrt(5.0) * (x[2] - x[3]), (x[1] - 2.0 * x[2]) ** 2,
                         np.sqrt(10.0) * (x[0] - x[3]) ** 2])

    def fr(x):
        return np.array([-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1],
                         -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]])
    r = vp.optimize.root(fr, [15.0, -2.0], jac="fd")
    s = so.root(fr, [15.0, -2.0], method="lm")
    np.testing.assert_allclose(r.x, s.x, rtol=FD_RTOL)
    assert r.nfev == s.nfev and r.success
    r = vp.optimize.root(powell, [3.0, -1.0, 0.0, 1.0])
    assert np.linalg.norm(r.fun) < 1e-15
    with pytest.raises(ValueError, match="hybrd/hybrj"):
        vp.optimize.root(powell, [3.0, -1.0, 0.0, 1.0], method="hybr")


def test_refusals():
    with pytest.raises(ValueError, match="at least as many residuals"):
        vp.optimize.least_squares(lambda_free_res3, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="takes 3 parameters and p0 has 2"):
        vp.optimize.curve_fit(lambda_free_model, T, Y, [1.0, 1.0])
    with pytest.raises(ValueError, match="machine epsilon"):
        vp.optimize.least_squares(rosen, [-1.2, 1.0], ftol=1e-20)
    with pytest.raises(ValueError, match="1-D arrays of the same length"):
        vp.optimize.curve_fit(lambda_free_model, np.vstack([T, T]), Y)
    with pytest.raises(_optimize.OptimizeProblemError, match="no top-level def"):
        vp.optimize.least_squares("%s:nope" % _viv_model(), [1.0])
    with pytest.raises(RuntimeError, match="Optimal parameters not found"):
        vp.optimize.curve_fit(lambda_free_model, T, Y, [1.0, 1.0, 0.0], maxfev=3)


def lambda_free_res3(x):
    return np.array([x[0] - 1.0, x[1] - 2.0, x[2] * x[3]])


def test_rank_deficient_covariance_is_infinite():
    def two_of_one(x, a, b):
        return (a + b) * x
    with pytest.warns(_optimize.OptimizeWarning):
        r = vp.optimize.curve_fit(two_of_one, T, 2.0 * T, [1.0, 0.5])
    assert np.all(np.isinf(r.pcov))

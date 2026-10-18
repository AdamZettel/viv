# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.ode: the solvers against SciPy's solve_ivp (steps and answers), the
forms a right-hand side can take, events, batches, and the refusals.

    python -m pytest python/tests/test_ode.py

The reference solutions of the stiff test set are held by the Viv tests
themselves (lib/ode/stiff_reference.viv, run here through r.check()).
"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import vipyr as vp  # noqa: E402
from vipyr import _ode  # noqa: E402

scipy_integrate = pytest.importorskip("scipy.integrate")
solve_ivp = scipy_integrate.solve_ivp

K = [0.04, 1e4, 3e7]


def rob(t, y):
    return [-K[0] * y[0] + K[1] * y[1] * y[2], K[0] * y[0] - K[1] * y[1] * y[2] - K[2] * y[1] ** 2,
            K[2] * y[1] ** 2]


def rob_jac(t, y):
    return [[-K[0], K[1] * y[2], K[1] * y[1]], [K[0], -K[1] * y[2] - 2 * K[2] * y[1], -K[1] * y[1]],
            [0.0, 2 * K[2] * y[1], 0.0]]


@pytest.mark.parametrize("method, name", [("radau", "Radau"), ("bdf", "BDF")])
def test_stiff_methods_take_scipys_steps_and_reach_its_answer(method, name):
    r = vp.ode.solve("robertson", (0.0, 40.0), [1.0, 0.0, 0.0], method=method, rtol=1e-6, atol=1e-10, params=K)
    s = solve_ivp(rob, (0.0, 40.0), [1.0, 0.0, 0.0], method=name, rtol=1e-6, atol=1e-10, jac=rob_jac)
    assert r.success and r.status == 1
    assert abs(r.stats["steps"] - (len(s.t) - 1)) <= 2
    assert abs(r.stats["nfev"] - s.nfev) <= 0.02 * s.nfev
    assert np.max(np.abs(r.y[:, -1] - s.y[:, -1]) / np.abs(s.y[:, -1])) < 1e-7


def test_dopri5_is_scipys_rk45_step_for_step():
    lv = lambda t, z: [1.5 * z[0] - z[0] * z[1], -3.0 * z[1] + z[0] * z[1]]  # noqa: E731
    s = solve_ivp(lv, (0.0, 15.0), [10.0, 5.0], rtol=1e-8, atol=1e-10)
    r = vp.ode.solve("lotka", (0.0, 15.0), [10.0, 5.0], method="dopri5", rtol=1e-8, atol=1e-10,
                     params=[1.5, 1.0, 3.0, 1.0])
    assert r.stats["steps"] == len(s.t) - 1 and r.stats["nfev"] == s.nfev
    # every step lands on SciPy's points, to the rounding the two first steps differ by
    np.testing.assert_allclose(r.t, s.t, rtol=1e-8)
    np.testing.assert_allclose(r.y, s.y, rtol=1e-7, atol=1e-12)


def test_van_der_pol_at_mu_1e6_by_radau():
    r = vp.ode.solve("vdpol", (0.0, 2.0), [2.0, 0.0], method="radau", rtol=1e-8, atol=1e-8, params=[1e-12])
    # SciPy's Radau at rtol = atol = 1e-10
    np.testing.assert_allclose(r.y[:, -1], [1.705546217535324, -0.8934763625465864], rtol=1e-7)
    # BDF fails at the first relaxation jump, as SciPy's does
    b = vp.ode.solve("vdpol", (0.0, 2.0), [2.0, 0.0], method="bdf", rtol=1e-8, atol=1e-8, params=[1e-12])
    assert not b.success and b.status == -1 and 0.8 < b.t[-1] < 0.81


def test_auto_changes_to_radau_on_a_stiff_problem_and_not_on_a_mild_one():
    r = vp.ode.solve("vdpol", (0.0, 2.0), [2.0, 0.0], method="auto", rtol=1e-6, atol=1e-6, params=[1e-6])
    assert r.stiff_at is not None and r.success
    m = vp.ode.solve("lotka", (0.0, 15.0), [10.0, 5.0], method="auto", params=[1.5, 1.0, 3.0, 1.0])
    assert m.stiff_at is None


def test_a_terminal_event_ends_the_solve_at_the_landing():
    r = vp.ode.solve("ball", (0.0, 10.0), [10.0, 0.0], method="dopri5", params=[9.81], events="ball_hits",
                     direction=[-1], terminal=[True])
    land = np.sqrt(2 * 10.0 / 9.81)
    assert r.status == 2 and r.message.startswith("A termination event")
    assert abs(r.t_events[0][0] - land) < 1e-12 and r.t[-1] == r.t_events[0][0]
    assert abs(r.y_events[0][0, 0]) < 1e-12


def test_event_directions_and_scipy_agree_on_the_crossings(tmp_path):
    src = tmp_path / "prey.viv"
    src.write_text(textwrap.dedent("""\
        def lv(t: Num, y: Arr, p: Arr) -> Arr {
            return [p[0] * y[0] - p[1] * y[0] * y[1], -p[2] * y[1] + p[3] * y[0] * y[1]]
        }
        def at_five(t: Num, y: Arr, p: Arr) -> Arr {
            return [y[0] - 5.0]
        }
    """))
    p = [1.5, 1.0, 3.0, 1.0]
    lv = lambda t, z: [1.5 * z[0] - z[0] * z[1], -3.0 * z[1] + z[0] * z[1]]  # noqa: E731
    for d in (1, -1, 0):
        ev = lambda t, z: z[0] - 5.0  # noqa: E731
        ev.direction = d
        s = solve_ivp(lv, (0.0, 15.0), [10.0, 5.0], rtol=1e-10, atol=1e-12, events=ev)
        r = vp.ode.solve("%s:lv" % src, (0.0, 15.0), [10.0, 5.0], method="dopri5", rtol=1e-10, atol=1e-12,
                         params=p, events="at_five", direction=[d])
        assert len(r.t_events[0]) == len(s.t_events[0])
        np.testing.assert_allclose(r.t_events[0], s.t_events[0], rtol=1e-9)


def test_every_form_of_rhs_gives_the_same_solution(tmp_path):
    kw = dict(method="radau", rtol=1e-6, atol=1e-10, params=K, t_eval=[1.0, 10.0, 40.0])
    base = vp.ode.solve("robertson", (0.0, 40.0), [1.0, 0.0, 0.0], **kw)
    # the same def in a file of one's own: the same bits
    src = tmp_path / "chem.viv"
    lib = (Path(vp.viv_root()) / "lib" / "ode" / "problems.viv").read_text()
    start = lib.index("def robertson(")
    src.write_text(lib[start:lib.index("# ---", start)])
    by_path = vp.ode.solve("%s:robertson" % src, (0.0, 40.0), [1.0, 0.0, 0.0], **kw)
    assert np.array_equal(by_path.y, base.y)
    # a def of a module vp.load made
    mod = vp.load(src)
    by_mod = vp.ode.solve(mod.robertson, (0.0, 40.0), [1.0, 0.0, 0.0], **kw)
    assert np.array_equal(by_mod.y, base.y)
    # a Python function, translated by py2viv: the same equations, other rounding
    by_py = vp.ode.solve(py_robertson, (0.0, 40.0), [1.0, 0.0, 0.0], **kw)
    np.testing.assert_allclose(by_py.y, base.y, rtol=1e-12)
    # finite differences instead of grad: a different Jacobian, the same answer to the tolerance
    fd = vp.ode.solve("robertson", (0.0, 40.0), [1.0, 0.0, 0.0], jac="fd", **kw)
    np.testing.assert_allclose(fd.y, base.y, rtol=1e-5)
    # a Jacobian def of one's own
    given = vp.ode.solve("robertson", (0.0, 40.0), [1.0, 0.0, 0.0], jac="robertson_jac", **kw)
    assert np.array_equal(given.y, base.y)


def py_robertson(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([-p[0] * y[0] + p[1] * y[1] * y[2],
                     p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] * y[1],
                     p[2] * y[1] * y[1]])


def test_a_batch_row_is_the_single_solve_bit_for_bit():
    P = np.column_stack([np.linspace(1.2, 4.0, 50) / 7.0, np.full(50, 1.0 / 7.0)])
    t = np.linspace(0.0, 180.0, 37)
    for method in ("dopri5", "radau", "bdf"):
        b = vp.ode.batch("sir", (0.0, 180.0), [0.999, 0.001, 0.0], P, t_eval=t, method=method, rtol=1e-8,
                         atol=1e-10)
        assert b.y.shape == (50, 3, 37) and b.success
        for row in (0, 17, 49):
            s = vp.ode.solve("sir", (0.0, 180.0), [0.999, 0.001, 0.0], method=method, rtol=1e-8, atol=1e-10,
                             params=P[row], t_eval=t)
            assert np.array_equal(s.y, b.y[row]) and s.stats["steps"] == b.stats["steps"][row]


def test_a_batch_of_terminal_events_pads_with_nan():
    P = np.array([[9.81], [1.62], [24.8]])
    t = np.linspace(0.0, 3.0, 31)
    b = vp.ode.batch("ball", (0.0, 3.0), [10.0, 0.0], P, t_eval=t, method="dopri5", events="ball_hits",
                     direction=[-1], terminal=[True])
    land = np.sqrt(2 * 10.0 / P[:, 0])
    np.testing.assert_allclose(b.first_event, np.where(land <= 3.0, land, np.nan), rtol=1e-12)
    assert np.isnan(b.y[0, 0, -1]) and not np.isnan(b.y[1, 0, -1])
    assert list(b.status) == [2, 1, 2]


def test_backward_integration_and_t_eval():
    f = vp.ode.solve("lotka", (0.0, 5.0), [10.0, 5.0], method="dopri5", rtol=1e-10, atol=1e-12,
                     params=[1.5, 1.0, 3.0, 1.0])
    b = vp.ode.solve("lotka", (5.0, 0.0), f.y[:, -1], method="dopri5", rtol=1e-10, atol=1e-12,
                     params=[1.5, 1.0, 3.0, 1.0], t_eval=[5.0, 2.5, 0.0])
    np.testing.assert_allclose(b.t, [5.0, 2.5, 0.0])
    np.testing.assert_allclose(b.y[:, -1], [10.0, 5.0], rtol=1e-7)


@pytest.mark.parametrize("call, exc, word", [
    (lambda: vp.ode.solve("robertson", (0, 1), [1, 0, 0], method="rk4", params=K), ValueError, "method"),
    (lambda: vp.ode.solve("no_such_problem", (0, 1), [1.0]), _ode.OdeProblemError, "no_such_problem"),
    (lambda: vp.ode.solve("robertson", (0, 1), [1, 0, 0], params=K, t_eval=[2.0]), ValueError, "t_eval"),
    (lambda: vp.ode.solve("robertson", (0, 1), [1, 0, 0], params=K, rtol=-1.0), ValueError, "rtol"),
    (lambda: vp.ode.solve("robertson", (0, 1), [[1, 0, 0]], params=K), ValueError, "y0"),
    (lambda: vp.ode.batch("sir", (0, 1), [1, 0, 0], np.ones((3, 2)), t_eval=[]), ValueError, "t_eval"),
    (lambda: vp.ode.solve("robertson", (0, 1), [1, 0, 0], params=K, no_such=1), TypeError, "no_such"),
])
def test_bad_arguments_are_refused_by_name(call, exc, word):
    with pytest.raises(exc) as e:
        call()
    assert word in str(e.value)


def test_a_missing_def_in_a_file_is_named(tmp_path):
    src = tmp_path / "m.viv"
    src.write_text("def f(t: Num, y: Arr, p: Arr) -> Arr {\n    return -y\n}\n")
    with pytest.raises(_ode.OdeProblemError, match="no top-level def g"):
        vp.ode.solve("%s:g" % src, (0.0, 1.0), [1.0])
    r = vp.ode.solve("%s:f" % src, (0.0, 1.0), [1.0], method="bdf", rtol=1e-10, atol=1e-12)
    assert abs(r.y[0, -1] - np.exp(-1.0)) < 1e-8

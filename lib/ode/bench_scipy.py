# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.ode against scipy.integrate.solve_ivp: wall time, warm, same problems,
same methods and tolerances.

    VIV=target/release/viv python3 lib/ode/bench_scipy.py [--quick]

Writes lib/ode/bench_scipy.json (the gallery prints it). Each time is the
best of several repeats after a warm-up call (the vp.ode driver compiled and
loaded, SciPy's modules imported). SciPy gets an analytic Jacobian written
in NumPy for the stiff problems, as a careful user would give it; vp.ode
uses its exact Jacobian from grad. Both solve the same equations with the
same step logic, and the table shows both step counts.

Single systems: one solve per call. Batch: B independent solves of one
model over a parameter sweep, a Python loop of solve_ivp against one
vp.ode.batch call. The batch rows report solves per second, and the SciPy
loop over all B is timed once (it is long).
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "python"))
import vipyr as vp  # noqa: E402

QUICK = "--quick" in sys.argv


def rob(t, y, k1, k2, k3):
    return [-k1 * y[0] + k2 * y[1] * y[2], k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] ** 2, k3 * y[1] ** 2]


def rob_jac(t, y, k1, k2, k3):
    return [[-k1, k2 * y[2], k2 * y[1]], [k1, -k2 * y[2] - 2 * k3 * y[1], -k2 * y[1]], [0.0, 2 * k3 * y[1], 0.0]]


def vdp(t, y, eps):
    return [y[1], ((1 - y[0] ** 2) * y[1] - y[0]) / eps]


def vdp_jac(t, y, eps):
    return [[0.0, 1.0], [(-2 * y[0] * y[1] - 1) / eps, (1 - y[0] ** 2) / eps]]


def hires(t, y):
    k = 280 * y[5] * y[7]
    return [-1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007, 1.71 * y[0] - 8.75 * y[1],
            -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4], 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
            -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6], -k + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
            k - 1.81 * y[6], -k + 1.81 * y[6]]


def hires_jac(t, y):
    J = np.zeros((8, 8))
    J[0, :3] = [-1.71, 0.43, 8.32]
    J[1, :2] = [1.71, -8.75]
    J[2, 2:5] = [-10.03, 0.43, 0.035]
    J[3, 1:4] = [8.32, 1.71, -1.12]
    J[4, 4:7] = [-1.745, 0.43, 0.43]
    J[5, 3:8] = [0.69, 1.71, -0.43 - 280 * y[7], 0.69, -280 * y[5]]
    J[6, 5:8] = [280 * y[7], -1.81, 280 * y[5]]
    J[7, 5:8] = [-280 * y[7], 1.81, -280 * y[5]]
    return J


def orego(t, y):
    return [77.27 * (y[1] + y[0] * (1 - 8.375e-6 * y[0] - y[1])), (y[2] - (1 + y[0]) * y[1]) / 77.27,
            0.161 * (y[0] - y[2])]


def orego_jac(t, y):
    return [[77.27 * (1 - 2 * 8.375e-6 * y[0] - y[1]), 77.27 * (1 - y[0]), 0.0],
            [-y[1] / 77.27, -(1 + y[0]) / 77.27, 1 / 77.27], [0.161, 0.0, -0.161]]


def lotka(t, z, a, b, c, d):
    return [a * z[0] - b * z[0] * z[1], -c * z[1] + d * z[0] * z[1]]


def kepler(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


def sir(t, y, beta, gamma):
    i = beta * y[0] * y[1]
    return [-i, i - gamma * y[1], gamma * y[1]]


def best(f, reps):
    f()                                   # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


# name, vp rhs, scipy rhs, scipy jac, method, t_span, y0, params, rtol, atol
E = 0.5
SINGLE = [
    ("Robertson to 1e11", "robertson", rob, rob_jac, "radau", (0.0, 1e11), [1.0, 0.0, 0.0], [0.04, 1e4, 3e7], 1e-8, 1e-20),
    ("Robertson to 1e11", "robertson", rob, rob_jac, "bdf", (0.0, 1e11), [1.0, 0.0, 0.0], [0.04, 1e4, 3e7], 1e-8, 1e-20),
    ("Van der Pol, mu = 1000", "vdpol", vdp, vdp_jac, "radau", (0.0, 2.0), [2.0, 0.0], [1e-6], 1e-8, 1e-8),
    ("Van der Pol, mu = 1000", "vdpol", vdp, vdp_jac, "bdf", (0.0, 2.0), [2.0, 0.0], [1e-6], 1e-8, 1e-8),
    ("HIRES", "hires", hires, hires_jac, "radau", (0.0, 321.8122), [1, 0, 0, 0, 0, 0, 0, 0.0057], None, 1e-8, 1e-12),
    ("HIRES", "hires", hires, hires_jac, "bdf", (0.0, 321.8122), [1, 0, 0, 0, 0, 0, 0, 0.0057], None, 1e-8, 1e-12),
    ("Oregonator", "oregonator", orego, orego_jac, "radau", (0.0, 360.0), [1.0, 2.0, 3.0], None, 1e-8, 1e-8),
    ("Oregonator", "oregonator", orego, orego_jac, "bdf", (0.0, 360.0), [1.0, 2.0, 3.0], None, 1e-8, 1e-8),
    ("Lotka-Volterra", "lotka", lotka, None, "dopri5", (0.0, 15.0), [10.0, 5.0], [1.5, 1.0, 3.0, 1.0], 1e-10, 1e-12),
    ("Kepler, e = 0.5, 10 orbits", "kepler", kepler, None, "dopri5", (0.0, 20 * np.pi),
     [1 - E, 0.0, 0.0, np.sqrt((1 + E) / (1 - E))], None, 1e-10, 1e-12),
]
SCIPY_METHOD = {"radau": "Radau", "bdf": "BDF", "dopri5": "RK45"}


def single_rows():
    rows = []
    for name, vrhs, srhs, sjac, method, span, y0, p, rtol, atol in SINGLE:
        args = tuple(p) if p else ()
        kw = dict(method=SCIPY_METHOD[method], rtol=rtol, atol=atol, args=args or None)
        if sjac is not None:
            kw["jac"] = sjac
        reps = 2 if QUICK else 5
        ts, rs = best(lambda: solve_ivp(srhs, span, y0, **kw), reps)
        tv, rv = best(lambda: vp.ode.solve(vrhs, span, y0, method=method, rtol=rtol, atol=atol, params=p), reps * 4)
        diff = float(np.max(np.abs(rv.y[:, -1] - rs.y[:, -1]) / np.maximum(np.abs(rs.y[:, -1]), atol)))
        rows.append(dict(problem=name, method=method, scipy_s=ts, vv_s=tv, speedup=ts / tv,
                         scipy_steps=len(rs.t) - 1, vv_steps=rv.stats["steps"], scipy_nfev=int(rs.nfev),
                         vv_nfev=rv.stats["nfev"], end_rel_diff=diff))
        print("%-28s %-7s scipy %9.4f s (%5d steps)  vp %9.5f s (%5d steps)  x%7.1f   end diff %.1e"
              % (name, method, ts, len(rs.t) - 1, tv, rv.stats["steps"], ts / tv, diff), flush=True)
    return rows


def batch_rows():
    rows = []
    B = 1000 if QUICK else 10000
    t_eval = np.linspace(0.0, 180.0, 181)
    beta = np.linspace(1.2, 4.0, B) / 7.0
    P = np.column_stack([beta, np.full(B, 1.0 / 7.0)])
    y0 = [0.999, 0.001, 0.0]
    for method in ("dopri5", "radau"):
        rtol, atol = 1e-6, 1e-9
        tv, rv = best(lambda: vp.ode.batch("sir", (0.0, 180.0), y0, P, t_eval=t_eval, method=method, rtol=rtol,
                                           atol=atol), 2 if QUICK else 3)
        kw = dict(method=SCIPY_METHOD[method], rtol=rtol, atol=atol, t_eval=t_eval)
        solve_ivp(sir, (0.0, 180.0), y0, args=tuple(P[0]), **kw)          # warm
        t0 = time.perf_counter()
        ys = np.stack([solve_ivp(sir, (0.0, 180.0), y0, args=tuple(P[b]), **kw).y for b in range(B)])
        ts = time.perf_counter() - t0
        diff = float(np.max(np.abs(ys - rv.y)))
        rows.append(dict(problem="%d SIR epidemics (R0 1.2 to 4)" % B, method=method, B=B, scipy_s=ts, vv_s=tv,
                         speedup=ts / tv, scipy_per_solve_ms=1e3 * ts / B, vv_per_solve_ms=1e3 * tv / B,
                         max_abs_diff=diff))
        print("%-28s %-7s scipy %9.3f s  vp %8.4f s  x%7.1f   (%.3f ms vs %.4f ms per solve; max diff %.1e)"
              % ("%d SIR" % B, method, ts, tv, ts / tv, 1e3 * ts / B, 1e3 * tv / B, diff), flush=True)
    return rows


def main():
    load = os.getloadavg()
    print("load average at start: %.1f %.1f %.1f" % load)
    single = single_rows()
    batch = batch_rows()
    out = dict(
        machine="%s %s, Python %s, NumPy %s, SciPy %s" % (platform.system(), platform.machine(),
                                                        platform.python_version(), np.__version__, scipy.__version__),
        load_average_start=[round(x, 1) for x in load], load_average_end=[round(x, 1) for x in os.getloadavg()],
        vv_build="vipyr exact mode (-O2 -ffp-contract=off)", quick=QUICK, single=single, batch=batch)
    if not QUICK:
        (HERE / "bench_scipy.json").write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", HERE / "bench_scipy.json")


if __name__ == "__main__":
    main()

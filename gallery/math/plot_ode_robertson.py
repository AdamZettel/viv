# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Robertson's stiff kinetics over sixteen decades
===============================================

Three species, rate constants 0.04, 1e4 and 3e7: the fast reactions settle
in microseconds and the slow one keeps going until t = 1e11. An explicit
method would need steps of about 1e-4 the whole way. `vp.ode.solve` with
Radau IIA and with BDF crosses the range in under two thousand steps,
lands on the published reference solution, and takes the steps SciPy's
`solve_ivp` takes.
"""

# %%
# ## The problem
#
# `"robertson"` is a def of `lib/ode/problems.viv`:
#
#     y1' = -k1 y1 + k2 y2 y3
#     y2' =  k1 y1 - k2 y2 y3 - k3 y2^2
#     y3' =  k3 y2^2
#
# The Jacobian is exact: the driver writes one reverse-mode `grad` per row.
# Output at 162 times, t = 0 and ten per decade from 1e-5 to 1e11, by the
# dense output of each step.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

k = [0.04, 1e4, 3e7]
t_out = np.concatenate([[0.0], np.logspace(-5, 11, 161)])
runs = {}
for method in ("radau", "bdf"):
    runs[method] = vp.ode.solve(
        "robertson",
        (0.0, 1e11),
        [1.0, 0.0, 0.0],
        method=method,
        rtol=1e-8,
        atol=1e-20,
        params=k,
        t_eval=t_out,
    )
    s = runs[method].stats
    print(
        "%-6s %5d steps, %5d rhs evaluations, %4d Jacobians, %4d LU "
        "factorizations, %3d rejected"
        % (method, s["steps"], s["nfev"], s["njev"], s["nlu"], s["rejected"])
    )

# %%
# ## Against the reference solution
#
# The IVP test set of Mazzia and Magherini (Bari) publishes the state at
# t = 1e11 to sixteen digits. At rtol = 1e-8 Radau IIA is within 1e-10 of it
# relative to each component, and BDF within 2e-7. Mass is conserved:
# y1 + y2 + y3 = 1 throughout.

ref = np.array(
    [0.2083340149701255e-07, 0.8333360770334713e-13, 0.9999999791665050]
)
for method, r in runs.items():
    err = np.abs(r.y[:, -1] - ref) / ref
    print(
        "%-6s y(1e11) = %s   largest relative error %.1e   max |sum - 1| = "
        "%.1e"
        % (
            method,
            np.array2string(r.y[:, -1], precision=10),
            err.max(),
            np.abs(r.y.sum(axis=0) - 1).max(),
        )
    )

# %%
# ## The same steps as SciPy
#
# `lib/ode/radau.viv` and `lib/ode/bdf.viv` follow SciPy's step-size,
# Newton and order logic, so they take SciPy's steps (to within a handful)
# and agree with its answer at the end to rounding.

from scipy.integrate import solve_ivp


def rob(t, y):
    return [
        -k[0] * y[0] + k[1] * y[1] * y[2],
        k[0] * y[0] - k[1] * y[1] * y[2] - k[2] * y[1] ** 2,
        k[2] * y[1] ** 2,
    ]


def rob_jac(t, y):
    return [
        [-k[0], k[1] * y[2], k[1] * y[1]],
        [k[0], -k[1] * y[2] - 2 * k[2] * y[1], -k[1] * y[1]],
        [0.0, 2 * k[2] * y[1], 0.0],
    ]


for method, name in (("radau", "Radau"), ("bdf", "BDF")):
    sp = solve_ivp(
        rob,
        (0.0, 1e11),
        [1.0, 0.0, 0.0],
        method=name,
        rtol=1e-8,
        atol=1e-20,
        jac=rob_jac,
    )
    d = np.abs(runs[method].y[:, -1] - sp.y[:, -1]) / sp.y[:, -1]
    print(
        "%-6s SciPy %5d steps, vp.ode %5d steps; end states differ by %.0e "
        "relative"
        % (method, len(sp.t) - 1, runs[method].stats["steps"], d.max())
    )

# %%
# ## The picture
#
# The species against log time, y2 scaled by 1e4 as is customary. The
# intermediate y2 jumps to its quasi-steady level within 1e-4 time units,
# then decays with y1 for ten decades while y3 takes over.

r = runs["radau"]
mask = t_out > 0
fig = figure(
    "Robertson's kinetics (Radau IIA, rtol 1e-8)", "t", "concentration"
)
fig.xscale("log")
fig.plot(t_out[mask], r.y[0, mask], "y1")
fig.plot(t_out[mask], 1e4 * r.y[1, mask], "1e4 y2")
fig.plot(t_out[mask], r.y[2, mask], "y3")
fig.style("y1", "s1 nomarker")
fig.style("1e4 y2", "s2 nomarker")
fig.style("y3", "s3 nomarker")

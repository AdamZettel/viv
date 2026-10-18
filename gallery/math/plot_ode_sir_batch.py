# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Ten thousand epidemics in one call
==================================

The SIR model run for 10,000 values of the basic reproduction number R0,
each an adaptive Dormand-Prince solve of its own, in one `vp.ode.batch`
call. The final size of each epidemic is checked against the closed form
it must satisfy, and one row is checked against a single `vp.ode.solve`
bit for bit.
"""

# %%
# ## The sweep
#
# `"sir"` in `lib/ode/problems.viv` is S' = -beta S I, I' = beta S I - gamma I,
# R' = gamma I, with the population as fractions. Recovery takes 7 days on
# average (gamma = 1/7) and R0 = beta / gamma runs from 0.5 to 5. One in
# ten thousand starts infected. Output daily for two years.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

B = 10_000
gamma = 1.0 / 7.0
R0 = np.linspace(0.5, 5.0, B)
P = np.column_stack([R0 * gamma, np.full(B, gamma)])
y0 = [1.0 - 1e-4, 1e-4, 0.0]
days = np.arange(0.0, 731.0)
r = vp.ode.batch(
    "sir",
    (0.0, 730.0),
    y0,
    P,
    t_eval=days,
    method="dopri5",
    rtol=1e-9,
    atol=1e-12,
)
print(
    "solutions:",
    r.y.shape,
    "(epidemic, S I R, day)   all succeeded:",
    r.success,
)
print(
    "steps per epidemic: %d to %d, %.0f on average"
    % (r.stats["steps"].min(), r.stats["steps"].max(), r.stats["steps"].mean())
)

# %%
# ## The final size
#
# With S + I + R = 1, dividing S' by R' and integrating gives
# ln(S0 / S_end) = R0 (1 - S_end) once I has died out: one equation for the
# fraction never infected. Solved here by bisection for every R0 above 1.5
# (below it the epidemic is still burning after two years), and compared
# with the batch's S on day 730.


def s_end(r0, s0=y0[0]):
    lo, hi = 1e-300, 1.0 / r0  # the root below the threshold S = 1/R0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.log(s0 / mid) - r0 * (1.0 - mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


done = R0 >= 1.5
exact = np.array([s_end(x) for x in R0[done]])
print(
    "largest |S(730) - S_end| over %d epidemics with R0 >= 1.5: %.1e"
    % (done.sum(), np.abs(r.y[done, 0, -1] - exact).max())
)
print(
    "R0 = 2: %.2f%% never infected (closed form %.2f%%)"
    % (
        100 * r.y[np.argmin(np.abs(R0 - 2.0)), 0, -1],
        100 * s_end(R0[np.argmin(np.abs(R0 - 2.0))]),
    )
)

# %%
# ## One row is one solve
#
# The batch is a loop in Viv over the same compiled solver, not a
# vectorized approximation: row b is exactly what `vp.ode.solve` returns
# for that row's parameters.

b = 7_777
one = vp.ode.solve(
    "sir",
    (0.0, 730.0),
    y0,
    method="dopri5",
    rtol=1e-9,
    atol=1e-12,
    params=P[b],
    t_eval=days,
)
print(
    "row %d equals vp.ode.solve bit for bit: %s (%d steps both)"
    % (b, np.array_equal(one.y, r.y[b]), one.stats["steps"])
)

# %%
# ## The pictures
#
# The infected fraction for six values of R0, and the fraction ever
# infected, 1 - S_end, against R0: the batch's every 100th epidemic on the
# closed form's curve.

fig = figure("Infected fraction, SIR, recovery in 7 days", "day", "I")
for target in (1.2, 1.5, 2.0, 2.5, 3.0, 4.0):
    i = int(np.argmin(np.abs(R0 - target)))
    label = "R0 = %.1f" % target
    fig.plot(days[::2], r.y[i, 1, ::2], label)
    fig.style(label, "nomarker")

grid = np.linspace(1.05, 5.0, 80)
fig = figure("Fraction ever infected", "$R_0$", "$1 - S_{\\mathrm{end}}$")
fig.plot(grid, 1.0 - np.array([s_end(x) for x in grid]), "closed form")
fig.style("closed form", "s1 nomarker")
sel = np.arange(0, B, 100)
sel = sel[R0[sel] >= 1.05]
fig.scatter(R0[sel], 1.0 - r.y[sel, 0, -1], "vp.ode.batch, day 730")

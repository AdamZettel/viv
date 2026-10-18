# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
An interconnection study on the IEEE 118-bus system
===================================================

A 200 MW generator asks to connect at bus 38 of MATPOWER's case118, a
345 kV bus. `vp.power.interconnect` solves the base case and all 240
single outages (every branch and generator) without the project and with
it, lists the limits the project breaks, gives its distribution factors on
every branch, and finds its transfer limit: the largest output with no
new violation, by bisection on the full AC N-1.

case118 has no branch ratings, so the limits here are bus voltages: 0.95
to 1.05 per unit in the base case and 0.90 to 1.10 after an outage.
"""

# %%
# ## Before and after
#
# The project holds bus 38 at its pre-project voltage within a 0.95 power
# factor either way; its 200 MW is absorbed by the reference bus (69).

import numpy as np
import vipyr as vp
from vipyr import _power
from vipyr.figure import figure

r = vp.power.interconnect("case118", 38, 200.0, mw_max=1500.0, tol_mw=1.0)
print(
    "violations before:",
    len(r.violations_before),
    " after:",
    len(r.violations_after),
    " new:",
    len(r.new),
    " worsened:",
    len(r.worsened),
)
res = {
    "before": r.violations_before,
    "new": r.new,
    "kinds": r.kinds,
    "elems": r.elems,
}
print("before the project (contingency, violation, bus row, Vm, limit, %):")
for line in _power.format_violations(res, "before"):
    print("  " + line)
print("new with it:", _power.format_violations(res, "new") or "none")

# %%
# ## Distribution factors
#
# The DC model's share of each MW of the project that flows on each
# branch, injected at bus 38 and taken out at bus 69. The largest ones mark
# the path the output takes.

top = np.argsort(-abs(r.df))[:5]
c, _ = _power.load_case("case118")
for k in top:
    print(
        "branch %3d (%3d-%3d): %+.3f MW per MW"
        % (k, c["branch"][k, 0], c["branch"][k, 1], r.df[k])
    )

# %%
# ## The transfer limit
#
# Each bisection step runs the whole N-1 again at one output. Above the
# limit a contingency drives a bus outside its emergency band.

print("transfer limit: %.1f MW" % r.transfer_limit_mw)
first = r.steps[r.steps[:, 1] > 0]
above = first[first[:, 0] > r.transfer_limit_mw]
print(
    "smallest output tried with a new violation: %.1f MW" % above[:, 0].min()
)
poi = int(np.where(c["bus"][:, 0] == 38)[0][0])
vset = vp.power.runpf("case118").vm[poi]
mw = np.linspace(0.0, 1400.0, 15)
worst_lo = []
for p in mw:
    q = p * np.tan(np.arccos(0.95))
    bus, gen = _power._api().inject_arrays(
        c["baseMVA"], c["bus"], c["gen"], c["branch"], poi, p, vset, -q, q, 1
    )
    n1 = vp.power.contingency(dict(c, bus=bus, gen=gen))
    vm = np.where(n1.vm[1:] > 0, n1.vm[1:], np.nan)
    worst_lo.append(np.nanmin(vm))
fig = figure(
    "Lowest post-contingency voltage against the project's output",
    "project output (MW)",
    "lowest Vm over all outages (per unit)",
)
fig.plot(mw, np.array(worst_lo), "lowest Vm")
fig.hline("emergency limit 0.90", 0.90)
fig.vline("transfer limit", r.transfer_limit_mw)

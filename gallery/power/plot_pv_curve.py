# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The PV curve of a transfer into bus 38 of the IEEE 118-bus system
=================================================================

How much power can be pushed into bus 38 of MATPOWER's case118, and
taken out at the reference bus, before the power flow has no solution?
`vp.power.cpf` traces the voltage at bus 38 against the transfer by
continuation, through the nose of the curve where Newton's method on the
plain power flow stops converging, and down the low-voltage branch.
"""

# %%
# ## The curve
#
# Pseudo-arclength continuation: a tangent step along the curve, then
# Newton's method on the power flow plus the step equation. Near the nose
# the trace steps back and approaches with smaller steps.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

r = vp.power.cpf("case118", 38, step=0.2, max_step=1.0, singular_values=True)
print(
    "points:",
    len(r.mw),
    " nose at %.2f MW, Vm(38) = %.4f per unit" % (r.nose_mw, r.v_poi[r.nose]),
)
print("lowest Vm in the network at the nose: %.4f" % r.v_min[r.nose])
upper = np.arange(len(r.mw)) <= r.nose
fig = figure(
    "PV curve, transfer into bus 38 (case118)",
    "transfer (MW)",
    "Vm (per unit)",
)
fig.plot(r.mw[upper], r.v_poi[upper], "bus 38, upper branch")
fig.plot(r.mw[~upper], r.v_poi[~upper], "bus 38, lower branch")
fig.plot(r.mw, r.v_min, "lowest Vm in the network")
fig.vline("nose", r.nose_mw)

# %%
# ## Why Newton fails there
#
# The smallest singular value of the power-flow Jacobian falls to nearly
# zero at the nose: the Jacobian is singular where the two branches meet.

print(
    "smallest singular value: %.3g at the base case, %.3g at the nose"
    % (r.smin[0], r.smin[r.nose])
)
fig = figure(
    "Smallest singular value of the Jacobian along the curve",
    "transfer (MW)",
    "$\\sigma_{\\min}$",
)
fig.plot(r.mw[upper], r.smin[upper], "upper branch")
fig.plot(r.mw[~upper], r.smin[~upper], "lower branch")
fig.yscale("log")

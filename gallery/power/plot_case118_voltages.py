# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Voltage profile of the IEEE 118-bus system
==========================================

Solve the power flow of MATPOWER's case118 (the IEEE 118-bus test system, a
model of part of the American Electric Power system in the Midwestern
United States) with
`vp.power.runpf`, and plot every bus's voltage magnitude and angle. The
Newton-Raphson solution is compared with the linear DC model, which assumes
flat 1.0 per-unit magnitudes and lossless branches, and with the solution
when generators are held to their reactive limits.
"""

# %%
# ## The AC solution
#
# Newton-Raphson in polar coordinates from the case's starting voltages. The
# 54 generator buses hold their voltage set points; the 64 load buses settle
# where the reactive power balances.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

nr = vp.power.runpf("case118")
print("converged:", nr.success, "in", nr.iterations, "iterations, largest mismatch %.1e per unit" % nr.mismatch)
print("Vm from %.4f to %.4f per unit" % (nr.vm.min(), nr.vm.max()))
print("Va from %.2f to %.2f degrees" % (nr.va.min(), nr.va.max()))
print("generation %.1f MW for %.1f MW of load: losses %.1f MW" %
      (nr.pg.sum(), nr.bus[:, 2].sum(), nr.pg.sum() - nr.bus[:, 2].sum()))

# %%
# ## Magnitudes
#
# Each point is one bus, in the case's bus order. The case's limits are 0.94
# and 1.06 per unit. Holding the generators to their reactive limits turns
# the ones that exceed them into fixed-Q injections; the buses around them
# move.

held = vp.power.runpf("case118", enforce_q_lims=True)
moved = np.argsort(-abs(held.vm - nr.vm))[:3]
at_limit = np.isclose(held.qg, held.gen[:, 3]) | np.isclose(held.qg, held.gen[:, 4])
print("generators held at a reactive limit:", int(at_limit.sum()))
print("with Q limits: %d iterations; largest change %.4f per unit at buses %s" %
      (held.iterations, abs(held.vm - nr.vm).max(), ", ".join(str(int(nr.bus[i, 0])) for i in moved)))

bus = nr.bus[:, 0]
fig = figure("Bus voltage magnitudes, case118", "bus", "Vm (per unit)", window=(0.0, 119.0, 0.93, 1.07))
fig.scatter(bus, nr.vm, "Newton-Raphson")
fig.scatter(bus, held.vm, "with Q limits")
fig.hline("Vmin", 0.94)
fig.hline("Vmax", 1.06)

# %%
# ## Angles, AC against DC
#
# The DC model gets the angles from one linear solve, with the reference
# bus (bus 69) held at its angle. Here its angles sit a few degrees off the
# AC ones across most of the network, and its branch flows differ by tens
# of MW on the most loaded branches.

dc = vp.power.runpf("case118", method="dc")
err = dc.va - nr.va
print("DC angle error: largest %.2f degrees, mean |error| %.2f degrees" % (abs(err).max(), abs(err).mean()))
print("DC branch flow error: largest %.1f MW" % abs(dc.pf - nr.pf).max())

fig = figure("Bus voltage angles, case118", "bus", "Va (degrees)")
fig.plot(bus, nr.va, "Newton-Raphson")
fig.plot(bus, dc.va, "DC model")

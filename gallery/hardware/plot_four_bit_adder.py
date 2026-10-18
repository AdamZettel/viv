# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A four-bit adder from NAND gates, simulated exhaustively
========================================================

Build a full adder out of nine NAND gates, chain four of them into a
ripple-carry adder, and check it on all 512 combinations of two four-bit
numbers and a carry in. The gate logic and the schematic's gate outlines are
Viv (`gallery/hardware/adder.viv`, which draws with `lib/gates.viv`), loaded
with `vp.load`; the simulation also counts gate delays, which shows the carry
rippling from bit to bit.
"""

# %%
# ## One full adder
#
# A full adder adds three bits: a, b and the carry in. Its sum is
# a xor b xor c and its carry out is ab or c(a xor b). The schematic shows
# that logic as two XOR gates, two AND gates and an OR; `adder.viv` builds
# each from NAND gates, nine in all, since NAND alone can make every gate.
# The truth table is all eight inputs.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

hw = vp.load("gallery/hardware/adder.viv")
print(" a b c | sum carry")
for k in range(8):
    a, b, c = (k >> 2) & 1, (k >> 1) & 1, k & 1
    s, co = hw.full_adder(float(a), float(b), float(c))
    print(" %d %d %d |  %d    %d" % (a, b, c, s, co))

AND, OR, XOR = 0, 1, 2
fig = figure("A full adder: the gates", "", "", window=(-0.6, 7.6, -0.4, 4.4))
fig.aspect("equal")
gates = {
    "xor 1": (XOR, 1.5, 3.2),
    "xor 2": (XOR, 4.0, 3.0),
    "and 1": (AND, 1.5, 1.4),
    "and 2": (AND, 4.0, 1.9),
    "or": (OR, 5.8, 1.1),
}
w, h = 1.0, 0.8
for name, (kind, x, y) in gates.items():
    fig.plot(hw.gate_x(kind, x, y, w, h), hw.gate_y(kind, x, y, w, h), "gates")
wires = [
    ((0.0, 3.4), (1.5, 3.4)),
    ((0.0, 3.0), (1.5, 3.0)),  # a, b into xor 1
    ((0.0, 1.6), (1.5, 1.6)),
    ((0.0, 1.2), (1.5, 1.2)),  # a, b into and 1
    ((2.5, 3.2), (4.0, 3.2)),
    ((0.0, 2.5), (3.6, 2.5)),  # a xor b, c into xor 2
    ((3.6, 2.5), (3.6, 2.8)),
    ((3.6, 2.8), (4.0, 2.8)),
    ((3.2, 3.2), (3.2, 2.1)),
    ((3.2, 2.1), (4.0, 2.1)),  # a xor b into and 2
    ((0.6, 2.5), (0.6, 1.7)),
    ((0.6, 1.7), (4.0, 1.7)),  # c into and 2
    ((5.0, 1.9), (5.4, 1.9)),
    ((5.4, 1.9), (5.4, 1.3)),
    ((5.4, 1.3), (5.85, 1.3)),
    ((2.5, 1.4), (5.4, 1.4)),
    ((5.4, 1.4), (5.4, 0.9)),
    ((5.4, 0.9), (5.85, 0.9)),
    ((5.0, 3.0), (7.2, 3.0)),
    ((6.8, 1.1), (7.2, 1.1)),
]
xs, ys = [], []
for (x0, y0), (x1, y1) in wires:
    xs += [x0, x1, float("nan")]
    ys += [y0, y1, float("nan")]
fig.plot(xs, ys, "wires")
fig.style("gates", "s1 thick nomarker")
fig.style("wires", "s2 thin nomarker")
for text, x, y in (
    ("a", -0.45, 3.3),
    ("b", -0.45, 2.9),
    ("c", -0.45, 2.4),
    ("a", -0.45, 1.5),
    ("b", -0.45, 1.1),
    ("sum", 7.25, 2.95),
    ("carry", 7.25, 1.05),
):
    fig.label("%s %.1f" % (text, y), text, x, y)

# %%
# ## All 512 additions
#
# `add4(a, b, cin)` runs the four full adders on every pair at once and
# returns the five-bit result. The maps show it for carry in 0: the sum bits
# (a + b mod 16) and the carry out, with a across and b down.

a, b = np.meshgrid(np.arange(16.0), np.arange(16.0))
wrong = 0
for cin in (0.0, 1.0):
    got = hw.add4(a.ravel(), b.ravel(), cin)
    wrong += int(np.sum(got != (a + b + cin).ravel()))
    if cin == 0.0:
        res = got.reshape(16, 16)
print("additions checked: %d, wrong: %d" % (2 * 256, wrong))
print(
    "7 + 9 = %d, 15 + 15 + 1 = %d"
    % (res[9, 7], hw.add4(np.array([15.0]), np.array([15.0]), 1.0)[0])
)
fig = figure("Sum bits, a + b mod 16", "a", "b")
fig.heatmap(res % 16, "sum")
fig = figure("Carry out", "a", "b")
fig.heatmap(np.floor(res / 16), "carry")

# %%
# ## The carry ripples
#
# Each NAND output settles one gate delay after its later input. With every
# input present at time 0, `settle_times` returns when each output bit is
# final. Every full adder has to wait for the carry from the one below, so
# the delay grows by two gates per bit: s_i settles after 6 + 2i delays, so
# the top sum bit of a 32-bit ripple adder would need 68. Carry-lookahead
# adders exist to shorten that chain.

t = hw.settle_times()
names = ["s0", "s1", "s2", "s3", "carry out"]
for n, v in zip(names, t):
    print("%-9s settles after %2d gate delays" % (n, v))
fig = figure("When each output settles", "", "gate delays")
fig.bars(np.arange(5), t, "settle time")
for k, n in enumerate(names):
    fig.xtick(n, k)

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
The digits lens shows which digits are real
===========================================

A program prints 16 or 17 significant digits whether it computed them or
not. `vp.digits` (the command `viv digits`) reruns a Viv program in the
interpreter with its floating-point evaluation perturbed and counts, for
every printed number, the leading digits that stay the same. Three kinds of
perturbation are tried: the order of every built-in reduction (pairwise,
left to right, reversed, random), FMA contraction of `a*b + c`, and Monte
Carlo arithmetic, Verificarlo's method, where each `+ - * /` gets random
noise of half a unit in the last place and the spread of the results over 16
runs gives the count. This page runs it on a cancellation, an ill-conditioned
solve swept over the condition number, and four real programs.
"""

# %%
# ## Cancellation
#
# `1 - cos(x)` for small x subtracts two numbers that agree in their leading
# digits. The rounding error of cos(x), about 1e-16, is all that is left of
# the difference once x^2/2 falls near it, so the digits that survive are
# about 16 - log10(2/x^2). The rewrite `2 sin(x/2)^2` computes the same
# quantity with no subtraction. Each value is printed to 17 digits, and the
# column after it counts how many of them are stable.

import numpy as np
import vipyr as vp
from vipyr.digits import digits
from vipyr.figure import figure

cancel = digits("examples/digits/cancel.viv", attrib=0)
print(
    "%-8s  %-26s %6s   %-26s %6s"
    % ("x", "1 - cos(x)", "digits", "2 sin(x/2)^2", "digits")
)
outs = cancel.outputs
for k in range(0, len(outs), 3):
    x, naive, good = outs[k], outs[k + 1], outs[k + 2]
    print(
        "%-8.0e  %-26.17g %6.1f   %-26.17g %6.1f   expected %.1f"
        % (
            x.value,
            naive.value,
            naive.stable,
            good.value,
            good.stable,
            max(0.0, 16 - np.log10(2 / x.value**2)),
        )
    )

# %%
# The count for `1 - cos(x)` comes from the Monte Carlo runs: reduction order
# and FMA do not touch it, since it has no reduction and no multiply-add. At
# x = 1e-8, cos(x) rounds to exactly 1 and the program prints 0, which has no
# correct digit.
#
# ## A sum in three orders
#
# 100000 copies of 0.1 added by a loop, by Kahan's compensated loop, and by
# the built-in `sum`, which adds pairwise. The table gives each
# perturbation's count separately. No perturbation reorders a loop. `sum` is
# where the order matters: added left to right instead of pairwise, it gives
# the loop's answer, which agrees with the pairwise one to 11.7 digits, and
# that is also how many digits of the loop's result are correct (the exact
# sum of the stored 0.1s is 10000.000000000000555). Monte Carlo arithmetic
# gives the loop 13.8: it measures the spread of rounding errors, and these
# all round the same way, so their sum is larger than their spread.

kahan = digits("examples/digits/kahan.viv", attrib=0)
print(
    "%-6s %-24s %6s %6s %6s %7s"
    % ("", "value", "order", "fma", "mca", "stable")
)
for name, o in zip(("loop", "kahan", "sum"), kahan.outputs):
    show = lambda v: "all" if v is None or v >= 15.95 else "%.1f" % v
    print(
        "%-6s %-24.17g %6s %6s %6s %7s"
        % (
            name,
            o.value,
            show(o.order),
            show(o.fma),
            show(o.mca),
            show(o.stable),
        )
    )

# %%
# ## Stable digits against the condition number
#
# `examples/digits/hilbert.viv` solves H x = H 1 for the n x n Hilbert
# matrix by Gaussian elimination with partial pivoting, written out in Viv
# loops. The exact answer is all ones, so the digits it got right are known:
# -log10 of the largest error. The rule of thumb for a backward-stable solve
# is 16 - log10(cond(H)) correct digits. For each n the lens reports the
# fewest stable digits over the n entries of x.

ns = list(range(2, 13))
conds, stable, correct = [], [], []
for n in ns:
    H = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)
    rep = digits("examples/digits/hilbert.viv", sets={"n": n}, attrib=0)
    conds.append(np.linalg.cond(H))
    stable.append(rep.stable.min())
    correct.append(
        min(15.95, -np.log10(np.abs(rep.values - 1.0).max() + 1e-300))
    )
print(
    "%3s  %9s  %14s  %13s  %14s"
    % ("n", "cond(H)", "16 - log10(c)", "stable (lens)", "correct digits")
)
for n, c, s, t in zip(ns, conds, stable, correct):
    print(
        "%3d  %9.2e  %14.1f  %13.1f  %14.1f" % (n, c, 16 - np.log10(c), s, t)
    )

fig = figure(
    "Hilbert solve: digits against the condition number",
    "$\\log_{10} \\mathrm{cond}(H)$",
    "digits",
)
lc = np.log10(conds)
fig.plot(lc, stable, "stable (digits lens)")
fig.plot(lc, correct, "correct (error against x = 1)")
fig.plot(lc, 16 - lc, "$16 - \\log_{10}$ cond")
fig.style("stable (digits lens)", "s1 circle")
fig.style("correct (error against x = 1)", "s2 square")
fig.style("$16 - \\log_{10}$ cond", "s3 dash nomarker")

# %%
# The lens needs no exact answer: it measures how much the computed digits
# move, which here tracks the error against the known solution within about
# a digit, and both fall one digit per decade of condition number.
#
# ## Checked against Verificarlo's method
#
# `examples/digits/verificarlo_ref.py` runs the same three toy programs, as
# their C would be written, with every operation done the way Verificarlo's
# `mca` mode specifies (noise on the operands and the result, the operation
# in exact rational arithmetic, one rounding to double) and computes the
# digits with the same formula. Over 32 samples each, the two counts agree to
# within 0.2 digits on all 20 printed numbers.
#
# ## Real programs
#
# `viv digits` on four programs from the repository, with six lines each
# perturbed alone to say where the lost digits go. These runs take minutes,
# so the table is read from examples/digits/real_programs.json, written by
# `python3 examples/digits/real_programs.py` (16 Monte Carlo samples,
# three random orders).

import json

real = json.load(open("examples/digits/real_programs.json"))
for prog in real["programs"]:
    print("%s  (%s)" % (prog["name"], prog["path"]))
    for o in prog["outputs"]:
        print(
            "    %-16s %-24.17g stable %4.1f"
            % (o["label"], o["value"], o["stable"])
        )
    if prog["top_line"]:
        print(
            "    most digits moved by %s: %s"
            % (prog["top_line"]["at"], prog["top_line"]["code"])
        )

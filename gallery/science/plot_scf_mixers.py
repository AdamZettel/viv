# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Passes to self-consistency in calcium-40 by density mixer
=========================================================

Solve calcium-40 with the Skyrme force SLy4 using `vp.nuclear.skyrme_hf`
three times, once per density mixer of `lib/scf/mix.viv`: linear mixing,
Pulay's mixing and Johnson's modified Broyden mixing (`mixing=`). Each pass
of the self-consistent field (`lib/scf/loop.viv`) builds new densities from
the last ones; the mixer decides what the next pass starts from. Count the
passes each needs and check that all three reach the same nucleus.
"""

# %%
# ## Passes by mixer
#
# Each run stops when the total energy changes by less than 1e-10 MeV
# between passes. Pulay and Broyden combine the last six passes
# (`history=6`); all three take half of the residual (`mix=0.5`).

import vipyr as vp
from vipyr.figure import figure

MIXERS = ("linear", "pulay", "broyden")
runs = {
    m: vp.nuclear.skyrme_hf(
        nucleus="40Ca", force="SLy4", e_tol=1e-10, mixing=m
    )
    for m in MIXERS
}
for m, r in runs.items():
    print(
        "%-8s passes %3d   E/A = %.12f MeV   r_ch = %.10f fm"
        % (m, r.iterations, r.e_per_a, r.r_ch)
    )

# %%
# ## Against the fraction mixed
#
# The passes for each mixer as the fraction `mix` goes from 0.3 to 0.7.
# Linear mixing is fastest near 0.5, takes twice as many passes at 0.3 and
# 0.6, and at 0.7 does not settle within the 300 passes allowed (the point
# at 300). Pulay and Broyden take 24 to 37 passes across the whole range:
# their step is fitted to the residuals they remember.

fractions = [0.3, 0.4, 0.5, 0.6, 0.7]
fig = figure("Calcium-40, SLy4: passes to 1e-10 MeV", "mix", "passes")
for k, m in enumerate(MIXERS):
    passes = [
        vp.nuclear.skyrme_hf(
            nucleus="40Ca", force="SLy4", e_tol=1e-10, mixing=m, mix=a
        ).iterations
        for a in fractions
    ]
    print(m, dict(zip(fractions, passes)))
    fig.plot(fractions, passes, m)
    fig.style(m, "s%d" % (k + 1))

# %%
# ## The same fixed point
#
# Two mixers given the same densities return different next densities, so
# they can be compared only once converged. Stopping on an energy change of
# 1e-10 MeV leaves the densities about 1e-9 from the fixed point (the energy
# is stationary there, so it settles before the densities do); at 1e-13 MeV
# the three agree to round-off. The reference is the linear run.

tight = {
    m: vp.nuclear.skyrme_hf(
        nucleus="40Ca", force="SLy4", e_tol=1e-13, mixing=m, max_iter=1000
    )
    for m in MIXERS
}
ref = tight["linear"]
for m, r in tight.items():
    print(
        "%-8s passes %3d   |dE/A| = %.1e MeV   |dr_ch| = %.1e fm"
        % (
            m,
            r.iterations,
            abs(r.e_per_a - ref.e_per_a),
            abs(r.r_ch - ref.r_ch),
        )
    )
    assert abs(r.e_per_a - ref.e_per_a) < 1e-12
    assert abs(r.r_ch - ref.r_ch) < 1e-11

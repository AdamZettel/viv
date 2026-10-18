# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
r"""
Landau damping at four wavelengths
==================================

Perturb a Maxwellian plasma with a small density wave and follow the electric
field as it decays without any collisions. `vp.plasma.landau` runs a
Vlasov-Poisson solver (`plasma/landau.viv`: Cheng and Knorr's split step, on
a $16 \times 256$ phase-space grid) and fits the decay rate and frequency to the peaks of the field
energy. The exact values are the root of the Langmuir dispersion relation for
a Maxwellian, which the program also computes, from the plasma dispersion
function; the fitted rate matches it to better than one part in $10^{3}$ at every $k$.
"""

# %%
# ## The field energy
#
# Each run starts from $f = (1 + \alpha \cos kx)\, e^{-v^2/2} / \sqrt{2\pi}$ with
# $\alpha = 10^{-4}$, so the wave stays linear. Lengths are in Debye lengths and
# times in inverse plasma frequencies. The field energy oscillates at twice
# the wave frequency, and its peaks fall as $e^{2\gamma t}$. Shorter waves
# (larger $k\lambda_D$) are damped faster, because more electrons move near
# the phase velocity $\omega/k$ and take energy from the wave.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

ks = [0.3, 0.4, 0.5, 0.6]
t_end = {0.3: 80.0, 0.4: 60.0, 0.5: 40.0, 0.6: 30.0}
runs = {}
for k in ks:
    runs[k] = vp.plasma.landau(
        setup="k05", k=k, t_end=t_end[k], t_fit0=8.0, t_fit1=t_end[k]
    )
    r = runs[k]
    print(
        "k = %.1f   gamma %.6f  exact %.6f  (rel %.1e)   omega %.5f  exact "
        "%.5f   %d peaks"
        % (
            k,
            r.gamma,
            r.gamma_exact,
            abs(r.gamma / r.gamma_exact - 1),
            r.omega,
            r.omega_exact,
            r.npeaks,
        )
    )

fig = figure(
    "Electric field energy", "$t$ ($1/\\omega_p$)", "$W_E/W_E(0)$"
)
for k in ks:
    r = runs[k]
    w = np.asarray(r.wfield)
    fig.plot(r.time[::2], w[::2] / w[0], "k = %.1f" % k)
    fig.style("k = %.1f" % k, "nomarker")
fig.yscale("log")

# %%
# ## Fitted against exact
#
# The rate from the simulation's peaks against the root of the dispersion
# relation, and the exact curve over a finer set of k from the same
# dispersion function (`vp.special.plasma_z`): the root of
# $1 + \bigl(1 + \zeta Z(\zeta)\bigr)/k^2 = 0$ with $\zeta = \omega/(k\sqrt{2})$, solved
# here by Newton's method with the derivative $Z'$.


def landau_root(k, omega0):
    w = complex(omega0, -0.01)
    for _ in range(50):
        zeta = w / (k * np.sqrt(2.0))
        z = vp.special.plasma_z(zeta=zeta)
        eps = 1.0 + (1.0 + zeta * z.Z) / k**2
        deps = (z.Z + zeta * z.Zp) / k**2 / (k * np.sqrt(2.0))
        w = w - eps / deps
    return w


kf = np.linspace(0.25, 0.65, 17)
roots = []
w0 = 1.1
for k in kf:
    w = landau_root(k, w0)
    roots.append(w)
    w0 = w.real
for k in ks:
    w = landau_root(k, runs[k].omega_exact)
    print(
        "k = %.1f   Newton root: omega %.6f  gamma %.6f   the program's "
        "exact gamma %.6f" % (k, w.real, w.imag, runs[k].gamma_exact)
    )

fig = figure(
    "Damping rate against wavenumber", "$k\\lambda_D$", "$-\\gamma$ ($\\omega_p$)"
)
fig.plot(kf, [-w.imag for w in roots], "dispersion relation")
fig.plot(ks, [-runs[k].gamma for k in ks], "Vlasov simulation")
fig.style("dispersion relation", "s2 dash nomarker")
fig.style("Vlasov simulation", "s1 noline circle")
fig.yscale("log")

# %%
# ## How it's checked
#
# Besides the rates, the solver reports how well it conserves what the
# Vlasov equation conserves: the number of particles, the L2 norm of f and
# the total energy, as relative drifts over the run.

for k in ks:
    r = runs[k]
    print(
        "k = %.1f   mass drift %.1e   L2 drift %.1e   energy drift %.1e"
        % (k, r.mass_drift, r.l2_drift, r.energy_drift)
    )
print("check() at the k05 deck:", vp.plasma.landau(setup="k05").check().passed)

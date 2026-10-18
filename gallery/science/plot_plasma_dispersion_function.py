# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
r"""
The Faddeeva and plasma dispersion functions on the complex plane
=================================================================

Map the Faddeeva function $w(z) = e^{-z^2} \operatorname{erfc}(-iz)$ and the plasma
dispersion function $Z(\zeta) = i\sqrt{\pi}\, w(\zeta)$ over a rectangle of the
complex plane, and measure their accuracy against SciPy and against mpmath
at 30 digits. `vp.special.faddeeva_w` and `vp.special.plasma_z` evaluate one
point per call; for a whole grid this page loads a ten-line Viv file,
`gallery/science/zgrid.viv`, that loops the same library function
(`lib/zfunc.viv`, Weideman's rational expansion) over the grid in one call.
"""

# %%
# ## A grid in one call
#
# `vp.load` compiles the file once and caches the library. `w_grid(x, y)`
# returns the real and imaginary parts of w at every x + iy, row by row.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

zg = vp.load("gallery/science/zgrid.viv")
x = np.linspace(-6.0, 6.0, 121)
y = np.linspace(-1.0, 4.0, 51)
re_w, im_w = zg.w_grid(x, y)
W = (re_w + 1j * im_w).reshape(len(y), len(x))
print("grid: %d x %d points" % W.shape)
print(
    "w(0) = %s   w(i) = %.15f (exp(1) erfc(1) = %.15f)"
    % (W[10, 60], W[20, 60].real, np.exp(1) * 0.15729920705028513)
)
print(
    "one point, vp.special.faddeeva_w(z=2+1j):",
    vp.special.faddeeva_w(z=2.0 + 1.0j).w,
)

# %%
# ## $|w(z)|$ and the phase of $Z$
#
# In the upper half plane $|w|$ is at most 1 and decays like $1/(\sqrt{\pi}\,|z|)$.
# Below the real axis $e^{-z^2}$ takes over and $|w|$ grows fast, which is why
# the plot stops at $\operatorname{Im} z = -1$. On the real axis the plasma dispersion
# function's imaginary part is $\sqrt{\pi}\, e^{-x^2}$, and its real part is odd,
# with extremes of about 1.08 at $x = -0.92$ and $-1.08$ at $x = 0.92$ (on the
# grid, at the nearest point, $x = -0.9$). The Landau damping example solves
# the Langmuir dispersion relation with this function.

fig = figure(
    "$\\log_{10} |w(x + iy)|$", "$\\mathrm{Re}\\,z$", "$\\mathrm{Im}\\,z$", window=(-6.0, 6.0, -1.0, 4.0)
)
fig.heatmap(np.log10(np.abs(W)), "log10 |w|")
fig.origin("lower")
fig.clim(-1.5, 1.5)
fig.colormap("diverging")

re_z, im_z = zg.z_grid(x, y)
Z = (re_z + 1j * im_z).reshape(len(y), len(x))
fig = figure(
    "$\\mathrm{Re}\\,Z(x + iy)$, the real part of the plasma dispersion function",
    "$\\mathrm{Re}\\,\\zeta$",
    "$\\mathrm{Im}\\,\\zeta$",
    window=(-6.0, 6.0, -1.0, 4.0),
)
fig.heatmap(np.clip(Z.real, -2.0, 2.0), "$\\mathrm{Re}\\,Z$")
fig.origin("lower")
fig.colormap("diverging")

fig = figure("$Z$ on the real axis", "$x$", "$Z(x)$")
fig.plot(x, Z[10].real, "$\\mathrm{Re}\\,Z$")
fig.plot(x, Z[10].imag, "$\\mathrm{Im}\\,Z = \\sqrt{\\pi}\\,e^{-x^2}$")
fig.style("$\\mathrm{Re}\\,Z$", "s1 nomarker")
fig.style("$\\mathrm{Im}\\,Z = \\sqrt{\\pi}\\,e^{-x^2}$", "s2 dash nomarker")
print(
    "Re Z(x) is largest at x = %.2f: %.4f"
    % (x[10 + np.argmax(Z[10, 10:-10].real)], Z[10, 10:-10].real.max())
)

# %%
# ## Accuracy
#
# Against SciPy's `wofz` on the whole grid, and against mpmath at 30 digits
# on every fifth row and fourth column of it, as relative errors |w - w_ref| /
# |w_ref|. The library's stated tolerance on this rectangle is 2e-14
# (`lib/zfunc_ref.py` measured it).

from scipy.special import wofz
import mpmath

rel_scipy = np.abs(W - wofz(x[None, :] + 1j * y[:, None])) / np.abs(W)
mpmath.mp.dps = 30
sub = [(k, j) for k in range(0, len(y), 5) for j in range(0, len(x), 4)]
rel_mp = np.zeros((len(range(0, len(y), 5)), len(range(0, len(x), 4))))
for k, j in sub:
    z = mpmath.mpc(x[j], y[k])
    exact = mpmath.exp(-z * z) * mpmath.erfc(-1j * z)
    rel_mp[k // 5, j // 4] = float(
        abs(mpmath.mpc(W[k, j].real, W[k, j].imag) - exact) / abs(exact)
    )
print("worst relative error against SciPy wofz: %.1e" % rel_scipy.max())
print(
    "worst relative error against mpmath (30 digits), %d points: %.1e"
    % (len(sub), rel_mp.max())
)

fig = figure(
    "Relative error against mpmath, $\\log_{10}$",
    "$\\mathrm{Re}\\,z$",
    "$\\mathrm{Im}\\,z$",
    window=(-6.0, 6.0, -1.0, 4.0),
)
fig.heatmap(np.log10(np.maximum(rel_mp, 1e-17)), "log10 error")
fig.origin("lower")
fig.clim(-17.0, -13.0)

# %%
# ## How it's checked
#
# The `faddeeva_w` and `plasma_z` functions agree with SciPy to 2e-14
# relative on this rectangle in their recorded cases, and `check()` reruns
# them. The grid file carries one test of its own, w(0) = 1.

print(
    "check():",
    vp.special.faddeeva_w(z=2.0 + 1.0j).check().passed,
    vp.special.plasma_z(zeta=1.0 + 0.5j).check().passed,
)

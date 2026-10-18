# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-upwind-advection-smears-by-its-modified-equation.viv.

u_t + u_x = 0 on the periodic unit interval, N = 200 cells of width h, cell
centers (i + 1/2) h, a Gaussian exp(-((x - 0.3)/0.05)^2), first-order upwind
u_i <- u_i - nu (u_i - u_{i-1}) for one transit (200/nu steps). A separate numpy
implementation: mass, extrema, peak, and the second moment of the pulse, taken
with displacements wrapped into [-1/2, 1/2), against the modified equation's
diffusion coefficient D = h (1 - nu) / 2.
Run: python3 posts/refs/upwind-advection-smears-by-its-modified-equation.py > posts/refs/upwind-advection-smears-by-its-modified-equation.out
"""
import numpy as np

N = 200
h = 1.0 / N
x = (np.arange(N) + 0.5) * h
X0, W = 0.3, 0.05
u0 = np.exp(-((x - X0) / W) ** 2)
sigma0_sq = W * W / 2.0


def run(nu):
    steps = round(1.0 / (nu * h))
    u = u0.copy()
    for _ in range(steps):
        u = u - nu * (u - np.roll(u, 1))
    return u, steps


def moments(u, center):
    d = (x - center + 0.5) % 1.0 - 0.5
    m = u.sum()
    mean = (d * u).sum() / m
    var = ((d - mean) ** 2 * u).sum() / m
    return center + mean, var


mean0, var0 = moments(u0, X0)
print(f"initial: mass {u0.sum() * h!r}, peak {u0.max()!r}, variance {var0!r} (w^2/2 = {sigma0_sq!r})")
for nu in (0.25, 0.4, 0.5, 0.625, 0.8, 1.0):
    u, steps = run(nu)
    T = steps * nu * h
    D = h * (1.0 - nu) / 2.0
    mean1, var1 = moments(u, X0)
    pred_peak = np.sqrt(var0 / (var0 + 2.0 * D * T))
    print(f"nu = {nu}: steps {steps}, T {T!r}")
    print(f"  mass change {u.sum() * h - u0.sum() * h:.3e}, max {u.max()!r}, min {u.min():.3e}")
    print(f"  mean {mean1!r}; variance growth {var1 - var0!r}, 2 D T = {2 * D * T!r}, differ by {var1 - var0 - 2 * D * T:.2e}")
    print(f"  peak {u.max()!r}; modified-equation peak sqrt(var0/(var0 + 2DT)) = {pred_peak!r}, relative gap {u.max() / pred_peak - 1:.3e}")
    if nu == 0.5:
        # the modified equation's Gaussian, sampled at the cell nearest the center
        var1g = var0 + 2 * D * T
        i = int(np.argmax(u))
        g = np.sqrt(var0 / var1g) * np.exp(-((x[i] - X0) ** 2) / (2 * var1g))
        print(f"  peak cell x = {x[i]!r}; the Gaussian there {g!r}; measured - Gaussian {u.max() - g:.3e}")

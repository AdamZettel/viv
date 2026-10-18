# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-burgers-in-nonconservative-form-puts-the-shock-in-the-wrong-place.viv.

Inviscid Burgers, u_t + (u^2/2)_x = 0, on [-1/2, 3/2] with N cells, Riemann
data u = 1 for x < 0 and u = u_R for x > 0, ghost cells holding the two states,
dt = h/2 (CFL 1/2 at the largest speed, 1), to t = 1.
  conservative: Godunov's flux max(f(max(ul, 0)), f(min(ur, 0))), f = u^2/2;
  nonconservative: u_t + u u_x = 0 upwinded from the left (u >= 0 here).
Shock position two ways: the u = (1 + u_R)/2 crossing, linearly interpolated,
and from the mass, x_s = (M - u_R * 3/2 - 1/2) / (1 - u_R) for a step between
x = -1/2 and 3/2 holding mass M. Speed = (x_s(1) - x_s(1/2)) / (1/2).
Rankine-Hugoniot: s = (f(1) - f(u_R)) / (1 - u_R) = (1 + u_R)/2.
Run: python3 posts/refs/burgers-in-nonconservative-form-puts-the-shock-in-the-wrong-place.py > posts/refs/burgers-in-nonconservative-form-puts-the-shock-in-the-wrong-place.out
"""
import numpy as np

XL, XR = -0.5, 1.5
UL = 1.0


def godunov(ul, ur):
    a = np.maximum(ul, 0.0)
    b = np.minimum(ur, 0.0)
    return np.maximum(0.5 * a * a, 0.5 * b * b)


def run(n, ur, conservative):
    h = (XR - XL) / n
    x = XL + (np.arange(n) + 0.5) * h
    u = np.where(x < 0.0, UL, ur)
    dt = 0.5 * h
    steps = round(1.0 / dt)
    out = {}
    for s in range(1, steps + 1):
        if conservative:
            F = godunov(np.append(UL, u), np.append(u, ur))
            u = u - dt / h * (F[1:] - F[:-1])
        else:
            um = np.append(UL, u[:-1])
            u = u - dt / h * u * (u - um)
        if 2 * s == steps or s == steps:
            out[s * dt] = u.copy()
    return x, h, out


def crossing(x, h, u, mid):
    i = int(np.argmax(u < mid))
    return x[i - 1] + h * (u[i - 1] - mid) / (u[i - 1] - u[i])


def by_mass(h, u, ur):
    m = u.sum() * h
    return (m - ur * XR + UL * XL) / (UL - ur)


for ur in (0.25, 0.0):
    rh = (UL + ur) / 2
    print(f"u_R = {ur}: Rankine-Hugoniot speed {rh}")
    for n in (400, 800, 1600):
        for cons in (True, False):
            x, h, out = run(n, ur, cons)
            p = {t: (crossing(x, h, u, rh), by_mass(h, u, ur)) for t, u in out.items()}
            sc = (p[1.0][0] - p[0.5][0]) / 0.5
            sm = (p[1.0][1] - p[0.5][1]) / 0.5
            inflow = 1.0 * (0.5 * UL * UL - 0.5 * ur * ur)
            m0 = UL * (0 - XL) + ur * XR
            dm = out[1.0].sum() * h - m0
            print(f"  N = {n} {'conservative   ' if cons else 'nonconservative'}: crossing at t=1 {p[1.0][0]!r}, "
                  f"speed by crossing {sc!r}, by mass {sm!r}; mass change - inflow {dm - inflow:.3e}")

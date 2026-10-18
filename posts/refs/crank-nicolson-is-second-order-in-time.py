# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference for posts/2026-09-25-crank-nicolson-is-second-order-in-time.viv.

u_t = u_xx on [0, 1], u = 0 at both ends, u(x, 0) = x(1 - x). N = 199 interior
points, h = 1/200, to T = 0.1. Crank-Nicolson (theta = 1/2) and backward Euler
(theta = 1), each step a banded solve by scipy. Two references:
  the PDE's Fourier series  sum over odd k of 8/(pi^3 k^3) sin(k pi x) exp(-k^2 pi^2 t);
  the semi-discrete solution exp(-A t) u0, A the second-difference matrix, by
  scipy's expm, which the post writes as a discrete sine series instead.
The time-stepping error is measured against the second, so the space error
does not enter the order in dt.
Run: python3 posts/refs/crank-nicolson-is-second-order-in-time.py > posts/refs/crank-nicolson-is-second-order-in-time.out
"""
import numpy as np
from scipy.linalg import expm, solve_banded

N = 199
h = 1.0 / (N + 1)
x = np.arange(1, N + 1) * h
T = 0.1
u0 = x * (1.0 - x)

A = (np.diag(2.0 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / h**2
semi = expm(-A * T) @ u0
pde = sum(8.0 / (np.pi**3 * k**3) * np.sin(k * np.pi * x) * np.exp(-k * k * np.pi**2 * T) for k in range(1, 2001, 2))
space_error = np.abs(semi - pde).max()
print(f"space error max|semi-discrete - PDE series| at T: {space_error!r}")

# the post's closed form for the semi-discrete solution: a discrete sine series
K = np.arange(1, N + 1, 2)
lam = 4.0 / h**2 * np.sin(K * np.pi * h / 2) ** 2
ck = 4.0 * h / np.tan(K * np.pi * h / 2) / lam
series = (ck[:, None] * np.sin(np.pi * np.outer(K, x)) * np.exp(-lam * T)[:, None]).sum(0)
print(f"discrete sine series vs expm: {np.abs(series - semi).max():.2e}; at t = 0 vs u0: "
      f"{np.abs((ck[:, None] * np.sin(np.pi * np.outer(K, x))).sum(0) - u0).max():.2e}")


def march(dt, theta):
    r = dt / h**2
    ab = np.zeros((3, N))
    ab[0, 1:] = -theta * r
    ab[1] = 1.0 + 2.0 * theta * r
    ab[2, :-1] = -theta * r
    u = u0.copy()
    for _ in range(round(T / dt)):
        lap = np.concatenate(([0.0], u[:-1])) + np.concatenate((u[1:], [0.0])) - 2.0 * u
        u = solve_banded((1, 1), ab, u + (1.0 - theta) * r * lap)
    return u


prev = None
for j in range(6):
    dt = 0.02 / 2**j
    cn = march(dt, 0.5)
    be = march(dt, 1.0)
    ecn = np.abs(cn - semi).max()
    ebe = np.abs(be - semi).max()
    epde = np.abs(cn - pde).max()
    line = f"dt {dt!r}: CN {ecn!r}  BE {ebe!r}  CN vs PDE series {epde!r}"
    if prev:
        line += f"  order CN {np.log2(prev[0] / ecn):.4f} BE {np.log2(prev[1] / ebe):.4f}"
    print(line)
    # the slowest mode alone: CN multiplies it by g = (1 - dt lam/2)/(1 + dt lam/2) per step
    g = (1 - dt * lam / 2) / (1 + dt * lam / 2)
    amp = ck * (g ** round(T / dt) - np.exp(-lam * T))
    print(f"    CN error in the k = 1 mode alone {abs(amp[0]):.4e}; modes with dt*lam > 2: k >= {K[np.argmax(dt * lam > 2)]}")
    prev = (ecn, ebe)

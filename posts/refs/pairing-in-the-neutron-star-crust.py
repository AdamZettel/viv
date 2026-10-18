# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-pairing-in-the-neutron-star-crust.viv.

Two independent checks of the post's Bogoliubov-de Gennes (BdG) solver.

1. The slab cell, the same model solved with numpy: the BdG matrix built in
   block form [[h - mu, Delta], [Delta, -(h - mu)]] (the post interleaves u
   and v site by site instead), diagonalized by numpy.linalg.eigh, and the
   gap equation iterated with the same linear mixing to a tolerance 100 times
   tighter than the post's.

2. The uniform gas on a ring (V = 0, periodic), where the BdG equations are
   diagonal in momentum and the self-consistent gap solves the lattice BCS
   equation
       1 = (g / dx) (1/N) sum_k 1 / (2 E_k),
       E_k = sqrt(xi_k^2 + Delta^2),  xi_k = 2 tau (1 - cos k dx) - mu,
   with k dx = 2 pi n / N. scipy's brentq solves it for Delta. No matrix is
   diagonalized.

    python3 posts/refs/pairing-in-the-neutron-star-crust.py > posts/refs/pairing-in-the-neutron-star-crust.out
"""
import numpy as np
import scipy
from scipy.optimize import brentq

T_KIN = 20.735   # hbar^2 / 2 m_n  [MeV fm^2]
N = 60
DX = 0.5
MU = 3.0
G = 15.0
TAU = T_KIN / DX**2

x = (np.arange(N) + 0.5) * DX
V = -40.0 / (1.0 + np.exp((np.abs(x - 15.0) - 5.0) / 0.6))


def bdg(hd, delta, periodic=False):
    h = np.diag(hd) - TAU * (np.eye(N, k=1) + np.eye(N, k=-1))
    if periodic:
        h[0, N - 1] = h[N - 1, 0] = -TAU
    H = np.block([[h, np.diag(delta)], [np.diag(delta), -h]])
    w, W = np.linalg.eigh(H)
    neg = w < 0
    u = W[:N, neg]
    v = W[N:, neg]
    rho = np.sum(u * u, axis=1)
    kappa = -np.sum(u * v, axis=1)
    return rho, kappa


def scf(hd, periodic=False, tol=1e-8):
    delta = np.full(N, 2.0)
    it = 0
    while True:
        rho, kappa = bdg(hd, delta, periodic)
        new = G / DX * kappa
        dmax = np.max(np.abs(new - delta))
        delta = 0.6 * new + 0.4 * delta
        it += 1
        if dmax < tol:
            return delta, rho, it


print(f"numpy {np.__version__}, scipy {scipy.__version__}")
hd = V + 2 * TAU - MU
delta, rho, it = scf(hd)
print(f"slab cell: {it} iterations")
print(f"Delta[5]  (x = {x[5]:.2f} fm, neutron gas) = {delta[5]:.8f} MeV")
print(f"Delta[29] (x = {x[29]:.2f} fm, inside the slab) = {delta[29]:.8f} MeV")
print(f"min Delta = {delta.min():.8f}, max Delta = {delta.max():.8f} MeV")
print(f"neutrons in the cell (sum of v^2... here the 'rho' convention of the post) = {rho.sum():.8f}")

# 2. uniform ring
ka = 2 * np.pi * np.arange(N) / N
xi = 2 * TAU * (1 - np.cos(ka)) - MU


def gap_eq(d):
    return G / DX / N * np.sum(1.0 / (2.0 * np.sqrt(xi**2 + d**2))) - 1.0


d_bcs = brentq(gap_eq, 1e-6, 20.0, xtol=1e-14)
print(f"uniform ring, lattice BCS equation: Delta = {d_bcs:.10f} MeV")
d_ring, _, it = scf(np.full(N, 2 * TAU - MU), periodic=True, tol=1e-10)
print(f"uniform ring, BdG by eigh: Delta = {d_ring.mean():.10f} MeV (spread {d_ring.max()-d_ring.min():.1e})")

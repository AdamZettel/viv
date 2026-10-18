# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-graphenes-dirac-cones.viv.

Nearest-neighbor tight-binding graphene, a different way from the post: the
2x2 complex Bloch Hamiltonian is built as a matrix and diagonalized by
numpy.linalg.eigvalsh (the post uses the closed form E = +-|f(k)|), and the
Fermi velocity is the slope of a least-squares line through the upper band
at 20 points within 1e-3 / a of K, averaged over directions; mpmath gives
the band at one point to 30 digits. Units: energies in t, lengths in the
carbon-carbon distance a, hbar = 1.

    python3 posts/refs/graphenes-dirac-cones.py > posts/refs/graphenes-dirac-cones.out

Output recorded 2026-09-25 (numpy 1.24.4, mpmath 1.3.0) is in
posts/refs/graphenes-dirac-cones.out.
"""
import mpmath as mp
import numpy as np

S3 = np.sqrt(3.0)
DELTAS = np.array([[0.5, S3 / 2], [0.5, -S3 / 2], [-1.0, 0.0]])
GAMMA = np.array([0.0, 0.0])
K = np.array([2 * np.pi / 3, 2 * np.pi / (3 * S3)])
M = np.array([2 * np.pi / 3, 0.0])


def bands(k):
    f = -np.sum(np.exp(1j * DELTAS @ k))
    H = np.array([[0.0, f], [np.conj(f), 0.0]])
    return np.linalg.eigvalsh(H)


for name, k in [("Gamma", GAMMA), ("M", M), ("K", K)]:
    lo, hi = bands(k)
    print(f"{name}: E = {lo!r}, {hi!r}")

mid = 0.5 * (K + M)
print(f"halfway from K to M: upper band {bands(mid)[1]!r}")
print(f"halfway from Gamma to K: upper band {bands(0.5 * K)[1]!r}")

# the same point to 30 digits
mp.mp.dps = 30
kx, ky = mp.mpf(2) * mp.pi / 3 * mp.mpf("0.5"), mp.mpf(2) * mp.pi / (3 * mp.sqrt(3)) * mp.mpf("0.5")
d = [(mp.mpf("0.5"), mp.sqrt(3) / 2), (mp.mpf("0.5"), -mp.sqrt(3) / 2), (mp.mpf(-1), mp.mpf(0))]
f = sum(mp.expj(kx * dx + ky * dy) for dx, dy in d)
print("halfway from Gamma to K, mpmath:", mp.nstr(abs(f), 20))

# Fermi velocity: least-squares slope of E_+(K + q u) against q, q in (0, 1e-3],
# for 12 directions u
slopes = []
for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    u = np.array([np.cos(phi), np.sin(phi)])
    q = np.linspace(5e-5, 1e-3, 20)
    E = np.array([bands(K + qq * u)[1] for qq in q])
    A = np.vstack([q, np.ones_like(q)]).T
    slopes.append(np.linalg.lstsq(A, E, rcond=None)[0][0])
slopes = np.array(slopes)
print(f"slope at K over 12 directions, a least-squares line through {len(q)} points in each: mean {slopes.mean():.10f}, spread {slopes.max() - slopes.min():.2e}")
print("3/2 =", 1.5)

# the second-order term along K -> M and K -> Gamma: (E - 1.5 q) / q^2 at q = 1e-3
for name, target in [("M", M), ("Gamma", GAMMA)]:
    u = (target - K) / np.linalg.norm(target - K)
    q = 1e-3
    print(f"(E - 1.5 q) / q^2 toward {name} at q = 1e-3: {(bands(K + q * u)[1] - 1.5 * q) / q**2:.6f}")

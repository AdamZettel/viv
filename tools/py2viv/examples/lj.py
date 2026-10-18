# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Lennard-Jones energy of 4096 atoms on a simple cubic lattice: 8,386,560 pairs.

A Lennard-Jones kernel: scalar indexing into an
(n, 3) NumPy array, a double loop over pairs, reduced units (epsilon = sigma = 1),
no cutoff. Every operation is +, -, * or /, so CPython and every Viv engine
round identically and the energy must agree to the last bit.
"""
import numpy as np


def lj_energy(pos, n):
    e = 0.0
    for i in range(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        for j in range(i + 1, n):
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s6 = 1.0 / (r2 * r2 * r2)
            e += 4.0 * (s6 * s6 - s6)
    return e


n_side = 16
spacing = 1.1
n = n_side * n_side * n_side
pos = np.zeros((n, 3))
p = 0
for ix in range(n_side):
    for iy in range(n_side):
        for iz in range(n_side):
            pos[p, 0] = ix * spacing
            pos[p, 1] = iy * spacing
            pos[p, 2] = iz * spacing
            p += 1

energy = lj_energy(pos, n)
print(energy)


def test_two_atoms_at_the_minimum():
    r = 2.0 ** (1.0 / 6.0)
    two = np.zeros((2, 3))
    two[1, 0] = r
    assert abs(lj_energy(two, 2) - (-1.0)) < 1e-12

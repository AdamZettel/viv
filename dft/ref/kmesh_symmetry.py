# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Which k-meshes keep a cubic crystal cubic? Silicon's valence-band top is a
TRIPLET; a mesh that is not closed under the point group gives a density that is
not cubic, and splits it.

    python3 kmesh_symmetry.py
"""
import time
import numpy as np
from pwdft_ref import scf, bands
from materials import build
HA = 27.211386245988
FCC_SHIFTS = [(.5, .5, .5), (.5, 0, 0), (0, .5, 0), (0, 0, .5)]

def mesh(n, shifts):
    """k = (i + s) / n for every shift s, halved by time reversal."""
    pts = {}
    for s in shifts:
        for i in range(n):
            for j in range(n):
                for l in range(n):
                    k = (np.array([i, j, l]) + np.array(s)) / n
                    k = k - np.round(k)
                    k[np.isclose(k, -0.5)] = 0.5
                    mk = -k
                    mk[np.isclose(mk, -0.5)] = 0.5
                    key, mkey = tuple(np.round(k, 9)), tuple(np.round(mk, 9))
                    if mkey in pts and mkey != key:
                        pts[mkey] += 1
                    else:
                        pts[key] = pts.get(key, 0) + 1
    ks = np.array(list(pts.keys()))
    w = np.array(list(pts.values()), float)
    return ks, w / w.sum()

cr, ecut, _, smear, nb, _ = build('Si')
for label, n, shifts in (("shifted 2x2x2 (the old mesh)", 2, [(.5, .5, .5)]),
                         ("shifted 4x4x4", 4, [(.5, .5, .5)]),
                         ("Gamma-centred 2x2x2", 2, [(0, 0, 0)]),
                         ("Gamma-centred 4x4x4", 4, [(0, 0, 0)]),
                         ("fcc 4-shift 2x2x2", 2, FCC_SHIFTS),
                         ("fcc 4-shift 4x4x4", 4, FCC_SHIFTS)):
    ks, w = mesh(n, shifts)
    t0 = time.time()
    out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.6, anderson=3)
    g = bands(cr, ecut, out['vtot_g'], out['dims'], [0, 0, 0], nb) * HA
    x = bands(cr, ecut, out['vtot_g'], out['dims'], [0.425, 0, 0.425], nb) * HA
    print(f"{label:<30} {len(ks):4d} k  E = {out['etot']:.6f} Ha   triplet split {1000*(g[3]-g[1]):7.3f} meV   "
          f"width {g[3]-g[0]:.3f}  gap(0.85 X) {x[4]-g[3]:.3f} eV   {time.time()-t0:4.0f} s", flush=True)

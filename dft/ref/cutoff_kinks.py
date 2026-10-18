# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A fixed plane-wave cutoff makes E_n(k) discontinuous: the basis {G : |k+G|^2/2 < Ecut}
changes as k moves, and AT a zone-boundary point a whole shell of G-vectors sits on the
cutoff at once. The size of the jump is a direct, local measure of how far the cutoff is
from converged. For each material and cutoff: the band energy AT L and at X, against the
value a parabola through the next three path points extrapolates to.

    python3 cutoff_kinks.py Ge 6,10,14,18

READ IT WITH CARE. A jump that SHRINKS as the cutoff rises is a basis discontinuity. One
that does not is physics: where bands stick together or cross at the symmetry point (X in
the diamond structure, the free-electron crossings of aluminum) a band SORTED by energy has
a real kink there, and a parabola through its neighbours means nothing. Measured:

    Ge    6 Ha   L 21.3 meV   X 24.0          Si   6 Ha   L 7.7   X 0.4
    Ge   10 Ha   L  2.6       X  6.9          Si  10 Ha   L 4.4   X 7.3   (sticking)
    Ge   14 Ha   L  2.7       X  0.7          GaAs 10-18 Ha   X ~10, flat (sticking)

So germanium moved from 6 Ha to 10; the others stayed.
"""
import sys
import numpy as np
from pwdft_ref import scf, bands, mp_grid, fft_dims
from materials import build
HA = 27.211386245988
name = sys.argv[1]
cr, ecut0, nk, smear, nb, _ = build(name)
ks, w = mp_grid((nk, nk, nk))
nv = cr.nelec // 2
L, G, X = np.array([.5, .5, .5]), np.zeros(3), np.array([.5, 0, .5])
h = 1.0 / 30
for ecut in [float(x) for x in sys.argv[2].split(',')]:
    pts = [L + (G - L) * h * j for j in range(4)] + [X + (G - X) * h * j for j in range(4)]
    dims = fft_dims(cr, ecut, list(ks) + pts)
    out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.6, anderson=3, dims=dims)
    E = np.array([bands(cr, ecut, out['vtot_g'], dims, k, nb) for k in pts]) * HA
    msg = f"{name} ecut {ecut:5.1f} Ha  grid {dims[0]:2d}"
    for label, o in (("L", 0), ("X", 4)):
        worst = 0.0
        for b in range(min(nb, nv + 2)):
            e0, e1, e2, e3 = E[o, b], E[o + 1, b], E[o + 2, b], E[o + 3, b]
            extrap = 3 * e1 - 3 * e2 + e3            # the quadratic through points 1, 2, 3, at 0
            worst = max(worst, abs(e0 - extrap))
        msg += f"   worst jump at {label}: {1000 * worst:7.2f} meV"
    print(msg, flush=True)

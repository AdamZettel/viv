# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""How hard is copper? Band energies vs plane-wave cutoff, numpy reference."""
import sys, time
import numpy as np
from pwdft_ref import scf, bands, mp_grid, pw_basis
from materials import build
HA = 27.211386245988
cr, _, _, smear, nb, _ = build('Cu')
nk = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ks, w = mp_grid((nk, nk, nk))
for ecut in [float(x) for x in sys.argv[1].split(',')]:
    t0 = time.time()
    out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.5, anderson=3)
    npw = len(pw_basis(cr, [0, 0, 0], ecut)[0])
    mu = out['mu']
    eg = (bands(cr, ecut, out['vtot_g'], out['dims'], [0, 0, 0], nb) - mu) * HA
    ex = (bands(cr, ecut, out['vtot_g'], out['dims'], [0.5, 0, 0.5], nb) - mu) * HA
    print(f"ecut {ecut:5.1f} Ha  npw {npw:5d}  grid {out['dims'][0]:3d}  iters {out['iters']:2d}  {time.time()-t0:6.1f} s"
          f"  | Gamma: s-bottom {eg[0]:7.3f}  d {eg[1]:6.3f} {eg[4]:6.3f}  | X: d-bottom {ex[0]:6.3f}  d-top {ex[4]:6.3f}", flush=True)

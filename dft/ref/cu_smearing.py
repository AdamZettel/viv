# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Copper: how the Fermi-Dirac width and the k-mesh move the bands relative to E_F.
A reduced cutoff keeps this cheap; the d bands are not converged in it, but the
QUESTION -- does the smearing tail reach the d bands and push E_F up -- is about
occupations, not about the basis.

    python3 cu_smearing.py 30 4:0.02,4:0.01,4:0.005

Measured at 30 Ha on a 32-point mesh (every band energy is relative to E_F, in eV):

    width 0.020 Ha   E_F 14.750   Gamma12 -1.525   X5 -2.044
    width 0.010 Ha   E_F 14.490   Gamma12 -1.155   X5 -1.666
    width 0.005 Ha   E_F 14.495   Gamma12 -1.118   X5 -1.626

Halving 0.02 moves everything 0.37 eV; halving again moves it 0.04. Hence 0.01.
"""
import sys, time
import numpy as np
from pwdft_ref import scf, bands, mp_grid
from materials import build
HA = 27.211386245988
cr, _, _, _, nb, _ = build('Cu')
ecut = float(sys.argv[1])
for spec in sys.argv[2].split(','):
    nk, smear = int(spec.split(':')[0]), float(spec.split(':')[1])
    ks, w = mp_grid((nk, nk, nk))
    t0 = time.time()
    out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.6, anderson=3)
    mu = out['mu']
    g = (bands(cr, ecut, out['vtot_g'], out['dims'], [0, 0, 0], nb) - mu) * HA
    x = (bands(cr, ecut, out['vtot_g'], out['dims'], [0.5, 0, 0.5], nb) - mu) * HA
    l = (bands(cr, ecut, out['vtot_g'], out['dims'], [0.5, 0.5, 0.5], nb) - mu) * HA
    print(f"nk {nk}  smear {smear:6.3f} Ha  {len(ks):3d} k  iters {out['iters']:2d}  {time.time()-t0:5.0f} s  E_F {mu*HA:8.4f} eV"
          f" | G1 {g[0]:7.3f}  G25' {g[1]:6.3f}  G12 {g[4]:6.3f} | X1 {x[0]:6.3f}  X5 {x[4]:6.3f} | L3 {l[4]:6.3f}  L2' {l[5]:6.3f}", flush=True)

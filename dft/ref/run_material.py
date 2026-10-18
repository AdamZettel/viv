# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Self-consistent LDA for one of the usual suspects, and its band energies."""
import sys, time
import numpy as np
from pwdft_ref import scf, bands, mp_grid, pw_basis
from materials import build
HA = 27.211386245988
name = sys.argv[1]
cr, ecut, nk, smear, nb, names = build(name)
if len(sys.argv) > 2: ecut = float(sys.argv[2])
if len(sys.argv) > 3: nk = int(sys.argv[3])
ks, w = mp_grid((nk, nk, nk))
t0 = time.time()
out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False)
npw = len(pw_basis(cr, [0, 0, 0], ecut)[0])
print(f"{name}: {len(ks)} k-points ({nk}^3 x 4 shifts, time-reversal halved), ecut {ecut} Ha, "
      f"{npw} PWs at Gamma, grid {out['dims']}, {out['iters']} iterations, {time.time()-t0:.1f} s")
print(f"  F (no Ewald) = {out['etot']:.10f} Ha" + (f"   E_F = {out['mu']*HA:.4f} eV" if out['mu'] is not None else ""))
def at(k): return bands(cr, ecut, out['vtot_g'], out['dims'], k, nb) * HA
G, X, L = [0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5]
eg, ex, el = at(G), at(X), at(L)
if smear > 0:
    ref = out['mu'] * HA
    print(f"  Gamma_1 (band bottom) {eg[0]-ref:8.3f} eV from E_F")
    print(f"  X  {np.round(ex[:4]-ref, 3)}")
    print(f"  L  {np.round(el[:4]-ref, 3)}")
else:
    nocc = cr.nelec // 2
    vbm = max(eg[nocc-1], ex[nocc-1], el[nocc-1])
    path = [at([0.5*t, 0, 0.5*t]) for t in np.linspace(0, 1, 21)] + [at([0.5*t]*3) for t in np.linspace(0, 1, 11)]
    cbm = min(p[nocc] for p in path)
    print(f"  VB width {eg[0]-vbm:8.3f}   direct gap at Gamma {eg[nocc]-eg[nocc-1]:7.3f}   "
          f"X gap {ex[nocc]-vbm:7.3f}   L gap {el[nocc]-vbm:7.3f}   minimum gap {cbm-vbm:7.3f} eV")

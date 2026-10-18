# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Are the band energies the check files assert converged in the plane-wave cutoff?

For one material and a list of cutoffs: the valence band width, and the lowest empty
state at Gamma, L and X, all in eV from the valence-band maximum (from E_F for a metal).
Self-consistent at each cutoff, dense diagonalization.

    python3 cutoff_convergence.py Ge 6,10,14,18,24

Measured, eV (the row in use is marked; "was" is what the first version of the decks used):

           Ha   width    Gamma     L        X       X4v      L3v
    Si      6   11.835  +2.549  +1.577  +0.670  -2.911  -1.234    was
    Si     10   11.984  +2.537  +1.425  +0.614  -2.877  -1.210    <- in use
    Si     14   11.984  +2.540  +1.410  +0.609  -2.865  -1.201
    Si     18   11.978  +2.539  +1.412  +0.609  -2.860  -1.199
    Ge      6   12.646  +0.884  +0.372  +0.652                    was: WRONG ORDERING
    Ge     10   12.808  -0.030  +0.109  +0.660
    Ge     14   12.781  -0.015  +0.129  +0.683                    <- in use
    Ge     18   12.780  -0.014  +0.132  +0.686
    Ge     24   12.783  -0.023  +0.128  +0.685
    GaAs   10   12.730  +0.452  +0.924  +1.354  -2.652  -1.129    was
    GaAs   14   12.699  +0.454  +0.944  +1.382  -2.641  -1.121    <- in use
    GaAs   18   12.686  +0.466  +0.954  +1.393  -2.637  -1.117
    C      20   21.530  +5.617  +8.445  +4.538  -6.405  -2.809    was
    C      30   21.374  +5.579  +8.405  +4.675  -6.304  -2.777    <- in use
    C      40   21.339  +5.556  +8.409  +4.711  -6.293  -2.790
    Al      6   E_F 7.784   Gamma1 -11.125   X -3.004 -1.618      was
    Al     10   E_F 7.741          -11.149     -2.970 -1.643      <- in use
    Al     14   E_F 7.733          -11.143     -2.963 -1.646

At 6 Ha germanium looked like a semiconductor with its gap at L. Converged, the s-like
Gamma2' state lies just BELOW the Gamma25' triplet: LDA germanium has no gap at all. The
numpy reference and the Viv code agreed perfectly at 6 Ha, because they shared the cutoff
-- agreement between two implementations says nothing about a parameter they share.
"""
import sys
import numpy as np
from pwdft_ref import scf, bands, mp_grid, fft_dims
from materials import build
HA = 27.211386245988
name = sys.argv[1]
cr, _, nk, smear, nb, _ = build(name)
ks, w = mp_grid((nk, nk, nk))
nv = cr.nelec // 2
L, G, X = np.array([.5, .5, .5]), np.zeros(3), np.array([.5, 0, .5])
for ecut in [float(x) for x in sys.argv[2].split(',')]:
    dims = fft_dims(cr, ecut, list(ks) + [L, G, X])
    out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.6, anderson=3, dims=dims)
    eg, el, ex = (bands(cr, ecut, out['vtot_g'], dims, k, nb) * HA for k in (G, L, X))
    if smear > 0:
        mu = out['mu'] * HA
        print(f"{name:<5}{ecut:5.1f} Ha  grid {dims[0]:2d}  E_F {mu:8.4f} | Gamma {np.round(eg[:3] - mu, 3)} | L {np.round(el[:3] - mu, 3)} | X {np.round(ex[:3] - mu, 3)}", flush=True)
        continue
    # the valence-band maximum is the Gamma25' triplet: the three-fold level among bands 2..5 at Gamma
    four = eg[nv - 3:nv + 1]
    i = int(np.argmax(np.diff(four)))
    triplet, singlet = (four[1:], four[0]) if i == 0 else (four[:3], four[3])
    vbm = triplet.mean()
    print(f"{name:<5}{ecut:5.1f} Ha  grid {dims[0]:2d}  iters {out['iters']:2d} | width {vbm - eg[0]:7.3f} | Gamma {singlet - vbm:+.3f}"
          f" | L {el[nv] - vbm:+.3f} | X {ex[nv] - vbm:+.3f} | X4v {ex[nv - 1] - vbm:+.3f}  L3v {el[nv - 1] - vbm:+.3f}", flush=True)

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The numbers the silicon notebook post asserts, from the numpy reference.

    python3 dft/ref/notebook_si.py            (needs numpy and PySCF; about 2 s)

posts/2026-09-25-silicon-bands-from-the-crystal-up.viv runs a smaller silicon
calculation than dft/si.viv, so that it finishes within a post's time budget in
the browser: a lower cutoff, the smallest four-shift k-mesh, a coarser band
path and plain linear density mixing. This script runs pwdft_ref.py, the dense
numpy implementation, at exactly those settings and prints every number the
notebook's tests compare against. The 10 Ha values the notebook compares with
are REFPATH in dft/si_check.viv, written there by dft/ref/make_deck.py.

Run it single-threaded (OMP_NUM_THREADS=1): the matrices are small, and a
threaded BLAS spends its time on synchronization.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwdft_ref import scf, bands, mp_grid, pw_basis, fft_nice, lda_pade, vloc_g, proj_g  # noqa: E402
from materials import build  # noqa: E402

HA = 27.211386245988

# the notebook's settings; change them there and here together
ECUT = 5.0
NKMESH = 1
NSEG = 10
MIX = 0.6
TOL = 1.0e-8
NBANDS = 8

cr, _, _, _, _, _ = build('Si')
ks, w = mp_grid((NKMESH,) * 3)
L, G, X = np.array([0.5, 0.5, 0.5]), np.zeros(3), np.array([0.5, 0.0, 0.5])
path = [L + (G - L) * t / NSEG for t in range(NSEG)] + [G + (X - G) * t / NSEG for t in range(NSEG + 1)]
# the grid the notebook sizes: over the SCF k-points AND the path
mm = max(np.abs(pw_basis(cr, k, ECUT)[0]).max() for k in list(ks) + path)
ng = fft_nice(4 * mm + 2)
dims = (ng, ng, ng)

pp = cr.pps[0]
print(f"cell volume {cr.omega:.9f} bohr^3")
print("local pseudopotential at g = 0, 1, 2 (Ha):", ", ".join(f"{x:.12f}" for x in vloc_g(pp, [0.0, 1.0, 2.0], cr.omega)))
for l, i, r in ((0, 1, pp['shells'][0]['r']), (0, 2, pp['shells'][0]['r']), (1, 1, pp['shells'][1]['r'])):
    print(f"projector l={l} i={i} at k = 0.5, 1, 2:", ", ".join(f"{x:.12f}" for x in proj_g(l, i, r, np.array([0.5, 1.0, 2.0]), cr.omega)))
print(f"k-points {len(ks)}, weights {w}")
print(f"plane waves at Gamma {len(pw_basis(cr, [0, 0, 0], ECUT)[0])}, "
      f"most at any k-point {max(len(pw_basis(cr, k, ECUT)[0]) for k in list(ks) + path)}, grid {ng}")

# the starting potential: the uniform density, one pass
first = scf(cr, ECUT, ks, w, nbands=NBANDS, mix=MIX, tol=TOL, maxit=1, verbose=False, dims=dims)
e0 = bands(cr, ECUT, first['vtot_g'], dims, G, NBANDS)
print("Gamma, starting potential (Ha):", ", ".join(f"{x:.9f}" for x in e0[:6]))

out = scf(cr, ECUT, ks, w, nbands=NBANDS, mix=MIX, tol=TOL, maxit=80, verbose=False, dims=dims)
print(f"SCF: {out['iters']} iterations, E = {out['etot']:.10f} Ha (no Ewald)")

b = np.array([bands(cr, ECUT, out['vtot_g'], dims, k, NBANDS) for k in path]) * HA
nocc = int(cr.nelec // 2)
vbm = b[:, nocc - 1].max()
b -= vbm
kgap = int(np.argmin(b[:, nocc]))
for name, p in (('L', 0), ('Gamma', NSEG), ('X', 2 * NSEG)):
    print(f"{name:6s}", ", ".join(f"{x:.6f}" for x in b[p, :6]))
print(f"gap {b[kgap, nocc]:.6f} eV at path point {kgap} ({(kgap - NSEG) / NSEG:.2f} of Gamma-X)")

rho = np.array([0.001, 0.01, 0.03, 0.1])
exc, vxc = lda_pade(rho)
print("LDA at rho = 0.001, 0.01, 0.03, 0.1:")
print("  e_xc", ", ".join(f"{x:.12f}" for x in exc))
print("  v_xc", ", ".join(f"{x:.12f}" for x in vxc))

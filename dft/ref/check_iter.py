# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The iterative solver against the dense one: same physics, same answer."""
import sys, time
import numpy as np
import pwdft_ref as dense
import pwdft_iter as it
from materials import build
HA = 27.211386245988
for name in sys.argv[1:]:
    cr, ecut, nk, smear, nb, _ = build(name)
    ks, w = dense.mp_grid((nk, nk, nk))
    dims = dense.fft_dims(cr, ecut, ks)
    t0 = time.time(); d = dense.scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, mix=0.5, anderson=3); td = time.time() - t0
    t0 = time.time(); i = it.scf(cr, ecut, ks, w, nb, dims, smear=smear, verbose=False); ti = time.time() - t0
    de = abs(d['etot'] - i['etot'])
    dev = max(np.abs(np.array(d['eigs'])[:, :nb - 2] - np.array(i['eigs'])[:, :nb - 2]).max(), 0)
    print(f"{name:<5} dense: F = {d['etot']:.10f} ({d['iters']} its, {td:5.1f} s)   "
          f"iterative: F = {i['etot']:.10f} ({i['iters']} its, {ti:5.1f} s)")
    print(f"      |dF| = {de:.2e} Ha    max |d eig| over the lowest {nb-2} bands = {dev*HA:.2e} eV"
          + (f"    |d E_F| = {abs(d['mu']-i['mu'])*HA:.2e} eV" if d['mu'] is not None else ""))

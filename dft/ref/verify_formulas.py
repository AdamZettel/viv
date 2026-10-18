# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Check the formulas pwdft_ref.py uses against things it does not define.

A closed form copied from a paper can be wrong in a way that still runs. Each
one here is tested against its real-space DEFINITION by radial quadrature, and
the LDA against libxc -- so an error has to be made twice, independently, to
survive.
"""
import os
import numpy as np
from math import pi, sqrt
from scipy.integrate import quad
from scipy.special import spherical_jn
import pyscf
from pwdft_ref import load_gth, proj_g, proj_r, vloc_g, lda_pade

GTH = os.path.join(os.path.dirname(pyscf.__file__), 'pbc/gto/pseudo/gth-pade.dat')
fails = 0

def check(label, got, want, tol):
    global fails
    err = abs(got - want) / max(abs(want), 1e-300)
    ok = err < tol
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {label:<46} rel err {err:.2e}")

# --- projectors: p(K) = 4 pi / sqrt(omega) * int r^2 p(r) j_l(K r) dr --------
print("GTH projectors vs radial quadrature of their definition")
omega = 1.0
for l, imax in [(0, 3), (1, 3), (2, 2), (3, 1)]:
    for i in range(1, imax + 1):
        for r in (0.27, 0.48):
            for K in (0.7, 2.9, 6.1):
                num = 4 * pi * quad(lambda x: x * x * proj_r(l, i, r, x) * spherical_jn(l, K * x),
                                    0, 40 * r, limit=400)[0]
                check(f"l={l} i={i} r={r} K={K}", float(proj_g(l, i, r, np.array([K]), omega)[0]), num, 1e-8)

# --- the projectors are normalized: int r^2 p(r)^2 dr = 1 --------------------
print("projector normalization")
for l, imax in [(0, 3), (1, 3), (2, 2), (3, 1)]:
    for i in range(1, imax + 1):
        check(f"l={l} i={i}", quad(lambda x: x * x * proj_r(l, i, 0.4, x) ** 2, 0, 20)[0], 1.0, 1e-10)

# --- local potential: FT of  -Z erf(r/(sqrt2 rl))/r + gaussian * poly --------
print("GTH local potential vs radial quadrature")
for name in ("Si GTH-PADE-q4", "C GTH-PADE-q4", "Cu GTH-PADE-q11"):
    pp = load_gth(GTH, name)
    z, rl, c = pp['zion'], pp['rloc'], pp['c']
    for G in (0.6, 2.2, 5.0):
        # short-range part only is integrable; the erf Coulomb tail has the
        # analytic transform -4 pi Z exp(-G^2 rl^2 / 2) / G^2, checked separately
        def vsr(x):
            t = (x / rl) ** 2
            return np.exp(-0.5 * t) * (c[0] + c[1] * t + c[2] * t * t + c[3] * t ** 3)
        sr = 4 * pi * quad(lambda x: x * x * vsr(x) * spherical_jn(0, G * x), 0, 30 * rl, limit=300)[0]
        # the erf part, with a convergence factor: -Z erf(r/(sqrt2 rl))/r
        lr = -4 * pi * z * np.exp(-0.5 * (G * rl) ** 2) / G ** 2
        check(f"{name.split()[0]:<2} G={G}", float(vloc_g(pp, np.array([G]), 1.0)[0]), sr + lr, 1e-9)
    # The erf Coulomb transform itself. Integrating -Z erf(..)/r directly does not
    # converge (a 1/r tail times an oscillation), so split it:
    #     -Z erf(a r)/r  =  -Z/r  +  Z erfc(a r)/r ,   a = 1/(sqrt2 rl)
    # FT[-Z/r] = -4 pi Z / G^2 is textbook; the erfc piece is short-ranged and
    # integrates cleanly. Their sum must be -4 pi Z exp(-G^2 rl^2/2) / G^2.
    from scipy.special import erfc
    G = 1.7
    sr = 4 * pi * quad(lambda x: x * x * (z * erfc(x / (sqrt(2) * rl)) / x) * spherical_jn(0, G * x),
                       0, 40 * rl, limit=400)[0]
    check(f"{name.split()[0]:<2} erf Coulomb tail, G={G}", -4 * pi * z * np.exp(-0.5 * (G * rl) ** 2) / G ** 2,
          -4 * pi * z / G ** 2 + sr, 1e-9)

# --- G -> 0 limit of the local potential is continuous -----------------------
print("local potential: the G = 0 value is the limit of the G != 0 formula + 4 pi Z / G^2")
for name in ("Si GTH-PADE-q4", "Cu GTH-PADE-q11"):
    pp = load_gth(GTH, name)
    g = 1e-4
    lim = float(vloc_g(pp, np.array([g]), 1.0)[0]) + 4 * pi * pp['zion'] / g ** 2
    check(name.split()[0], float(vloc_g(pp, np.array([0.0]), 1.0)[0]), lim, 1e-6)

# --- LDA: Teter-93 Pade vs libxc ---------------------------------------------
print("Teter-93 Pade LDA vs libxc (through PySCF)")
from pyscf.dft import libxc
rho = np.array([1e-4, 1e-2, 0.03, 0.1, 0.5, 2.0])
exc_ref, vxc_ref = libxc.eval_xc('LDA_XC_TETER93', rho, spin=0, deriv=1)[:2]
eps, v = lda_pade(rho)
for k, n in enumerate(rho):
    check(f"eps_xc  n={n}", eps[k], exc_ref[k], 1e-10)
    check(f"v_xc    n={n}", v[k], vxc_ref[0][k], 1e-10)

print()
print("ALL FORMULAS VERIFIED" if fails == 0 else f"{fails} FORMULA CHECK(S) FAILED")
raise SystemExit(1 if fails else 0)

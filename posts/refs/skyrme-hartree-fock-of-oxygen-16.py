# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-skyrme-hartree-fock-of-oxygen-16.viv.

1. A numpy twin of the post's solver: the same functional and the same
   discretization, but each (l, j) block of the single-particle Hamiltonian
   is diagonalized densely by numpy.linalg.eigh (the post uses inverse
   iteration on the tridiagonal matrix), and the derivatives are
   numpy.gradient(edge_order=2). Run at h = 0.1 fm and h = 0.05 fm (box 15
   fm), with the Richardson extrapolation (4 x_h/2 - x_h) / 3.

2. The published run. skyrme_rpa (G. Colo, L. Cao, N. Van Giai and
   L. Capelli, Comput. Phys. Commun. 184, 142 (2013); CPC Program Library
   AENF_v1_0, distributed on Mendeley Data, doi:10.17632/9hdvznfzvs.1) ships
   with a test run, sample_skyrme_rpa.out: Hartree-Fock of 16O with SLy5,
   150 points of 0.1 fm. Its numbers are copied below from that file. The
   program was compiled (gfortran 15, -std=legacy) and rerun on
   2026-09-25; it reproduces the sample output line for line, and rerun with
   300 points of 0.05 fm it gives ETot/A = -8.0259916 MeV and RC = 2.7992 fm
   with the same single-particle energies to 1 keV.

   SLy5 as skyrme_rpa's skyrme.in lists it ("NPA 643 (1998) 441", Chabanat
   et al.): t0 = -2484.88, t1 = 483.13, t2 = -549.40, t3 = 13763 (MeV fm^n),
   x0 = 0.778, x1 = -0.328, x2 = -1, x3 = 1.267, W0 = W0' = 126 MeV fm^5,
   alpha = 1/6 (0.1666666 there), with the J^2 terms included.

   Conventions of skyrme_rpa, used by the post and by the twin:
   hbar c = 197.327053 MeV fm, m = (938.27231 + 939.56563)/2 MeV, the
   centre-of-mass correction as hbar^2/2m -> hbar^2/2m (1 - 1/A) in the
   kinetic energy and the single-particle equation, e^2 = 1.43986 MeV fm,
   Coulomb exchange in the Slater approximation, and the charge density as
   the proton density folded with a Gaussian of range 0.65 fm, which adds
   (3/2)(0.65 fm)^2 to the mean square radius.

    python3 posts/refs/skyrme-hartree-fock-of-oxygen-16.py > posts/refs/skyrme-hartree-fock-of-oxygen-16.out
"""
import numpy as np

# skyrme_rpa, sample_skyrme_rpa.out (16O, SLy5, 150 x 0.1 fm)
PUB = dict(E_A=-8.0261107, rp=2.6836, rn=2.6585, rc=2.7991,
           p=(-32.346, -17.138, -11.083), n=(-36.138, -20.613, -14.429))

HBARC = 197.327053
MN = (938.27231 + 939.56563) / 2
E2 = 1.43986
A, Z, NN = 16, 8, 8
t0, t1, t2, t3 = -2484.88, 483.13, -549.40, 13763.0
x0, x1, x2, x3 = 0.778, -0.328, -1.0, 1.267
W0, ALPHA = 126.0, 1.0 / 6.0
HB = HBARC**2 / (2 * MN) * (A - 1) / A
ORBS = [(0, 0.5, 2), (1, 1.5, 4), (1, 0.5, 2)]   # (l, j, 2j+1): 1s1/2, 1p3/2, 1p1/2
MIX, TOL = 0.4, 1e-9


def run(h, nr):
    r = (np.arange(nr) + 1) * h

    def d(f):
        return np.gradient(f, h, edge_order=2)

    def lap(f):
        g = d(f)
        return d(g) + 2 / r * g

    def integ(f):
        return 4 * np.pi * np.trapz(f * r**2, r)

    def coul(rp):
        cin = np.cumsum(rp * r**2)
        out = np.cumsum((rp * r)[::-1])[::-1] - rp * r
        return E2 * 4 * np.pi * h * (cin / r + out)

    def solve(l, sdl, B, U, W, V):
        Bp = np.empty(nr + 2)
        Bp[1:-1] = B
        Bp[0] = B[0]
        Bp[-1] = HB
        Bh = 0.5 * (Bp[:-1] + Bp[1:])
        dg = (Bh[:-1] + Bh[1:]) / h**2 + B * l * (l + 1) / r**2 + d(B) / r + U + V + sdl * W / r
        off = -Bh[1:-1] / h**2
        e, v = np.linalg.eigh(np.diag(dg) + np.diag(off, 1) + np.diag(off, -1))
        return e[0], v[:, 0] / np.sqrt(h) / r

    prof = 1 / (1 + np.exp((r - 1.2 * A**(1 / 3)) / 0.6))
    rho_q = [prof * NN / integ(prof), prof * Z / integ(prof)]
    tau_q = [np.zeros(nr), np.zeros(nr)]
    J_q = [np.zeros(nr), np.zeros(nr)]
    e_old = 0.0
    for it in range(1, 500):
        rho, tau, J = rho_q[0] + rho_q[1], tau_q[0] + tau_q[1], J_q[0] + J_q[1]
        lev = {}
        new = []
        for q in (0, 1):
            rq, tq, Jq = rho_q[q], tau_q[q], J_q[q]
            B = HB + 0.125 * (t1 * (2 + x1) + t2 * (2 + x2)) * rho + 0.125 * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * rq
            U = t0 * ((1 + x0 / 2) * rho - (x0 + 0.5) * rq)
            U += 0.25 * t1 * ((1 + x1 / 2) * (tau - 1.5 * lap(rho)) - (x1 + 0.5) * (tq - 1.5 * lap(rq)))
            U += 0.25 * t2 * ((1 + x2 / 2) * (tau + 0.5 * lap(rho)) + (x2 + 0.5) * (tq + 0.5 * lap(rq)))
            U += (t3 / 12) * ((1 + x3 / 2) * (2 + ALPHA) * rho**(ALPHA + 1)
                              - (x3 + 0.5) * (2 * rho**ALPHA * rq + ALPHA * rho**(ALPHA - 1) * (rho_q[0]**2 + rho_q[1]**2)))
            U += -W0 * ((J + Jq) / r + 0.5 * d(J + Jq))
            W = 0.5 * W0 * (d(rho) + d(rq)) + 0.125 * (t1 - t2) * Jq - 0.125 * (t1 * x1 + t2 * x2) * J
            V = 0 * r if q == 0 else coul(rq) - E2 * (3 / np.pi)**(1 / 3) * np.cbrt(rq)
            nr_, nt, nJ = np.zeros(nr), np.zeros(nr), np.zeros(nr)
            for (l, j, deg) in ORBS:
                sdl = j * (j + 1) - l * (l + 1) - 0.75
                e, R = solve(l, sdl, B, U, W, V)
                lev[(q, l, j)] = e
                w = deg / (4 * np.pi)
                dR = d(R)
                nr_ += w * R**2
                nt += w * (dR**2 + l * (l + 1) / r**2 * R**2)
                nJ += w * sdl / r * R**2
            new.append((nr_, nt, nJ))
        for q in (0, 1):
            rho_q[q] = (1 - MIX) * rho_q[q] + MIX * new[q][0]
            tau_q[q] = (1 - MIX) * tau_q[q] + MIX * new[q][1]
            J_q[q] = (1 - MIX) * J_q[q] + MIX * new[q][2]
        rn, rp = rho_q
        rho, tau, J = rn + rp, tau_q[0] + tau_q[1], J_q[0] + J_q[1]
        dr, drn, drp = d(rho), d(rn), d(rp)
        tn, tp = tau_q
        Jn, Jp = J_q
        H = HB * tau + 0.5 * t0 * ((1 + x0 / 2) * rho**2 - (x0 + 0.5) * (rn**2 + rp**2))
        H += 0.25 * t1 * ((1 + x1 / 2) * (rho * tau + 0.75 * dr**2) - (x1 + 0.5) * (rn * tn + 0.75 * drn**2 + rp * tp + 0.75 * drp**2))
        H += 0.25 * t2 * ((1 + x2 / 2) * (rho * tau - 0.25 * dr**2) + (x2 + 0.5) * (rn * tn - 0.25 * drn**2 + rp * tp - 0.25 * drp**2))
        H += (t3 / 12) * rho**ALPHA * ((1 + x3 / 2) * rho**2 - (x3 + 0.5) * (rn**2 + rp**2))
        H += (1 / 16) * (t1 - t2) * (Jn**2 + Jp**2) - (1 / 16) * (t1 * x1 + t2 * x2) * J**2
        H += 0.5 * W0 * (J * dr + Jn * drn + Jp * drp)
        E = integ(H) + 0.5 * integ(coul(rp) * rp) - 0.75 * E2 * (3 / np.pi)**(1 / 3) * integ(np.cbrt(rp) * rp)
        if abs(E - e_old) < TOL and it > 3:
            break
        e_old = E
    r2p = integ(rp * r**2) / Z
    out = dict(it=it, E=E, E_A=E / A, rp=np.sqrt(r2p), rn=np.sqrt(integ(rn * r**2) / NN),
               rc=np.sqrt(r2p + 1.5 * 0.65**2),
               n=tuple(lev[(0, l, j)] for (l, j, _) in ORBS), p=tuple(lev[(1, l, j)] for (l, j, _) in ORBS))
    return out


print(f"numpy {np.__version__}")
res = {}
for h, nr in [(0.1, 150), (0.05, 300)]:
    o = run(h, nr)
    res[h] = o
    print(f"h = {h}: {o['it']} iterations  E = {o['E']:.8f} MeV  E/A = {o['E_A']:.10f}  r_p = {o['rp']:.10f}  "
          f"r_n = {o['rn']:.10f}  r_ch = {o['rc']:.10f}")
    print("   neutron 1s1/2 1p3/2 1p1/2: " + "  ".join(f"{e:.8f}" for e in o['n']))
    print("   proton  1s1/2 1p3/2 1p1/2: " + "  ".join(f"{e:.8f}" for e in o['p']))
a, b = res[0.1], res[0.05]


def rich(key):
    return (4 * b[key] - a[key]) / 3


print("Richardson (4 x_0.05 - x_0.1) / 3:")
for key in ("E_A", "rp", "rn", "rc"):
    print(f"   {key}: {rich(key):.8f}   skyrme_rpa sample: {PUB[key]}   difference {rich(key) - PUB[key]:+.2e}")
for q, name in ((0, "n"), (1, "p")):
    ext = [(4 * y - x) / 3 for x, y in zip(a[name], b[name])]
    print(f"   {name} levels: " + "  ".join(f"{e:.4f}" for e in ext) + "   skyrme_rpa: " + "  ".join(str(e) for e in PUB[name]))

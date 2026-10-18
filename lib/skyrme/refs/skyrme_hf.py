# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# An independent NumPy Skyrme-HF solver, kept outside this repository and
# copied unchanged below this header: the numpy reference of lib/skyrme/spherical.viv. It diagonalizes each
# (l, j) block densely (numpy.linalg.eigh) and fills shells in a
# prescribed order; lib/skyrme/refs/compare.py sets its constants to
# skyrme_rpa's and compares.
"""
Spherical Skyrme-Hartree-Fock solver -- Vautherin & Brink (Phys. Rev. C 5, 626, 1972).

Method: radial box + finite-difference matrix diagonalization (effective-mass
kinetic operator kept in self-adjoint divergence form and Hermitized on a
uniform mesh). Self-consistent loop with linear density mixing.

Functional conventions follow Xayavong & Smirnova, arXiv:2111.13531 App. A
(which derives them from Vautherin-Brink), EXCEPT:
  - the genuine 3-body term is used directly:  H3 = (1/4) t3 rho_n rho_p rho,
    equivalently the density-dependent form with x3=1, gamma=1. Its s.p.
    potential is derived by hand (their printed A10 has an x3-1/2 sign typo).

All energies in MeV, lengths in fm.
"""
from __future__ import annotations
import numpy as np
if not hasattr(np, "trapezoid"):   # numpy < 2 compatibility
    np.trapezoid = np.trapz
from dataclasses import dataclass, field

# ---- physical constants -----------------------------------------------------
HBAR2_2M = 20.7355     # hbar^2 / 2m  [MeV fm^2], m = (m_p+m_n)/2 average nucleon
E2       = 1.439964    # e^2  [MeV fm]  (elementary charge squared, Gaussian)


# ---- Skyrme force -----------------------------------------------------------
@dataclass
class Skyrme:
    name: str
    t0: float; t1: float; t2: float; t3: float
    x0: float; W0: float
    x1: float = 0.0; x2: float = 0.0; x3: float = 1.0   # VB: only x0, x1=x2=0
    gamma: float = 1.0                                   # 3-body -> alpha=1
    three_body: bool = True   # use genuine 3-body H3 = 1/4 t3 rho_n rho_p rho
    j2_terms: bool = True     # include (t1,t2) J^2 tensor terms in E and Wso
                              # (VB/SIII: True;  SLy4/SkM*: fit without them)


# Original Vautherin-Brink forces (1972). VALUES TO VERIFY against Table.
SI  = Skyrme("SI",  t0=-1057.3, t1=235.9, t2=-100.0, t3=14463.5, x0=0.56, W0=120.0)
SII = Skyrme("SII", t0=-1169.9, t1=585.6, t2= -27.1, t3= 9331.1, x0=0.34, W0=105.0)


# ---- nucleus / orbital bookkeeping ------------------------------------------
@dataclass
class Orbital:
    n: int      # radial node index (1 = lowest, i.e. 1s,1p,...)
    l: int
    j2: int     # 2*j  (odd integer)
    @property
    def deg(self) -> int:        # (2j+1) full occupancy
        return self.j2 + 1
    @property
    def sig_dot_l(self) -> float:  # <sigma . l> = j(j+1)-l(l+1)-3/4  = l or -(l+1)
        j = self.j2 / 2.0
        return j*(j+1) - self.l*(self.l+1) - 0.75
    def label(self) -> str:
        spec = "spdfghijklm"
        return f"{self.n}{spec[self.l]}{self.j2}/2"


@dataclass
class Nucleus:
    name: str
    Z: int
    N: int
    protons: list[Orbital]
    neutrons: list[Orbital]
    @property
    def A(self) -> int:
        return self.Z + self.N


def _orbs(spec_list):
    """Build orbitals from (n, l, j2) tuples."""
    return [Orbital(n, l, j2) for (n, l, j2) in spec_list]

# spectroscopic filling sequences (n, l, 2j), in shell-model order
_FILL = {
    "s16":  [(1,0,1),(1,1,3),(1,1,1)],
    "s20":  [(1,0,1),(1,1,3),(1,1,1),(1,2,5),(2,0,1),(1,2,3)],
    "s40":  [(1,0,1),(1,1,3),(1,1,1),(1,2,5),(2,0,1),(1,2,3),
             (1,3,7),(2,1,3),(1,3,5),(2,1,1)],
    "s50":  [(1,0,1),(1,1,3),(1,1,1),(1,2,5),(2,0,1),(1,2,3),
             (1,3,7),(2,1,3),(1,3,5),(2,1,1),(1,4,9)],
    "s82":  [(1,0,1),(1,1,3),(1,1,1),(1,2,5),(2,0,1),(1,2,3),
             (1,3,7),(2,1,3),(1,3,5),(2,1,1),(1,4,9),
             (1,4,7),(2,2,5),(2,2,3),(3,0,1),(1,5,11)],
    "s126": [(1,0,1),(1,1,3),(1,1,1),(1,2,5),(2,0,1),(1,2,3),
             (1,3,7),(2,1,3),(1,3,5),(2,1,1),(1,4,9),
             (1,4,7),(2,2,5),(2,2,3),(3,0,1),(1,5,11),
             (1,5,9),(2,3,7),(2,3,5),(3,1,3),(3,1,1),(1,6,13)],
}

def _check(tag, orbs, want):
    got = sum(o.deg for o in orbs)
    assert got == want, f"{tag}: filled {got}, expected {want}"

def closed_shell_16O() -> Nucleus:
    p = _orbs(_FILL["s16"]); n = _orbs(_FILL["s16"])
    _check("16O p", p, 8); _check("16O n", n, 8)
    return Nucleus("16O", 8, 8, p, n)

def closed_shell_40Ca() -> Nucleus:
    p = _orbs(_FILL["s20"]); n = _orbs(_FILL["s20"])
    _check("40Ca p", p, 20); _check("40Ca n", n, 20)
    return Nucleus("40Ca", 20, 20, p, n)

def closed_shell_48Ca() -> Nucleus:
    p = _orbs(_FILL["s20"]); n = _orbs(_FILL["s40"][:7])  # N=28: through 1f7/2
    _check("48Ca p", p, 20); _check("48Ca n", n, 28)
    return Nucleus("48Ca", 20, 28, p, n)

def closed_shell_90Zr() -> Nucleus:
    p = _orbs(_FILL["s40"]); n = _orbs(_FILL["s50"])
    _check("90Zr p", p, 40); _check("90Zr n", n, 50)
    return Nucleus("90Zr", 40, 50, p, n)

def closed_shell_208Pb() -> Nucleus:
    p = _orbs(_FILL["s82"]); n = _orbs(_FILL["s126"])
    _check("208Pb p", p, 82); _check("208Pb n", n, 126)
    return Nucleus("208Pb", 82, 126, p, n)


# ---- radial grid ------------------------------------------------------------
class Grid:
    def __init__(self, rmax=20.0, h=0.1):
        self.h = h
        self.N = int(round(rmax / h))
        self.r = (np.arange(self.N) + 1) * h   # r_1..r_N, excludes origin
    def integrate(self, f):
        """int f(r) 4 pi r^2 dr  via trapezoid on the mesh (u(0)=0 boundary)."""
        return 4.0 * np.pi * np.trapezoid(f * self.r**2, self.r)


# ---- densities from occupied orbitals --------------------------------------
def build_densities(grid, states_q):
    """states_q: list of (Orbital, R(r) array) for one isospin.
    Returns rho, tau, J (spin-orbit density), all on grid.r."""
    r = grid.r
    rho = np.zeros_like(r); tau = np.zeros_like(r); J = np.zeros_like(r)
    for orb, R in states_q:
        w = orb.deg / (4.0 * np.pi)
        dR = np.gradient(R, grid.h, edge_order=2)
        rho += w * R**2
        tau += w * (dR**2 + orb.l*(orb.l+1)/r**2 * R**2)
        J   += w * orb.sig_dot_l / r * R**2
    return rho, tau, J


def coulomb_direct(rho_p, r, h):
    """Direct Coulomb potential of a spherical proton density:
        V(r) = e^2 [ (1/r) int_0^r rho_p 4pi r'^2 dr'  +  int_r^inf rho_p 4pi r' dr' ].
    Discretized so the inner sum uses shells j<=i and the outer uses j>i
    (clean partition -- no double counting of the r'=r shell)."""
    cum_in = np.cumsum(rho_p * r**2)                 # sum_{j<=i} rho r^2
    rev_ge = np.cumsum((rho_p * r)[::-1])[::-1]      # sum_{j>=i} rho r
    outer  = rev_ge - rho_p * r                      # sum_{j>i}  rho r
    return E2 * 4*np.pi*h * (cum_in / r + outer)


def deriv(f, h):
    return np.gradient(f, h, edge_order=2)

def laplacian_radial(f, r, h):
    """Delta f = f'' + (2/r) f'  for a spherically symmetric field."""
    df  = np.gradient(f, h, edge_order=2)
    d2f = np.gradient(df, h, edge_order=2)
    return d2f + 2.0/r * df


# ---- mean fields ------------------------------------------------------------
def mean_fields(force, grid, dens_n, dens_p):
    """Return per-isospin dicts of B (=hbar^2/2m*), U (central), Wso, and Vcoul.
    dens_* = (rho, tau, J)."""
    f = force; r = grid.r; h = grid.h
    rho_n, tau_n, J_n = dens_n
    rho_p, tau_p, J_p = dens_p
    rho = rho_n + rho_p; tau = tau_n + tau_p; J = J_n + J_p

    lap_rho   = laplacian_radial(rho,   r, h)
    lap_rho_n = laplacian_radial(rho_n, r, h)
    lap_rho_p = laplacian_radial(rho_p, r, h)
    drho   = deriv(rho,   h)
    drho_n = deriv(rho_n, h)
    drho_p = deriv(rho_p, h)

    def B_of(rho_q):
        # hbar^2/2m*_q = hbar^2/2m + 1/8[t1(2+x1)+t2(2+x2)]rho
        #                          + 1/8[t2(1+2x2) - t1(1+2x1)]rho_q
        # (like-particle term: t1 enters with a MINUS -- from -(x1+1/2) in H_Sk)
        return (HBAR2_2M
                + 0.125*(f.t1*(2+f.x1) + f.t2*(2+f.x2)) * rho
                + 0.125*(f.t2*(1+2*f.x2) - f.t1*(1+2*f.x1)) * rho_q)

    def U_central(rho_q, tau_q, lap_q):
        # t0
        U = f.t0*((1+f.x0/2)*rho - (f.x0+0.5)*rho_q)
        # t1
        U += 0.25*f.t1*((1+f.x1/2)*(tau - 1.5*lap_rho)
                        - (f.x1+0.5)*(tau_q - 1.5*lap_q))
        # t2
        U += 0.25*f.t2*((1+f.x2/2)*(tau + 0.5*lap_rho)
                        + (f.x2+0.5)*(tau_q + 0.5*lap_q))
        return U

    # three-body (genuine): U3,q = 1/4 t3 rho_{q'} (2 rho_q + rho_{q'})
    if f.three_body:
        U3_n = 0.25*f.t3 * rho_p * (2*rho_n + rho_p)
        U3_p = 0.25*f.t3 * rho_n * (2*rho_p + rho_n)
    else:  # density-dependent generalized form
        g = f.gamma
        def U3(rho_q):
            return (f.t3/12.0)*((1+f.x3/2)*(2+g)*rho**(g+1)
                    - (f.x3+0.5)*(2*rho**g*rho_q + g*rho**(g-1)*(rho_n**2+rho_p**2)))
        U3_n, U3_p = U3(rho_n), U3(rho_p)

    # spin-orbit central contribution: -(W0/2) div(J+Jq) = -W0[ (J+Jq)/r + 1/2 d/dr(J+Jq) ]
    def U_so(J_q):
        s = J + J_q
        return -f.W0*(s/r + 0.5*deriv(s, h))

    U_n = U_central(rho_n, tau_n, lap_rho_n) + U3_n + U_so(J_n)
    U_p = U_central(rho_p, tau_p, lap_rho_p) + U3_p + U_so(J_p)

    # spin-orbit form factor W_q (multiplies <sigma.l>/r in the s.p. eq)
    def Wso(rho_q, J_q):
        w = 0.5*f.W0*(drho + deriv(rho_q, h))
        if f.j2_terms:   # (t1,t2) J^2 contribution; x1=x2=0 -> 1/8(t1-t2)Jq
            w += 0.125*(f.t1 - f.t2)*J_q - 0.125*(f.t1*f.x1 + f.t2*f.x2)*J
        return w
    W_n = Wso(rho_n, J_n)
    W_p = Wso(rho_p, J_p)

    # Coulomb (direct, spherical) + Slater exchange, protons only
    Vc_dir = coulomb_direct(rho_p, r, h)
    Vc_ex = -E2*(3.0/np.pi)**(1/3) * np.cbrt(rho_p)
    Vcoul = Vc_dir + Vc_ex

    return dict(B_n=B_of(rho_n), B_p=B_of(rho_p),
                U_n=U_n, U_p=U_p, W_n=W_n, W_p=W_p, Vcoul=Vcoul)


# ---- single-particle solver for one (l, j, isospin) block -------------------
def solve_block(grid, l, sig_dot_l, B, U, W, Vcoul_or_zero, nstates):
    """Diagonalize the radial HF Hamiltonian; return (eps[:nstates], R[:,:nstates])."""
    r = grid.r; h = grid.h; Nn = grid.N
    # half-point effective mass (pad: B(0)~B[0], B(R_box)~HBAR2_2M vacuum)
    Bpad = np.empty(Nn+2)
    Bpad[1:-1] = B; Bpad[0] = B[0]; Bpad[-1] = HBAR2_2M
    Bhalf = 0.5*(Bpad[:-1] + Bpad[1:])      # length N+1: B_{i-1/2}, i=1..N and B_{N+1/2}
    Bm = Bhalf[:-1]   # B_{i-1/2}
    Bp = Bhalf[1:]    # B_{i+1/2}

    diag = (Bm + Bp)/h**2 \
        + B*l*(l+1)/r**2 + deriv(B, h)/r \
        + U + Vcoul_or_zero \
        + sig_dot_l * W / r
    off = -Bp[:-1]/h**2   # coupling i,i+1

    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    eps, vec = np.linalg.eigh(H)
    # normalize so that int R^2 r^2 dr = 1  ->  int u^2 dr = 1  ->  sum u^2 h = 1
    u = vec / np.sqrt(h)
    R = u / r[:, None]
    return eps[:nstates], R[:, :nstates]


def solve_isospin(grid, orbitals, B, U, W, Vcoul):
    """Solve all needed (l,j) blocks; return list of (Orbital, R) for occupied states."""
    # group by (l, j2); the n index selects which eigenstate
    need = {}
    for o in orbitals:
        need.setdefault((o.l, o.j2), 0)
        need[(o.l, o.j2)] = max(need[(o.l, o.j2)], o.n)
    solutions = {}
    for (l, j2), nmax in need.items():
        sdl = (j2/2.0)*(j2/2.0+1) - l*(l+1) - 0.75
        eps, R = solve_block(grid, l, sdl, B, U, W, Vcoul, nmax)
        solutions[(l, j2)] = (eps, R)
    out = []
    levels = []
    for o in orbitals:
        eps, R = solutions[(o.l, o.j2)]
        out.append((o, R[:, o.n-1]))
        levels.append((o, eps[o.n-1]))
    return out, levels


# ---- total energy from the functional ---------------------------------------
def total_energy(force, grid, dens_n, dens_p, com_correction=True, A=None):
    f = force; r = grid.r; h = grid.h
    rho_n, tau_n, J_n = dens_n
    rho_p, tau_p, J_p = dens_p
    rho = rho_n+rho_p; tau = tau_n+tau_p; J = J_n+J_p
    drho = deriv(rho, h); drho_n = deriv(rho_n, h); drho_p = deriv(rho_p, h)

    kin_fac = 1.0
    if com_correction and A:
        kin_fac = (A-1)/A
    H = HBAR2_2M*kin_fac*tau

    H += 0.5*f.t0*((1+f.x0/2)*rho**2 - (f.x0+0.5)*(rho_n**2+rho_p**2))

    H += 0.25*f.t1*((1+f.x1/2)*(rho*tau + 0.75*drho**2)
                    - (f.x1+0.5)*(rho_n*tau_n + 0.75*drho_n**2
                                  + rho_p*tau_p + 0.75*drho_p**2))
    H += 0.25*f.t2*((1+f.x2/2)*(rho*tau - 0.25*drho**2)
                    + (f.x2+0.5)*(rho_n*tau_n - 0.25*drho_n**2
                                  + rho_p*tau_p - 0.25*drho_p**2))

    if f.three_body:
        H += 0.25*f.t3 * rho_n * rho_p * rho
    else:
        H += (f.t3/12.0)*rho**f.gamma*((1+f.x3/2)*rho**2
                                       - (f.x3+0.5)*(rho_n**2+rho_p**2))

    # J^2 terms (x1=x2=0): + 1/16 (t1-t2) sum Jq^2
    if f.j2_terms:
        H += (1.0/16.0)*(f.t1 - f.t2)*(J_n**2 + J_p**2)
        H += -(1.0/16.0)*(f.t1*f.x1 + f.t2*f.x2)*J**2
    # spin-orbit W0 term:  (W0/2)(J.drho + sum Jq.drho_q)
    H += 0.5*f.W0*(J*drho + J_n*drho_n + J_p*drho_p)

    E_nuc = 4*np.pi*np.trapezoid(H*r**2, r)

    # Coulomb energy
    Vc_dir = coulomb_direct(rho_p, r, h)
    E_cdir = 0.5*4*np.pi*np.trapezoid(Vc_dir*rho_p*r**2, r)
    E_cex  = -0.75*E2*(3/np.pi)**(1/3)*4*np.pi*np.trapezoid(np.cbrt(rho_p)*rho_p*r**2, r)

    return E_nuc + E_cdir + E_cex, dict(E_nuc=E_nuc, E_cdir=E_cdir, E_cex=E_cex)


# ---- self-consistent loop ---------------------------------------------------
def run_hf(force, nucleus, grid=None, maxiter=200, mix=0.4, tol=1e-6,
           com_correction=False, verbose=True):
    if grid is None:
        grid = Grid()
    r = grid.r
    # initial guess: Fermi/Woods-Saxon-like densities
    R0 = 1.2 * nucleus.A**(1/3)
    a  = 0.6
    prof = 1.0/(1.0+np.exp((r-R0)/a))
    rho_p = prof * nucleus.Z / (4*np.pi*np.trapezoid(prof*r**2, r))
    rho_n = prof * nucleus.N / (4*np.pi*np.trapezoid(prof*r**2, r))
    tau_n = np.zeros_like(r); tau_p = np.zeros_like(r)
    J_n = np.zeros_like(r);  J_p = np.zeros_like(r)

    E_old = 0.0
    for it in range(1, maxiter+1):
        dens_n = (rho_n, tau_n, J_n)
        dens_p = (rho_p, tau_p, J_p)
        fld = mean_fields(force, grid, dens_n, dens_p)

        occ_n, lev_n = solve_isospin(grid, nucleus.neutrons,
                                     fld['B_n'], fld['U_n'], fld['W_n'],
                                     np.zeros_like(r))
        occ_p, lev_p = solve_isospin(grid, nucleus.protons,
                                     fld['B_p'], fld['U_p'], fld['W_p'],
                                     fld['Vcoul'])

        nrho_n, ntau_n, nJ_n = build_densities(grid, occ_n)
        nrho_p, ntau_p, nJ_p = build_densities(grid, occ_p)

        # linear mixing
        rho_n = (1-mix)*rho_n + mix*nrho_n
        rho_p = (1-mix)*rho_p + mix*nrho_p
        tau_n = (1-mix)*tau_n + mix*ntau_n
        tau_p = (1-mix)*tau_p + mix*ntau_p
        J_n   = (1-mix)*J_n   + mix*nJ_n
        J_p   = (1-mix)*J_p   + mix*nJ_p

        E, parts = total_energy(force, grid, (rho_n,tau_n,J_n), (rho_p,tau_p,J_p),
                                com_correction=com_correction, A=nucleus.A)
        dE = E - E_old; E_old = E
        if verbose and (it <= 3 or it % 10 == 0 or abs(dE) < tol):
            print(f"  it{it:3d}  E={E:12.4f}  BE/A={-E/nucleus.A:8.4f}  dE={dE:+.2e}")
        if abs(dE) < tol and it > 3:
            break

    # observables
    r2p = 4*np.pi*np.trapezoid(rho_p*r**4, r) / nucleus.Z
    r2n = 4*np.pi*np.trapezoid(rho_n*r**4, r) / nucleus.N
    rms_p = np.sqrt(r2p); rms_n = np.sqrt(r2n)
    rms_ch = np.sqrt(r2p + 0.64)   # fold proton finite size (0.8 fm)^2

    return dict(E=E, BE_per_A=-E/nucleus.A, parts=parts,
                rms_p=rms_p, rms_n=rms_n, rms_ch=rms_ch,
                levels_p=lev_p, levels_n=lev_n,
                dens=(rho_n, rho_p), grid=grid, iters=it,
                fields=fld,
                dens_n=(rho_n, tau_n, J_n), dens_p=(rho_p, tau_p, J_p))


def sp_levels(res, orbitals_p, orbitals_n):
    """Energies (MeV) of any requested orbitals in the *converged* HF field --
    including unoccupied 'particle' states above the Fermi surface."""
    g = res["grid"]; fld = res["fields"]; z = np.zeros_like(g.r)
    _, lev_p = solve_isospin(g, orbitals_p, fld["B_p"], fld["U_p"], fld["W_p"], fld["Vcoul"])
    _, lev_n = solve_isospin(g, orbitals_n, fld["B_n"], fld["U_n"], fld["W_n"], z)
    return lev_p, lev_n


# SIII (Beiner et al. 1975) -- same alpha=1, x3=1 structure as SI/SII
SIII = Skyrme("SIII", t0=-1128.75, t1=395.0, t2=-95.0, t3=14000.0, x0=0.45, W0=120.0)

# Modern forces with density-dependent t3 (gamma != 1) and x1,x2 != 0.
# These exercise the generalized 3-body path and are fit WITHOUT J^2 terms.
SLy4 = Skyrme("SLy4", t0=-2488.913, t1=486.818, t2=-546.395, t3=13777.0,
              x0=0.834, x1=-0.3438, x2=-1.0, x3=1.354, W0=123.0,
              gamma=1/6, three_body=False, j2_terms=False)
SkMs = Skyrme("SkM*", t0=-2645.0, t1=410.0, t2=-135.0, t3=15595.0,
              x0=0.09, x1=0.0, x2=0.0, x3=0.0, W0=130.0,
              gamma=1/6, three_body=False, j2_terms=False)


def mstar_over_m(force, rho0=0.16):
    """Symmetric-matter effective mass ratio, for validation against literature."""
    B = (HBAR2_2M
         + 0.125*(force.t1*(2+force.x1)+force.t2*(2+force.x2))*rho0
         + 0.125*(force.t2*(1+2*force.x2)-force.t1*(1+2*force.x1))*(rho0/2))
    return HBAR2_2M / B


if __name__ == "__main__":
    # Reproduction of Vautherin & Brink across the doubly-magic chain.
    # (A-1)/A c.m. correction ON -- matches VB convention.
    EXPT = {"16O": (7.976, 2.699), "40Ca": (8.551, 3.478),
            "48Ca": (8.666, 3.477), "90Zr": (8.710, 4.269),
            "208Pb": (7.867, 5.501)}
    NUCLEI = [closed_shell_16O(), closed_shell_40Ca(), closed_shell_48Ca(),
              closed_shell_90Zr(), closed_shell_208Pb()]
    for force in (SIII, SI, SII):
        print(f"\n=== {force.name}  (m*/m = {mstar_over_m(force):.3f}) ===")
        print(f"{'nucleus':8s} {'BE/A':>7s} {'exp':>6s} {'d':>6s}   "
              f"{'r_ch':>6s} {'exp':>6s} {'d':>6s}")
        for nuc in NUCLEI:
            grid = Grid(rmax=22.0, h=0.1)
            res = run_hf(force, nuc, grid=grid, mix=0.25, maxiter=400,
                         com_correction=True, verbose=False)
            bea, rch = EXPT[nuc.name]
            print(f"{nuc.name:8s} {res['BE_per_A']:7.3f} {bea:6.3f} "
                  f"{res['BE_per_A']-bea:+6.3f}   {res['rms_ch']:6.3f} "
                  f"{rch:6.3f} {res['rms_ch']-rch:+6.3f}")

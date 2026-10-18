# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Plane-wave pseudopotential DFT -- the numpy reference for dft/pwdft.viv.

This is the ORACLE, not the product. It exists so that every number the Viv
code prints has something independent to agree with: same equations, different
language, different linear algebra (numpy.linalg.eigh instead of eig_herm,
numpy.fft instead of Viv's fft3). The two must match to ~1e-8.

It also checks ITSELF against things it does not define:
  * the GTH projector and local-potential closed forms, against brute-force
    radial quadrature of their real-space definitions
  * the Teter-93 Pade LDA, against libxc (through PySCF)
so a transcription error in a formula cannot hide by being made twice.

Physics: Kohn-Sham DFT, LDA (Teter 93 Pade -- the functional the GTH-PADE
pseudopotentials were generated with), GTH/HGH separable norm-conserving
pseudopotentials, a plane-wave basis with dense diagonalization at each k.
Hartree atomic units throughout.
"""
import numpy as np
from math import pi, sqrt, gamma, factorial

# ------------------------------------------------------------------ GTH tables
def load_gth(path, name):
    """Parse one entry of a CP2K-format GTH_POTENTIALS file."""
    lines = open(path).read().split('\n')
    i = next(k for k, l in enumerate(lines)
             if not l.startswith('#') and l.split() and l.split()[:2] == name.split()[:2])
    toks = lambda k: lines[k].split()
    zion = sum(int(t) for t in toks(i + 1))
    t = toks(i + 2)
    rloc, nexp = float(t[0]), int(t[1])
    c = [float(x) for x in t[2:2 + nexp]] + [0.0] * (4 - nexp)
    nshell = int(toks(i + 3)[0])
    k = i + 4
    shells = []
    for l in range(nshell):
        t = toks(k)
        r, npr = float(t[0]), int(t[1])
        h = np.zeros((3, 3))
        row = [float(x) for x in t[2:]]
        k += 1
        for a in range(npr):
            if a > 0:
                row = [float(x) for x in toks(k)]
                k += 1
            for b, val in enumerate(row):
                h[a, a + b] = h[a + b, a] = val
        shells.append(dict(l=l, r=r, nproj=npr, h=h))
    return dict(name=name, zion=zion, rloc=rloc, c=c, shells=shells)


# ------------------------------------------------------------- GTH closed forms
def vloc_g(pp, g, omega):
    """Local pseudopotential of ONE atom in reciprocal space, including 1/omega.
    g is |G| (array). The G = 0 entry is the finite part (the divergent -4 pi Z /
    (omega G^2) cancels against Hartree and Ewald)."""
    g = np.asarray(g, float)
    z, rl, (c1, c2, c3, c4) = pp['zion'], pp['rloc'], pp['c']
    x2 = (g * rl) ** 2
    e = np.exp(-0.5 * x2)
    poly = (c1 + c2 * (3 - x2) + c3 * (15 - 10 * x2 + x2 ** 2)
            + c4 * (105 - 105 * x2 + 21 * x2 ** 2 - x2 ** 3))
    out = np.empty_like(g)
    nz = g > 1e-12
    out[nz] = (-4 * pi * z / g[nz] ** 2 * e[nz] + sqrt(8 * pi ** 3) * rl ** 3 * e[nz] * poly[nz]) / omega
    out[~nz] = (2 * pi * z * rl ** 2 + sqrt(8 * pi ** 3) * rl ** 3 * (c1 + 3 * c2 + 15 * c3 + 105 * c4)) / omega
    return out


def proj_g(l, i, r, k, omega):
    """GTH projector p_i^l(|K|), i = 1..3, including 1/sqrt(omega). Defined so that
    <K|V_nl|K'> = sum_l (2l+1)/(4 pi) P_l(cos) sum_ij p_i(K) h_ij p_j(K')."""
    k = np.asarray(k, float)
    x2 = (k * r) ** 2
    e = np.exp(-0.5 * x2) * pi ** 1.25 / sqrt(omega)
    if l == 0:
        return [4 * sqrt(2 * r ** 3) * e,
                8 * sqrt(2 * r ** 3 / 15) * (3 - x2) * e,
                16 * sqrt(2 * r ** 3 / 105) * (15 - 10 * x2 + x2 ** 2) * e / 3][i - 1]
    if l == 1:
        return [8 * sqrt(r ** 5 / 3) * k * e,
                16 * sqrt(r ** 5 / 105) * k * (5 - x2) * e,
                32 * sqrt(r ** 5 / 1155) * k * (35 - 14 * x2 + x2 ** 2) * e / 3][i - 1]
    if l == 2:
        return [8 * sqrt(2 * r ** 7 / 15) * k ** 2 * e,
                16 * sqrt(2 * r ** 7 / 105) * k ** 2 * (7 - x2) * e / 3][i - 1]
    if l == 3:
        return [16 * sqrt(r ** 9 / 105) * k ** 3 * e][i - 1]
    raise ValueError(l)


def proj_r(l, i, r, x):
    """The same projector in REAL space -- its definition (HGH eq. 3)."""
    a = l + (4 * i - 1) / 2.0
    return sqrt(2) * x ** (l + 2 * (i - 1)) * np.exp(-x ** 2 / (2 * r * r)) / (r ** a * sqrt(gamma(a)))


def sph_jl(l, x):
    from scipy.special import spherical_jn
    return spherical_jn(l, x)


# -------------------------------------------------------------------- LDA (Pade)
_A = (0.4581652932831429, 2.217058676663745, 0.7405551735357053, 0.01968227878617998)
_B = (1.0, 4.504130959426697, 1.110667363742916, 0.02359291751427506)


def lda_pade(n):
    """Teter-93 Pade LDA. Returns (eps_xc, v_xc) per particle / potential."""
    n = np.maximum(n, 1e-30)
    rs = (3.0 / (4 * pi * n)) ** (1.0 / 3.0)
    a0, a1, a2, a3 = _A
    b1, b2, b3, b4 = _B
    num = a0 + rs * (a1 + rs * (a2 + rs * a3))
    den = rs * (b1 + rs * (b2 + rs * (b3 + rs * b4)))
    eps = -num / den
    dnum = a1 + rs * (2 * a2 + rs * 3 * a3)
    dden = b1 + rs * (2 * b2 + rs * (3 * b3 + rs * 4 * b4))
    deps = -(dnum * den - num * dden) / den ** 2
    return eps, eps - rs / 3.0 * deps


# ------------------------------------------------------------------ the crystal
class Crystal:
    def __init__(self, avec, species, pps):
        """avec: 3x3 rows = primitive vectors (bohr). species: list of
        (pp_index, fractional_position)."""
        self.a = np.array(avec, float)
        self.omega = abs(np.linalg.det(self.a))
        self.b = 2 * pi * np.linalg.inv(self.a).T      # rows: b_i . a_j = 2 pi delta
        self.pps = pps
        self.atoms = [(s, np.array(f, float) @ self.a) for s, f in species]
        self.nelec = sum(pps[s]['zion'] for s, _ in species)


def pw_basis(cr, kfrac, ecut):
    """Integer triples m with |k + G|^2 / 2 <= ecut, G = m . b."""
    mmax = int(np.ceil(sqrt(2 * ecut) / np.min(np.linalg.norm(cr.b, axis=1)))) + 2
    rng = np.arange(-mmax, mmax + 1)
    m = np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    kg = (m + kfrac) @ cr.b
    keep = 0.5 * np.sum(kg ** 2, axis=1) <= ecut
    # a fixed, reproducible order: ascending kinetic energy, ties by index
    m, kg = m[keep], kg[keep]
    order = np.lexsort((m[:, 2], m[:, 1], m[:, 0], np.round(np.sum(kg ** 2, 1), 9)))
    return m[order], kg[order]


def legendre(l, c):
    return [np.ones_like(c), c, 1.5 * c * c - 0.5, 2.5 * c ** 3 - 1.5 * c][l]


def vnl_matrix(cr, m, kg):
    """The nonlocal pseudopotential in the plane-wave basis at one k."""
    npw = len(m)
    knorm = np.linalg.norm(kg, axis=1)
    safe = np.where(knorm > 1e-12, knorm, 1.0)
    khat = kg / safe[:, None]
    cosg = np.clip(khat @ khat.T, -1, 1)
    v = np.zeros((npw, npw), complex)
    for s, tau in cr.atoms:
        pp = cr.pps[s]
        g = m @ cr.b
        phase = np.exp(-1j * (g @ tau))
        sf = phase[:, None] * phase[None, :].conj()      # exp(-i (G - G') . tau)
        for sh in pp['shells']:
            l, r, npr, h = sh['l'], sh['r'], sh['nproj'], sh['h']
            if npr == 0:
                continue
            p = np.array([proj_g(l, i + 1, r, knorm, cr.omega) for i in range(npr)])
            rad = np.einsum('ia,ij,jb->ab', p, h[:npr, :npr], p)
            v += sf * (2 * l + 1) / (4 * pi) * legendre(l, cosg) * rad
    return v


def fft_dims(cr, ecut, kfracs):
    """A grid big enough that V(G - G') and the density are both exact."""
    mm = 0
    for kf in kfracs:
        m, _ = pw_basis(cr, kf, ecut)
        mm = max(mm, np.abs(m).max())
    n = fft_nice(4 * mm + 2)
    return (n, n, n)


def fft_nice(n):
    """The smallest integer >= n with no prime factor above 5. Any side >= 4 mm + 2
    is correct; these are the sides Viv's FFT has a dedicated butterfly for (7 goes
    through the slow generic one, larger primes through Bluestein). The rule has
    to match pw_common.viv's exactly: the XC energy is a sum over this grid, so a
    different side is a (slightly) different number."""
    while True:
        rest = n
        for p in (2, 3, 5):
            while rest % p == 0:
                rest //= p
        if rest == 1:
            return n
        n += 1


FCC_SHIFTS = ((.5, .5, .5), (.5, 0, 0), (0, .5, 0), (0, 0, .5))


def mp_grid(n, shifts=FCC_SHIFTS):
    """The symmetric k-mesh for an fcc crystal, k = (i + s) / n over FOUR shifts,
    halved by time reversal (k and -k give one density). Returns fractional
    k-points and weights summing to 1.

    Four shifts because the density is only as symmetric as the mesh, and a single
    (1/2 1/2 1/2)-shifted mesh in primitive fcc coordinates is NOT closed under the
    cubic point group: it splits silicon's valence-band triplet by 56 meV at 2x2x2
    (kmesh_symmetry.py). The four are that shift's orbit. Same order as kmesh.viv."""
    pts = {}
    for s in shifts:
        for i in range(n[0]):
            for j in range(n[1]):
                for l in range(n[2]):
                    k = np.array([(i + s[0]) / n[0], (j + s[1]) / n[1], (l + s[2]) / n[2]])
                    k = k - np.round(k)                      # into (-1/2, 1/2]
                    k[np.isclose(k, -0.5)] = 0.5
                    mk = -k
                    mk[np.isclose(mk, -0.5)] = 0.5
                    key, mkey = tuple(np.round(k, 9)), tuple(np.round(mk, 9))
                    if mkey in pts and mkey != key:
                        pts[mkey] += 1
                    else:
                        pts[key] = pts.get(key, 0) + 1
    ks = np.array(list(pts.keys()))
    w = np.array(list(pts.values()), float)
    return ks, w / w.sum()


def occupations(eigs, weights, nelec, smear):
    """Occupations f in [0, 2] and the Fermi level. smear = 0: an insulator,
    the lowest nelec/2 bands filled. smear > 0: Fermi-Dirac of that width (Ha),
    with the Fermi level found by bisection on the electron count."""
    eigs = np.asarray(eigs)
    if smear <= 0:
        f = np.zeros_like(eigs)
        f[:, : int(round(nelec / 2))] = 2.0
        return f, None, 0.0
    def count(mu):
        x = np.clip((eigs - mu) / smear, -200, 200)
        return 2.0 / (1.0 + np.exp(x))
    lo, hi = eigs.min() - 1.0, eigs.max() + 1.0
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if np.sum(weights[:, None] * count(mu)) > nelec:
            hi = mu
        else:
            lo = mu
    f = count(mu)
    h = np.clip(f / 2.0, 1e-300, 1 - 1e-16)
    ent = -2.0 * np.sum(weights[:, None] * (h * np.log(h) + (1 - h) * np.log(1 - h)))
    return f, mu, ent


def scf(cr, ecut, kfracs, kweights, nbands, mix=0.6, tol=1e-9, maxit=80, smear=0.0, verbose=True, anderson=0, dims=None):
    kweights = np.asarray(kweights, float)
    # the Viv engine sizes its grid for the band path's k-points too; a caller
    # that will compare against it passes the grid for that union
    if dims is None:
        dims = fft_dims(cr, ecut, kfracs)
    N = np.prod(dims)
    fr = [np.fft.fftfreq(d, 1.0 / d).astype(int) for d in dims]
    M = np.array(np.meshgrid(*fr, indexing='ij'))
    G = np.tensordot(M, cr.b, axes=([0], [0]))
    g2 = np.sum(G ** 2, axis=-1)
    gn = np.sqrt(g2)
    vps = np.zeros(dims, complex)
    for s, tau in cr.atoms:
        vps += vloc_g(cr.pps[s], gn.ravel(), cr.omega).reshape(dims) * np.exp(-1j * (G @ tau))
    bases = [pw_basis(cr, kf, ecut) for kf in kfracs]
    vnls = [vnl_matrix(cr, m, kg) for m, kg in bases]
    rho = np.full(dims, cr.nelec / cr.omega)
    e_old = 0.0
    hist = []
    for it in range(maxit):
        rho_g = np.fft.fftn(rho) / N
        with np.errstate(divide='ignore', invalid='ignore'):
            vh_g = np.where(g2 > 1e-12, 4 * pi * rho_g / g2, 0.0)
        exc, vxc = lda_pade(rho)
        vtot_g = vps + vh_g + np.fft.fftn(vxc) / N
        evs, cs = [], []
        for (m, kg), vnl in zip(bases, vnls):
            d = m[:, None, :] - m[None, :, :]
            h = vtot_g[d[..., 0] % dims[0], d[..., 1] % dims[1], d[..., 2] % dims[2]] + vnl
            h[np.diag_indices(len(m))] += 0.5 * np.sum(kg ** 2, axis=1)
            ev, c = np.linalg.eigh(h)
            evs.append(ev[:nbands])
            cs.append(c[:, :nbands])
        f, mu, ent = occupations(evs, kweights, cr.nelec, smear)
        new = np.zeros(dims)
        eband = 0.0
        for (m, kg), c, ev, fk, w in zip(bases, cs, evs, f, kweights):
            eband += w * np.sum(fk * ev)
            for n_ in range(nbands):
                if fk[n_] < 1e-14:
                    continue
                box = np.zeros(dims, complex)
                box[m[:, 0] % dims[0], m[:, 1] % dims[1], m[:, 2] % dims[2]] = c[:, n_]
                psi = np.fft.ifftn(box) * N / sqrt(cr.omega)
                new += w * fk[n_] * np.abs(psi) ** 2
        eh = 0.5 * cr.omega * np.sum(np.where(g2 > 1e-12, 4 * pi * np.abs(rho_g) ** 2 / np.maximum(g2, 1e-30), 0.0))
        dv = cr.omega / N
        etot = eband - eh - np.sum(vxc * rho) * dv + np.sum(exc * rho) * dv - smear * ent
        drho = sqrt(np.sum((new - rho) ** 2) * dv)
        if verbose:
            print(f"  it {it:2d}  F(no Ewald) = {etot: .10f}  |drho| = {drho:.3e}" + (f"  mu = {mu:.6f}" if mu is not None else ""))
        if drho < tol and abs(etot - e_old) < tol:
            break
        e_old = etot
        if anderson <= 0:
            rho = rho + mix * (new - rho)
        else:
            # Anderson acceleration over the last `anderson` iterates. With
            # F = rho_out - rho_in the residual, pick the combination of recent
            # iterates whose residual is smallest, then take a damped step from
            # it. Linear mixing is the anderson = 0 special case.
            F = new - rho
            hist.append((rho.copy(), F.copy()))
            hist[:] = hist[-(anderson + 1):]
            m = len(hist) - 1
            rbar, fbar = rho, F
            if m > 0:
                dF = [F - hist[-1 - j][1] for j in range(1, m + 1)]
                dR = [rho - hist[-1 - j][0] for j in range(1, m + 1)]
                A = np.array([[np.sum(a * b) for b in dF] for a in dF])
                bvec = np.array([np.sum(a * F) for a in dF])
                theta = np.linalg.solve(A + 1e-14 * np.trace(A) * np.eye(m), bvec)
                rbar = rho - sum(t * d for t, d in zip(theta, dR))
                fbar = F - sum(t * d for t, d in zip(theta, dF))
            rho = rbar + mix * fbar
    return dict(etot=etot, rho=rho, vtot_g=vtot_g, dims=dims, eigs=evs, iters=it + 1, mu=mu, occ=f)


def bands(cr, ecut, vtot_g, dims, kfrac, nbands):
    m, kg = pw_basis(cr, kfrac, ecut)
    d = m[:, None, :] - m[None, :, :]
    h = vtot_g[d[..., 0] % dims[0], d[..., 1] % dims[1], d[..., 2] % dims[2]] + vnl_matrix(cr, m, kg)
    h[np.diag_indices(len(m))] += 0.5 * np.sum(kg ** 2, axis=1)
    return np.linalg.eigvalsh(h)[:nbands]

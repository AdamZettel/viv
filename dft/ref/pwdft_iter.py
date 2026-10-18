# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Iterative plane-wave DFT -- the numpy reference for dft/pwdft_it.viv.

pwdft_ref.py builds H(k) and diagonalizes it, which is O(npw^3) per k-point and
stops being possible somewhere around a thousand plane waves. Copper needs
~1500. This is the architecture every production plane-wave code uses instead:

  * H is never built. It is APPLIED: kinetic energy is diagonal in G, the local
    potential is diagonal in r (so: FFT, multiply, FFT back), and the nonlocal
    pseudopotential is a sum of projectors, sum_IJ |beta_I> D_IJ <beta_J|.
  * Only the bands that are needed are solved for, by LOBPCG -- locally optimal
    block preconditioned conjugate gradient -- warm-started from the previous
    SCF iteration's wavefunctions.

Cost per k-point goes from npw^3 to (bands) x (npw log npw).

It is written to be mirrored line for line in Viv: vectors are ROWS, arrays are
padded to a common npw with a mask, the starting vectors come from the same
MINSTD stream Viv's rand() uses, and every iteration count is fixed rather than
data-dependent, so the two implementations can be compared step by step.
"""
import numpy as np
from math import pi, sqrt
from pwdft_ref import vloc_g, proj_g, lda_pade, occupations

# ----------------------------------------------------------------- MINSTD
class Minstd:
    """Viv's srand()/rand(): Park-Miller, exact in f64, bit-identical here."""
    def __init__(self, seed):
        self.s = float(int(abs(seed)) % 2147483646 + 1)
    def __call__(self):
        self.s = (16807.0 * self.s) % 2147483647.0
        return self.s / 2147483647.0


# ------------------------------------------------- real spherical harmonics
def real_ylm(l, khat):
    """Real spherical harmonics S_lm(khat) for one l, as rows [m, npw]. They
    satisfy sum_m S_lm(a) S_lm(b) = (2l+1)/(4 pi) P_l(a.b), which is what makes
    the separable projector form equal to the Legendre form pwdft_ref.py uses."""
    x, y, z = khat[:, 0], khat[:, 1], khat[:, 2]
    if l == 0:
        return np.array([np.full_like(x, sqrt(1 / (4 * pi)))])
    if l == 1:
        c = sqrt(3 / (4 * pi))
        return np.array([c * x, c * y, c * z])
    if l == 2:
        c = sqrt(15 / (4 * pi))
        return np.array([c * x * y, c * y * z, c * z * x,
                         sqrt(5 / (16 * pi)) * (3 * z * z - 1),
                         sqrt(15 / (16 * pi)) * (x * x - y * y)])
    raise ValueError("f projectors (l = 3) are not implemented")


def basis_loop_order(cr, kfrac, ecut):
    """Plane waves in the order Viv's loops produce them: m1 outer, m3 inner."""
    amax = np.linalg.norm(cr.a, axis=1).max()
    bmax = np.linalg.norm(cr.b, axis=1).max()
    mm = int(np.floor((sqrt(2 * ecut) + bmax) * amax / (2 * pi))) + 1
    rng = np.arange(-mm, mm + 1)
    m = np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    kg = (m + np.asarray(kfrac, float)) @ cr.b
    keep = 0.5 * np.sum(kg ** 2, axis=1) <= ecut
    return m[keep], kg[keep]


class KPoint:
    """Everything one k-point needs to have H applied to it."""
    def __init__(self, cr, kfrac, ecut, dims):
        self.m, self.kg = basis_loop_order(cr, kfrac, ecut)
        self.npw = len(self.m)
        self.t = 0.5 * np.sum(self.kg ** 2, axis=1)               # kinetic energies
        self.idx = (self.m[:, 0] % dims[0], self.m[:, 1] % dims[1], self.m[:, 2] % dims[2])
        knorm = np.linalg.norm(self.kg, axis=1)
        khat = self.kg / np.where(knorm > 1e-12, knorm, 1.0)[:, None]
        g = self.m @ cr.b
        betas, blocks = [], []
        for s, tau in cr.atoms:
            phase = np.exp(-1j * (g @ tau))
            for sh in cr.pps[s]['shells']:
                l, r, npr, h = sh['l'], sh['r'], sh['nproj'], sh['h']
                if npr == 0:
                    continue
                ylm = real_ylm(l, khat)
                # at K = 0 the direction is undefined; p_i^l(0) = 0 for l > 0 anyway
                rad = [proj_g(l, i + 1, r, knorm, cr.omega) for i in range(npr)]
                for mrow in ylm:
                    first = len(betas)
                    for i in range(npr):
                        betas.append(phase * mrow * rad[i])
                    blocks.append((first, npr, h[:npr, :npr]))
        self.beta = np.array(betas)                                  # [nproj, npw]
        self.d = np.zeros((len(betas), len(betas)))
        for first, npr, h in blocks:
            self.d[first:first + npr, first:first + npr] = h


def hpsi(kp, x, vr, dims):
    """Apply H to the rows of x without ever forming H."""
    n = np.prod(dims)
    out = kp.t[None, :] * x
    for b in range(x.shape[0]):
        box = np.zeros(dims, complex)
        box[kp.idx] = x[b]
        box = np.fft.fftn(np.fft.ifftn(box) * vr)
        out[b] += box[kp.idx]
    if len(kp.beta):
        pr = x @ kp.beta.conj().T                                    # <beta_J | psi_b>
        out += (pr @ kp.d.T) @ kp.beta
    return out


def ortho_matrix(s, drop=1e-10):
    """Canonical orthonormalization: eigendecompose the overlap and discard its
    near-null directions. Returns the [m, ns] matrix that does it. Rank-revealing,
    so a search direction that has become linearly dependent is dropped instead
    of poisoning the Rayleigh-Ritz step -- the classic way LOBPCG falls over."""
    o = s.conj() @ s.T
    w, u = np.linalg.eigh(o)
    keep = w > drop * w.max()
    return (u[:, keep] / np.sqrt(w[keep])).T


def ritz(s, hs, nb):
    """The nb lowest Ritz pairs in the span of the (orthonormal) rows of s."""
    a = s.conj() @ hs.T
    e, c = np.linalg.eigh(0.5 * (a + a.conj().T))    # symmetrized: hs is tracked, not recomputed
    return e[:nb], c[:, :nb].T


def tpa(kp, x, r):
    """Teter-Payne-Allan preconditioner, one kinetic scale per band."""
    out = np.empty_like(r)
    for b in range(r.shape[0]):
        ekin = max(float(np.sum(np.abs(x[b]) ** 2 * kp.t)), 1e-8)
        q = kp.t / (1.5 * ekin)
        num = 27 + q * (18 + q * (12 + q * 8))
        out[b] = r[b] * num / (num + 16 * q ** 4)
    return out


def lobpcg(kp, x, vr, dims, nit, nconv=None, tol=0.0):
    """Up to nit LOBPCG steps on the rows of x. Returns Ritz values, vectors, and
    the largest residual norm among the lowest nconv bands; stops early once that
    is under tol. The bands above nconv are a buffer: the top of a block
    converges slowly, and is not held to the tolerance."""
    nb = x.shape[0]
    nconv = nb if nconv is None else nconv
    hx = hpsi(kp, x, vr, dims)
    # a Rayleigh-Ritz step first, so x are Ritz vectors and x^H H x is diagonal
    mt = ortho_matrix(x)
    x, hx = mt @ x, mt @ hx
    _, ct = ritz(x, hx, nb)
    x, hx = ct @ x, ct @ hx
    p = hp = None
    res = 0.0
    for _ in range(nit):
        lam = np.real(np.sum(x.conj() * hx, axis=1))
        r = hx - lam[:, None] * x
        rn = np.sqrt(np.sum(np.abs(r) ** 2, axis=1))
        res = float(rn[:nconv].max())
        w = tpa(kp, x, r)
        # soft locking: a converged band offers no search direction (its zero row
        # is dropped by ortho_matrix); it still takes part in Rayleigh-Ritz
        w[rn < max(tol, 1e-9)] = 0.0
        if res < tol:
            break
        w = w - (w @ x.conj().T) @ x
        # unit rows: a residual's norm is the residual, so un-normalized its overlap
        # eigenvalue is res^2, and ortho_matrix -- which drops directions 1e-10
        # below the largest -- would discard the correction at res ~ 1e-5 and stall
        w = w / np.maximum(np.sqrt(np.sum(np.abs(w) ** 2, axis=1)), 1e-300)[:, None]
        hw = hpsi(kp, w, vr, dims)
        s = np.vstack([x, w] + ([p] if p is not None else []))
        hs = np.vstack([hx, hw] + ([hp] if hp is not None else []))
        mt = ortho_matrix(s)
        _, ct = ritz(mt @ s, mt @ hs, nb)
        km = ct @ mt                       # the new vectors in terms of the rows [x; w; p]
        xn, hxn = km @ s, km @ hs
        # The new search direction: the part of xn that came from w and p -- the
        # same combination with the x columns struck out. Built DIRECTLY, never as
        # xn - (xn x^H) x: that subtraction cancels to the size of the step and
        # leaves p with no relative accuracy, (p, hp) stops being consistent with
        # H once normalized, and a converged iteration then DIVERGES.
        km[:, :nb] = 0.0
        p, hp = km @ s, km @ hs
        pn = np.maximum(np.sqrt(np.sum(np.abs(p) ** 2, axis=1)), 1e-300)[:, None]
        p, hp = p / pn, hp / pn
        x, hx = xn, hxn
    return np.real(np.sum(x.conj() * hx, axis=1)), x, res


def start_vectors(kp, nb, rng):
    """The nb lowest-kinetic plane waves, plus a little MINSTD noise so that no
    eigenvector is orthogonal to the starting subspace by symmetry."""
    x = np.zeros((nb, kp.npw), complex)
    t = kp.t.copy()
    for b in range(nb):
        i = int(np.argmin(t))                # ties: lowest index, as Viv's loop finds
        x[b, i] = 1.0
        t[i] = 1e300
    for b in range(nb):
        for i in range(kp.npw):
            x[b, i] += 0.02 * (rng() - 0.5) / (1.0 + kp.t[i])
    return x


def scf(cr, ecut, kfracs, kweights, nbands, dims, smear=0.0, mix=0.5, anderson=3,
        nit=3, tol=1e-9, eig_tol=1e-5, maxit=80, verbose=True):
    kweights = np.asarray(kweights, float)
    n = np.prod(dims)
    fr = [np.fft.fftfreq(d, 1.0 / d).astype(int) for d in dims]
    mgrid = np.array(np.meshgrid(*fr, indexing='ij'))
    gvec = np.tensordot(mgrid, cr.b, axes=([0], [0]))
    g2 = np.sum(gvec ** 2, axis=-1)
    vps = np.zeros(dims, complex)
    for s, tau in cr.atoms:
        vps += vloc_g(cr.pps[s], np.sqrt(g2).ravel(), cr.omega).reshape(dims) * np.exp(-1j * (gvec @ tau))
    kps = [KPoint(cr, kf, ecut, dims) for kf in kfracs]
    rng = Minstd(1)
    xs = [start_vectors(kp, nbands, rng) for kp in kps]
    rho = np.full(dims, cr.nelec / cr.omega)
    dv = cr.omega / n
    hist, e_old = [], 0.0
    nconv = nbands          # every band, until there are occupations to go by
    for it in range(maxit):
        rho_g = np.fft.fftn(rho) / n
        with np.errstate(divide='ignore', invalid='ignore'):
            vh_g = np.where(g2 > 1e-12, 4 * pi * rho_g / g2, 0.0)
        exc, vxc = lda_pade(rho)
        vr = np.real(np.fft.ifftn(vps + vh_g) * n) + vxc
        evs, worst = [], 0.0
        for q, kp in enumerate(kps):
            e, xs[q], res = lobpcg(kp, xs[q], vr, dims, nit, nconv, 0.1 * eig_tol)
            evs.append(e)
            worst = max(worst, res)
        f, mu, ent = occupations(evs, kweights, cr.nelec, smear)
        # the bands the density depends on: up to the highest one occupied anywhere
        nconv = 1 + max(b for fk in f for b in range(nbands) if fk[b] > 1e-10)
        new = np.zeros(dims)
        eband = 0.0
        for kp, x, ev, fk, w in zip(kps, xs, evs, f, kweights):
            eband += w * np.sum(fk * ev)
            for b in range(nbands):
                if fk[b] < 1e-14:
                    continue
                box = np.zeros(dims, complex)
                box[kp.idx] = x[b]
                new += w * fk[b] * np.abs(np.fft.ifftn(box) * n / sqrt(cr.omega)) ** 2
        eh = 0.5 * cr.omega * np.sum(np.where(g2 > 1e-12, 4 * pi * np.abs(rho_g) ** 2 / np.maximum(g2, 1e-30), 0.0))
        etot = eband - eh - np.sum(vxc * rho) * dv + np.sum(exc * rho) * dv - smear * ent
        resid = new - rho
        drho = sqrt(np.sum(resid ** 2) * dv)
        if verbose:
            print(f"  it {it:2d}  F = {etot: .10f}  |drho| = {drho:.3e}  eig res = {worst:.2e}"
                  + (f"  mu = {mu:.6f}" if mu is not None else ""), flush=True)
        if drho < tol and abs(etot - e_old) < tol and worst < eig_tol:
            break
        e_old = etot
        hist.append((rho.copy(), resid.copy()))
        hist[:] = hist[-(anderson + 1):]
        m = len(hist) - 1
        rbar, fbar = rho, resid
        if m > 0:
            df = [resid - hist[-1 - j][1] for j in range(1, m + 1)]
            dr = [rho - hist[-1 - j][0] for j in range(1, m + 1)]
            amat = np.array([[np.sum(a * b) for b in df] for a in df])
            bvec = np.array([np.sum(a * resid) for a in df])
            theta = np.linalg.solve(amat + 1e-14 * np.trace(amat) * np.eye(m), bvec)
            rbar = rho - sum(t * d for t, d in zip(theta, dr))
            fbar = resid - sum(t * d for t, d in zip(theta, df))
        rho = rbar + mix * fbar
    return dict(etot=etot, rho=rho, vr=vr, dims=dims, eigs=evs, iters=it + 1, mu=mu, occ=f, kps=kps, xs=xs)


def bands_at(cr, ecut, vr, dims, kfrac, nbands, nit=40, eig_tol=1e-5):
    """Non-self-consistent bands at one k: LOBPCG from a cold start, until the
    valence bands and two above are converged."""
    kp = KPoint(cr, kfrac, ecut, dims)
    x = start_vectors(kp, nbands, Minstd(7))
    nconv = min(nbands, cr.nelec // 2 + 2)
    e, x, res = lobpcg(kp, x, vr, dims, nit, nconv, 0.1 * eig_tol)
    return e, res

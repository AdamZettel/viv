# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""An independent power flow in NumPy/SciPy, the check for lib/power.

Written in MATPOWER's matrix form (the MATPOWER manual, sections 3 and 4;
PYPOWER is the same algebra): Y from the incidence matrices Cf and Ct,
the Jacobian from dS/dVa = j diag(V) conj(diag(I) - Y diag(V)) and
dS/dVm = diag(V) conj(Y diag(V/|V|)) + conj(diag(I)) diag(V/|V|), solved by
SciPy's spsolve (SuperLU). lib/power assembles everything entry by entry
and factors with its own LU, so the two share formulas, not code.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

DEG = np.pi / 180


def _internal(c):
    bus, gen, br = c["bus"], c["gen"], c["branch"]
    idx = {int(b): i for i, b in enumerate(bus[:, 0])}
    f = np.array([idx[int(x)] for x in br[:, 0]])
    t = np.array([idx[int(x)] for x in br[:, 1]])
    g = np.array([idx[int(x)] for x in gen[:, 0]])
    return f, t, g


def make_ybus(c, f, t, unit_tap=False, no_r=False, no_b=False, no_shift=False, no_bs=False):
    bus, br = c["bus"], c["branch"]
    nb, nl = bus.shape[0], br.shape[0]
    st = br[:, 10]
    r = np.zeros(nl) if no_r else br[:, 2]
    ys = st / (r + 1j * br[:, 3])
    bc = np.zeros(nl) if no_b else st * br[:, 4]
    tap = np.ones(nl)
    if not unit_tap:
        tap[br[:, 8] != 0] = br[br[:, 8] != 0, 8]
    if not no_shift:
        tap = tap * np.exp(1j * DEG * br[:, 9])
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ysh = (bus[:, 4] + (0 if no_bs else 1j * bus[:, 5])) / c["baseMVA"]
    rows = np.arange(nl)
    Cf = sp.csr_matrix((np.ones(nl), (rows, f)), (nl, nb))
    Ct = sp.csr_matrix((np.ones(nl), (rows, t)), (nl, nb))
    Yf = sp.diags(yff) @ Cf + sp.diags(yft) @ Ct
    Yt = sp.diags(ytf) @ Cf + sp.diags(ytt) @ Ct
    Y = Cf.T @ Yf + Ct.T @ Yt + sp.diags(ysh)
    return Y.tocsc(), Yf.tocsr(), Yt.tocsr()


def bus_types(c, g):
    bus, gen = c["bus"], c["gen"]
    on = np.zeros(bus.shape[0])
    np.add.at(on, g, gen[:, 7] > 0)
    ty = bus[:, 1]
    ref = np.flatnonzero((ty == 3) & (on > 0))
    pv = np.flatnonzero((ty == 2) & (on > 0))
    pq = np.flatnonzero(((ty == 1) | (on == 0)) & (ty != 4))
    if len(ref) == 0:
        ref, pv = pv[:1], pv[1:]
    return ref, pv, pq


def sbus_v0(c, g):
    bus, gen = c["bus"], c["gen"]
    on = gen[:, 7] > 0
    S = -(bus[:, 2] + 1j * bus[:, 3]) / c["baseMVA"]
    np.add.at(S, g[on], (gen[on, 1] + 1j * gen[on, 2]) / c["baseMVA"])
    vm = bus[:, 7].copy()
    vm[g[on]] = gen[on, 5]
    return S, vm * np.exp(1j * DEG * bus[:, 8])


def jacobian(Y, V, pv, pq):
    I = Y @ V
    Vn = V / abs(V)
    dVa = 1j * sp.diags(V) @ np.conj(sp.diags(I) - Y @ sp.diags(V))
    dVm = sp.diags(V) @ np.conj(Y @ sp.diags(Vn)) + np.conj(sp.diags(I)) @ sp.diags(Vn)
    pvpq = np.r_[pv, pq]
    dVa, dVm = sp.csr_matrix(dVa), sp.csr_matrix(dVm)
    return sp.vstack([sp.hstack([dVa[pvpq][:, pvpq].real, dVm[pvpq][:, pq].real]),
                      sp.hstack([dVa[pq][:, pvpq].imag, dVm[pq][:, pq].imag])]).tocsc()


def newton(Y, S, V, pv, pq, tol=1e-8, max_it=10):
    pvpq = np.r_[pv, pq]
    npv, npq = len(pv), len(pq)
    Va, Vm = np.angle(V), abs(V)

    def F(V):
        mis = V * np.conj(Y @ V) - S
        return np.r_[mis[pvpq].real, mis[pq].imag]

    f = F(V)
    it = 0
    while np.max(abs(f)) >= tol and it < max_it:
        it += 1
        dx = -spsolve(jacobian(Y, V, pv, pq), f)
        Va[pvpq] += dx[:npv + npq]
        Vm[pq] += dx[npv + npq:]
        V = Vm * np.exp(1j * Va)
        f = F(V)
    return V, np.max(abs(f)) < tol, it


def solution(c, f, t, g, Y, Yf, Yt, V, ref):
    """bus Vm, Va (deg), gen Pg, Qg, branch Pf, Qf, Pt, Qt as MATPOWER's pfsoln."""
    bus, gen, base = c["bus"], c["gen"].copy(), c["baseMVA"]
    on = np.flatnonzero(gen[:, 7] > 0)
    Sg = V[g[on]] * np.conj(Y[g[on], :] @ V)
    gen[on, 2] = Sg.imag * base + bus[g[on], 3]
    if len(on) > 1:
        nb = bus.shape[0]
        Cg = sp.csr_matrix((np.ones(len(on)), (np.arange(len(on)), g[on])), (len(on), nb))
        ngg = Cg @ np.asarray(Cg.sum(axis=0)).ravel()
        gen[on, 2] /= ngg
        M = abs(gen[on, 2]).sum() + sum(abs(x) for x in gen[on, 3] if np.isfinite(x)) \
            + sum(abs(x) for x in gen[on, 4] if np.isfinite(x))
        qmax = np.minimum(gen[on, 3], M)
        qmin = np.maximum(gen[on, 4], -M)
        tot, tmin, tmax = Cg.T @ gen[on, 2], Cg.T @ qmin, Cg.T @ qmax
        eps = np.finfo(float).eps
        q = qmin + (Cg @ ((tot - tmin) / (tmax - tmin + eps))) * (qmax - qmin)
        flat = Cg @ (abs(tmax - tmin) < 10 * eps)
        q[flat > 0] = qmin[flat > 0] + (Cg @ (tot - tmin))[flat > 0] / ngg[flat > 0]
        gen[on, 2] = q
    for r in ref:
        rg = on[g[on] == r]
        Pg = (V[r] * np.conj(Y[r, :] @ V)).real.item() * base + bus[r, 2]
        gen[rg[0], 1] = Pg - gen[rg[1:], 1].sum()
    Sf = V[f] * np.conj(Yf @ V) * base
    St = V[t] * np.conj(Yt @ V) * base
    return dict(vm=abs(V), va=np.angle(V) / DEG, pg=gen[:, 1], qg=gen[:, 2], pf=Sf.real, qf=Sf.imag, pt=St.real,
                qt=St.imag)


def runpf_nr(c, tol=1e-8):
    f, t, g = _internal(c)
    Y, Yf, Yt = make_ybus(c, f, t)
    ref, pv, pq = bus_types(c, g)
    S, V0 = sbus_v0(c, g)
    V, ok, it = newton(Y, S, V0, pv, pq, tol)
    out = solution(c, f, t, g, Y, Yf, Yt, V, ref)
    out.update(success=ok, iterations=it)
    return out


def runpf_dc(c):
    """MATPOWER's makeBdc + dcpf."""
    f, t, g = _internal(c)
    bus, br, base = c["bus"], c["branch"], c["baseMVA"]
    nb, nl = bus.shape[0], br.shape[0]
    tap = np.where(br[:, 8] != 0, br[:, 8], 1.0)
    b = br[:, 10] / br[:, 3] / tap
    rows = np.arange(nl)
    Cft = sp.csr_matrix((np.r_[np.ones(nl), -np.ones(nl)], (np.r_[rows, rows], np.r_[f, t])), (nl, nb))
    Bf = sp.diags(b) @ Cft
    B = (Cft.T @ Bf).tocsr()
    Pfinj = b * (-br[:, 9] * DEG)
    Pbusinj = Cft.T @ Pfinj
    ref, pv, pq = bus_types(c, g)
    S, _ = sbus_v0(c, g)
    P = S.real - Pbusinj - bus[:, 4] / base
    Va = bus[:, 8] * DEG
    pvpq = np.r_[pv, pq]
    Va[pvpq] = spsolve(B[pvpq][:, pvpq].tocsc(), P[pvpq] - B[pvpq][:, ref] @ Va[ref])
    gen = c["gen"].copy()
    on = np.flatnonzero(gen[:, 7] > 0)
    rg = on[g[on] == ref[0]]
    gen[rg[0], 1] += (B[ref[0], :] @ Va - P[ref[0]]).item() * base
    pf = (Bf @ Va + Pfinj) * base
    z = np.zeros(nl)
    return dict(vm=np.ones(nb), va=Va / DEG, pg=gen[:, 1], qg=0 * gen[:, 2], pf=pf, qf=z, pt=-pf, qt=z,
                success=True, iterations=1)

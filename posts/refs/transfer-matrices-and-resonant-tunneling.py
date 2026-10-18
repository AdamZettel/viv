# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-transfer-matrices-and-resonant-tunneling.viv.

Independent of the Viv program: instead of multiplying 2x2 transfer matrices,
this solves the matching conditions of the whole potential at once, as one
linear system (numpy.linalg.solve), and finds the resonances of the double
barrier with scipy (bounded maximization of T, then brentq on a centered difference
of dT/dE to place each peak, and brentq for the energies where T = 1/2).

hbar = m = 1, so k = sqrt(2 (E - V)).

    python3 posts/refs/transfer-matrices-and-resonant-tunneling.py > posts/refs/transfer-matrices-and-resonant-tunneling.out
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq

np.set_printoptions(precision=17)


def scatter_direct(E, slabs, V_lead=0.0):
    """T and R by solving all matching equations at once.

    Regions: lead 0, slabs 1..n, lead n+1. psi_j = A_j e^{ik_j (x - x_j)} + B_j e^{-ik_j (x - x_j)},
    x_j the left edge of region j (x_0 = x_1 = 0 for the left lead). Unknowns:
    B_0 = r, (A_j, B_j) for each slab, A_{n+1} = t; A_0 = 1, B_{n+1} = 0.
    """
    Vs = [V_lead] + [V for V, _ in slabs] + [V_lead]
    ws = [0.0] + [d for _, d in slabs] + [0.0]
    ks = [np.sqrt(complex(2.0 * (E - V))) for V in Vs]
    edges = np.concatenate([[0.0], np.cumsum([d for _, d in slabs])])
    n = len(slabs)
    nreg = n + 2
    # unknown index: region j has (A_j, B_j) at 2j, 2j+1; A_0 and B_{n+1} fixed
    nunk = 2 * nreg
    rows = []
    rhs = []
    # origin of region j: left edge (lead 0 uses 0)
    orig = [0.0] + list(edges[:-1]) + [edges[-1]]
    for j in range(nreg - 1):
        x = edges[j]
        for deriv in (0, 1):
            row = np.zeros(nunk, dtype=complex)
            for reg, sign in ((j, 1.0), (j + 1, -1.0)):
                k = ks[reg]
                ea = np.exp(1j * k * (x - orig[reg]))
                eb = np.exp(-1j * k * (x - orig[reg]))
                fa = ea if deriv == 0 else 1j * k * ea
                fb = eb if deriv == 0 else -1j * k * eb
                row[2 * reg] += sign * fa
                row[2 * reg + 1] += sign * fb
            rows.append(row)
    A = np.array(rows)
    # move A_0 = 1 to the rhs, drop B_{n+1} (= 0)
    b = -A[:, 0]
    keep = [c for c in range(nunk) if c not in (0, nunk - 1)]
    sol = np.linalg.solve(A[:, keep], b)
    r = sol[0]
    t = sol[-1]
    return abs(t) ** 2, abs(r) ** 2


V0, a = 3.0, 1.0
single = [(V0, a)]
b, L = 1.0, 4.0
double = [(V0, b), (0.0, L), (V0, b)]


def barrier_analytic(E):
    if E < V0:
        kap = np.sqrt(2 * (V0 - E))
        return 1.0 / (1 + V0 ** 2 * np.sinh(kap * a) ** 2 / (4 * E * (V0 - E)))
    q = np.sqrt(2 * (E - V0))
    return 1.0 / (1 + V0 ** 2 * np.sin(q * a) ** 2 / (4 * E * (E - V0)))


Es = np.linspace(0.05, 6.0, 500)
d = max(abs(scatter_direct(E, single)[0] - barrier_analytic(E)) for E in Es if abs(E - V0) > 1e-9)
print(f"single barrier: max |T_direct - T_analytic| on 500 energies = {d:.2e}")
d = max(abs(sum(scatter_direct(E, single)) - 1) for E in Es)
print(f"single barrier: max |T + R - 1| = {d:.2e}")
d = max(abs(sum(scatter_direct(E, double)) - 1) for E in Es)
print(f"double barrier: max |T + R - 1| = {d:.2e}")

for E in (0.5, 1.0, 1.5):
    print(f"single barrier T({E}) = {barrier_analytic(E):.17g}")

for n in (1, 2, 3):
    E = V0 + (n * np.pi) ** 2 / (2 * a * a)
    print(f"over-barrier resonance n={n}: E = {E:.17g}, T = {scatter_direct(E, single)[0]:.15f}")

# double-barrier resonances below the top: scan, then maximize T in each bracket
T = lambda E: scatter_direct(E, double)[0]
grid = np.linspace(0.005, V0 - 0.005, 30000)
Tg = np.array([T(E) for E in grid])
peaks = [i for i in range(1, len(grid) - 1) if Tg[i] > Tg[i - 1] and Tg[i] >= Tg[i + 1] and Tg[i] > 0.5]
print(f"double barrier (V0 = {V0}, b = {b}, L = {L}): {len(peaks)} resonances below the top")
res = []
for i in peaks:
    m = minimize_scalar(lambda E: -T(E), bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                        options={"xatol": 1e-13})
    # refine: the peak is where dT/dE = 0; brentq on a centered difference
    h = 1e-7
    dT = lambda E: (T(E + h) - T(E - h)) / (2 * h)
    Er = brentq(dT, m.x - 1e-5, m.x + 1e-5, xtol=1e-15)
    res.append(Er)
    left = [E for E in grid if E < Er and T(E) < 0.5]
    right = [E for E in grid if E > Er and T(E) < 0.5]
    lo = brentq(lambda E: T(E) - 0.5, left[-1], Er, xtol=1e-15)
    hi = brentq(lambda E: T(E) - 0.5, Er, right[0], xtol=1e-15)
    print(f"  E_res = {Er:.14f}   T = {T(Er):.15f}   FWHM = {hi - lo:.12e}   T_single(E_res) = {scatter_direct(Er, [(V0, b)])[0]:.6e}")
# the well between the barriers, if the barriers were infinite: E_n = (n pi / L)^2 / 2
for n in range(1, len(res) + 1):
    print(f"  infinite-well level n={n}: {(n * np.pi / L) ** 2 / 2:.6f}")
for i in range(len(res) - 1):
    mid = 0.5 * (res[i] + res[i + 1])
    print(f"T midway between resonances {i + 1} and {i + 2} (E = {mid:.12f}) = {T(mid):.12e}")
for E in (0.5, 1.0, 1.5, 2.5):
    print(f"double barrier T({E}) = {T(E):.15e}")

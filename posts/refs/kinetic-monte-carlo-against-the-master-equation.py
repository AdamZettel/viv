# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-kinetic-monte-carlo-against-the-master-equation.viv.

The master equation dp/dt = W p for the chain A <-> B <-> C is solved exactly
here by the matrix exponential, p(t) = expm(W t) p(0) (scipy.linalg.expm), in
place of the Runge-Kutta march the Viv program uses. The stationary state and
the mean first-passage time from A to C are solved as linear systems. A
separate Gillespie code (numpy's generator) checks that the KMC error bars the
post uses, sqrt(p (1 - p) / N), are the run-to-run spread.

    python3 posts/refs/kinetic-monte-carlo-against-the-master-equation.py > posts/refs/kinetic-monte-carlo-against-the-master-equation.out
"""
import numpy as np
from scipy.linalg import expm, null_space
from scipy import stats

np.set_printoptions(precision=17)
# K[i, j] = rate of the jump i -> j; states A = 0, B = 1, C = 2
K = np.array([[0.0, 2.0, 0.0],
              [0.5, 0.0, 1.0],
              [0.0, 0.25, 0.0]])
W = K.T - np.diag(K.sum(axis=1))
p0 = np.array([1.0, 0.0, 0.0])
print("generator W (dp/dt = W p):")
print(W)
print("eigenvalues of W:", np.sort(np.linalg.eigvals(W).real))

for t in (0.5, 1.0, 2.0, 5.0):
    p = expm(W * t) @ p0
    print(f"p({t}) = [{p[0]!r}, {p[1]!r}, {p[2]!r}]")

ns = null_space(W)[:, 0]
ns = ns / ns.sum()
print("stationary p =", ns.tolist(), " (1, 4, 16) / 21 =", [1 / 21, 4 / 21, 16 / 21])

# B's peak: where d p_B / dt = 0
from scipy.optimize import brentq
dpB = lambda t: (W @ (expm(W * t) @ p0))[1]
tpk = brentq(dpB, 0.1, 3.0, xtol=1e-14)
print(f"p_B peaks at t = {tpk!r}, p_B = {(expm(W * tpk) @ p0)[1]!r}")

# mean first-passage time A -> C: tau = -(W_T)^-1 1 on the transient states {A, B}
Q = K[:2, :2] - np.diag(K[:2].sum(axis=1))
tau = np.linalg.solve(-Q, np.ones(2))
print(f"mean first-passage time to C: from A {tau[0]!r}, from B {tau[1]!r}")
# its variance, for the standard error of a sample mean: E[T^2] = 2 (-Q)^-1 tau
m2 = 2 * np.linalg.solve(-Q, tau)
print(f"  standard deviation of the first-passage time from A: {np.sqrt(m2[0] - tau[0] ** 2)!r}")

# ---- the error bars: a numpy Gillespie run repeated -------------------------
N = 10000
REPS = 200
TSAMP = np.array([0.5, 1.0, 2.0])
rng = np.random.default_rng(7)
out = np.zeros((REPS, len(TSAMP), 3))
fpt = np.zeros(REPS)
for r in range(REPS):
    state = np.zeros(N, dtype=int)
    t = np.zeros(N)
    first = np.full(N, np.nan)
    rec = np.zeros((len(TSAMP), N), dtype=int)
    done = np.zeros((len(TSAMP), N), dtype=bool)
    alive = np.ones(N, dtype=bool)
    while alive.any():
        idx = np.nonzero(alive)[0]
        s = state[idx]
        rates = K[s]
        a = rates.sum(axis=1)
        tn = t[idx] + rng.exponential(1.0 / a)
        for k, ts in enumerate(TSAMP):
            hit = (~done[k, idx]) & (tn > ts)
            rec[k, idx[hit]] = s[hit]
            done[k, idx[hit]] = True
        u = rng.random(len(idx)) * a
        nxt = np.where(u < rates[:, 0], 0, np.where(u < rates[:, 0] + rates[:, 1], 1, 2))
        state[idx] = nxt
        t[idx] = tn
        newc = (nxt == 2) & np.isnan(first[idx])
        first[idx[newc]] = tn[newc]
        alive[idx] = ~(done[:, idx].all(axis=0) & ~np.isnan(first[idx]))
    for k in range(len(TSAMP)):
        out[r, k] = np.bincount(rec[k], minlength=3) / N
    fpt[r] = first.mean()
for k, ts in enumerate(TSAMP):
    p = expm(W * ts) @ p0
    print(f"t = {ts}: replica mean {out[:, k].mean(0).round(5).tolist()}, spread {out[:, k].std(0, ddof=1).round(5).tolist()}, "
          f"sqrt(p(1-p)/N) {np.sqrt(p * (1 - p) / N).round(5).tolist()}")
print(f"first-passage mean over {REPS} runs of {N}: {fpt.mean():.5f}, spread {fpt.std(ddof=1):.5f}")
print(f"P(|z| > 4) = {2 * stats.norm.sf(4):.3e}; for 75 comparisons, at most {75 * 2 * stats.norm.sf(4):.4f}")

# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-a-random-walk-becomes-diffusion.viv.

Two independent checks of the Viv program:

1. The same walk replayed in Python, draw for draw. Viv's rand() is Park-Miller
   MINSTD (lib/random.viv documents it), each step is A (2 u - 1), and the sums
   are taken in the same order, so the replay must give the Viv run's numbers.
2. The statistics of the walk: the exact moments of a sum of uniform steps,
   and the run-to-run spread of every measured quantity from 2000 independent
   replicas drawn with numpy's own generator (not MINSTD), which is what the
   post's tolerances rest on.

    python3 posts/refs/a-random-walk-becomes-diffusion.py > posts/refs/a-random-walk-becomes-diffusion.out
"""
import math
import numpy as np
from scipy import stats

WALKERS = 4000
NSTEP = 200
DT = 0.01
SIGMA = 0.1                      # per-step standard deviation
A = SIGMA * math.sqrt(3.0)       # uniform steps on [-A, A]
D = SIGMA ** 2 / (2 * DT)
SEED = 20260925.0
KURT_STEPS = [1, 2, 4, 8, 16, 32, 64, 128]
MSD_STEPS = [50, 100, 150, 200]
NBIN = 24
HALF = 3.0                       # histogram spans +- HALF standard deviations

print(f"D = sigma^2 / (2 dt) = {D}")

# ---- 1. replay of the Viv run --------------------------------------------
state = math.floor(abs(SEED)) % 2147483646.0 + 1.0


def rand():
    global state
    state = (16807.0 * state) % 2147483647.0
    return state / 2147483647.0


x = [0.0] * WALKERS
msd = []
g2 = []
for n in range(1, NSTEP + 1):
    s2 = 0.0
    s4 = 0.0
    for i in range(WALKERS):
        x[i] = x[i] + A * (2.0 * rand() - 1.0)
        s2 = s2 + x[i] * x[i]
        s4 = s4 + x[i] * x[i] * x[i] * x[i]
    m2 = s2 / WALKERS
    m4 = s4 / WALKERS
    msd.append(m2)
    g2.append(m4 / (m2 * m2) - 3.0)
print("replay: MSD at steps", MSD_STEPS, "=", [repr(msd[n - 1]) for n in MSD_STEPS])
print("replay: excess kurtosis at steps", KURT_STEPS, "=", [round(g2[n - 1], 6) for n in KURT_STEPS])
tt = np.arange(1, NSTEP + 1) * DT
D_fit = float(np.sum(tt * np.array(msd)) / (2 * np.sum(tt * tt)))
print(f"replay: least-squares D through the origin = {D_fit!r}")
sd_T = math.sqrt(2 * D * NSTEP * DT)
edges = np.linspace(-HALF * sd_T, HALF * sd_T, NBIN + 1)
counts, _ = np.histogram(np.array(x), bins=edges)
p = np.diff(stats.norm.cdf(edges / sd_T))
expect = WALKERS * p
chi2 = float(np.sum((counts - expect) ** 2 / expect))
print("replay: histogram counts =", counts.tolist())
print(f"replay: chi-square against the Gaussian = {chi2:.6f}")

# ---- 2. exact moments and replica spreads -----------------------------------
print()
print("exact: MSD = 2 D t; excess kurtosis of a sum of n uniform steps = -6/(5n)")
for n in MSD_STEPS:
    var_x2 = 2 * n * n * SIGMA ** 4 - 1.2 * n * SIGMA ** 4
    print(f"  n = {n}: 2Dt = {2 * D * n * DT:.6f}, standard error of the MSD = sqrt(Var x^2 / M) = {math.sqrt(var_x2 / WALKERS):.6e}")

REPS = 2000
rng = np.random.default_rng(12345)
msd_r = np.zeros((REPS, len(MSD_STEPS)))
g2_r = np.zeros((REPS, len(KURT_STEPS)))
dfit_r = np.zeros(REPS)
chi_r = np.zeros(REPS)
for r in range(REPS):
    steps = rng.uniform(-A, A, size=(NSTEP, WALKERS))
    xs = np.cumsum(steps, axis=0)
    m2 = np.mean(xs ** 2, axis=1)
    m4 = np.mean(xs ** 4, axis=1)
    msd_r[r] = m2[np.array(MSD_STEPS) - 1]
    g2_r[r] = (m4 / m2 ** 2 - 3.0)[np.array(KURT_STEPS) - 1]
    dfit_r[r] = np.sum(tt * m2) / (2 * np.sum(tt * tt))
    c, _ = np.histogram(xs[-1], bins=edges)
    chi_r[r] = np.sum((c - expect) ** 2 / expect)
print(f"replicas: {REPS} runs of {WALKERS} walkers, numpy PCG64")
print("  MSD mean / spread:", [f"{a:.5f}/{b:.3e}" for a, b in zip(msd_r.mean(0), msd_r.std(0, ddof=1))])
print(f"  D_fit mean {dfit_r.mean():.6f}, spread {dfit_r.std(ddof=1):.3e}")
for k, n in enumerate(KURT_STEPS):
    print(f"  kurtosis n = {n}: exact {-1.2 / n:+.6f}, replica mean {g2_r[:, k].mean():+.6f}, spread {g2_r[:, k].std(ddof=1):.4f}")
df = NBIN
bound = df + 3 * math.sqrt(2 * df)
print(f"  chi-square: replica mean {chi_r.mean():.3f} (df = {df}), fraction above {bound:.2f}: {np.mean(chi_r > bound):.4f}; "
      f"chi2({df}) tail above it: {stats.chi2.sf(bound, df):.4f}")

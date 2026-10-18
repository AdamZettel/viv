# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-metropolis-ising-against-onsager.viv.

Independent of the Viv program: the closed forms are evaluated in mpmath at
50 digits, and the finite-size value and the autocorrelation time come from
a separate Metropolis code (numba, its own random numbers, many chains).

    python3 posts/refs/metropolis-ising-against-onsager.py > posts/refs/metropolis-ising-against-onsager.out

Output recorded 2026-09-25 (numpy 1.24.4, numba 0.57.1, mpmath 1.3.0) is in
posts/refs/metropolis-ising-against-onsager.out.
"""
import mpmath as mp
import numpy as np
from numba import njit

mp.mp.dps = 50

# Onsager's critical temperature (J = k_B = 1): sinh(2/Tc) = 1
Tc = 2 / mp.log(1 + mp.sqrt(2))
print("Tc =", mp.nstr(Tc, 17))
print("sinh(2/Tc) - 1 =", mp.nstr(mp.sinh(2 / Tc) - 1, 3))


def onsager(T):
    """Yang's spontaneous magnetization (1952), zero above Tc."""
    sh = mp.sinh(2 / mp.mpf(T))
    if sh <= 1:
        return mp.mpf(0)
    return (1 - sh ** -4) ** mp.mpf("0.125")


for T in ["1.8", "2.0", "2.2"]:
    print(f"M({T}) =", mp.nstr(onsager(T), 17))

# ---- a second Metropolis code: L = 16, T = 1.8, sequential sweeps like the
# post's, 64 independent chains, 20000 measured sweeps each.
L = 16
T = 1.8
CHAINS = 64
EQUIL = 2000
MEAS = 20000


@njit(cache=True)
def run(L, beta, equil, meas, seed):
    np.random.seed(seed)
    s = np.ones((L, L))
    w4 = np.exp(-4 * beta)
    w8 = np.exp(-8 * beta)
    out = np.empty(meas)
    for sweep in range(equil + meas):
        for i in range(L):
            for j in range(L):
                nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
                dE = 2 * s[i, j] * nb
                if dE <= 0 or (dE == 4 and np.random.random() < w4) or (dE == 8 and np.random.random() < w8):
                    s[i, j] = -s[i, j]
        if sweep >= equil:
            out[sweep - equil] = abs(s.sum()) / (L * L)
    return out


series = np.array([run(L, 1 / T, EQUIL, MEAS, 1000 + c) for c in range(CHAINS)])
chain_means = series.mean(axis=1)
m16 = chain_means.mean()
m16_err = chain_means.std(ddof=1) / np.sqrt(CHAINS)
print(f"{CHAINS} chains of {MEAS} measured sweeps (after {EQUIL}) at L = {L}, {L * L} spins")
print(f"<|m|> at L=16, T=1.8: {m16:.6f} +- {m16_err:.6f}")
print(f"finite-size offset <|m|>_16 - M_inf: {m16 - float(onsager('1.8')):+.6f}")

# integrated autocorrelation time of |m| (in sweeps), summed until the
# autocorrelation first drops below zero
x = series - series.mean(axis=1, keepdims=True)
var = (x * x).mean()
tau = 0.5
for lag in range(1, 200):
    rho = (x[:, :-lag] * x[:, lag:]).mean() / var
    if rho < 0:
        break
    tau += rho
print(f"integrated autocorrelation time: {tau:.2f} sweeps")
print(f"single-sweep standard deviation of |m|: {np.sqrt(var):.5f}")
# expected standard error of the post's run: 1200 measured sweeps
print(f"expected sigma of a 1200-sweep mean: {np.sqrt(var * 2 * tau / 1200):.5f}")


# ---- the post's sweep over temperature: from all up, 100 sweeps to settle,
# then the mean |m| over 200 sweeps. The same protocol, 256 chains at each
# temperature: the mean and the chain-to-chain standard deviation of that
# 200-sweep average, which is what one run of the post scatters by.
T_SWEEP = [1.6, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3.2]
SWEEP_CHAINS = 256
print(f"T, mean of the 200-sweep |m| over {SWEEP_CHAINS} chains, its chain-to-chain std, Onsager:")
for Ts in T_SWEEP:
    avgs = np.array([run(L, 1 / Ts, 100, 200, 5000 + c).mean() for c in range(SWEEP_CHAINS)])
    print(f"  {Ts:.1f}  {avgs.mean():.4f}  {avgs.std(ddof=1):.4f}  {float(onsager(str(Ts))):.4f}")

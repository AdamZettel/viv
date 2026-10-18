# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference values for posts/2026-09-25-two-stream-instability-in-a-particle-code.viv.

1. Linear theory: two cold beams at +-v0, each of density n0/2, omega_p = 1.
   The dispersion relation 1 = (1/2)[1/(w - k v0)^2 + 1/(w + k v0)^2] is the
   quartic w^4 - (2a^2 + 1) w^2 + a^4 - a^2 = 0 with a = k v0; numpy.roots
   solves it and the largest imaginary part is the growth rate.
2. The same PIC algorithm as the post (quiet start, CIC deposit, trapezoid
   Gauss's law, CIC gather, leapfrog), written in numpy, at the browser
   settings and at refined ones, to show how far the measured rate sits from
   linear theory and why.

Run: python3 posts/refs/two_stream_pic.py   (numpy 1.26)
"""
import numpy as np

K = np.sqrt(3.0 / 8.0)


def theory(a):
    r = np.roots([1.0, 0.0, -(2 * a * a + 1.0), 0.0, a**4 - a * a])
    return max(r.imag)


def pic(npart, ng, dt, nstep, amp=1e-3, t0=8.0, t1=18.0):
    L = 2 * np.pi / K
    dx = L / ng
    half = npart // 2
    x0 = (np.arange(half) + 0.5) * L / half
    x = np.concatenate([x0 + amp * np.sin(K * x0), x0 - amp * np.sin(K * x0)]) % L
    v = np.concatenate([np.ones(half), -np.ones(half)])
    nodes = np.arange(ng) * dx
    phase = np.exp(-1j * K * nodes)

    def field(x):
        s = x / dx
        j = np.floor(s).astype(int)
        f = s - j
        rho = np.ones(ng)
        np.add.at(rho, j % ng, -(ng / npart) * (1 - f))
        np.add.at(rho, (j + 1) % ng, -(ng / npart) * f)
        E = np.zeros(ng)
        for i in range(1, ng):
            E[i] = E[i - 1] + 0.5 * (rho[i - 1] + rho[i]) * dx
        E -= E.mean()
        return E

    def gather(x, E):
        s = x / dx
        j = np.floor(s).astype(int)
        f = s - j
        return E[j % ng] * (1 - f) + E[(j + 1) % ng] * f

    E = field(x)
    v = v + 0.5 * dt * gather(x, E)  # v at -dt/2 -> the leapfrog stagger
    ts, amps, energy = [], [], []
    for n in range(nstep):
        E = field(x)
        ts.append(n * dt)
        amps.append(2.0 / ng * abs(np.sum(E * phase)))
        energy.append(0.5 * np.sum(E * E) * dx)
        v = v - dt * gather(x, E)
        x = (x + dt * v) % L
    ts, amps = np.array(ts), np.array(amps)
    m = (ts >= t0) & (ts <= t1)
    g = np.polyfit(ts[m], np.log(amps[m]), 1)[0]
    return g, ts, amps, np.array(energy)


gth = theory(K)
print("cold two-beam growth rate at k v0 = sqrt(3/8):", repr(gth), " 1/(2 sqrt 2) =", repr(1 / (2 * np.sqrt(2))))
print("growth rate at a = 0.48:", repr(theory(0.48)))
print("growth rate at a = 1.2 (stable):", repr(theory(1.2)))

g, ts, amps, en = pic(2000, 32, 0.1, 250)
print("browser settings (2000 particles, 32 cells, dt 0.1): gamma =", repr(g), " rel err", (g - gth) / gth)
i = np.argmax(amps)
print("  peak mode amplitude", repr(amps[i]), "at t =", ts[i])
print("  field energy at t=0 and peak:", en[0], en.max(), "ratio", en.max() / en[0])
print("  amplitude at t = 8, 18, 24.9:", repr(amps[80]), repr(amps[180]), repr(amps[-1]))
for label, args in [("dt 0.05", (2000, 32, 0.05, 500)),
                    ("64 cells", (2000, 64, 0.1, 250)),
                    ("8000 particles", (8000, 32, 0.1, 250)),
                    ("64 cells, dt 0.05, 8000 particles", (8000, 64, 0.05, 500))]:
    g2 = pic(*args)[0]
    print(f"  {label:36s} gamma = {g2:.6f}  rel err {(g2 - gth) / gth:+.2e}")
for (t0, t1) in [(4, 12), (6, 16), (8, 18)]:
    g3 = pic(2000, 32, 0.1, 250, t0=t0, t1=t1)[0]
    print(f"  window [{t0}, {t1}]: gamma = {g3:.6f}  rel err {(g3 - gth) / gth:+.2e}")

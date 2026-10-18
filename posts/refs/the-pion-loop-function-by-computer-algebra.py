# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-the-pion-loop-function-by-computer-algebra.viv.

The post builds L(q) = (w/q) ln((w + q)/(2 m_pi)), w = sqrt(4 m_pi^2 + q^2)
(Machleidt and Entem, Phys. Rep. 503, 1 (2011), arXiv:1105.2919, Eqs. (4.11)
and (4.12); m_pi = 138.039 MeV, their average pion mass) as an expression
tape, evaluates it and differentiates it symbolically.

This script never uses that closed form for its reference values. It
evaluates the loop as a Feynman-parameter integral,

    L(q) = 1 + (1/2) int_0^1 ln(1 + x (1 - x) q^2 / m_pi^2) dx,

with mpmath's tanh-sinh quadrature at 30 digits, and dL/dq by mpmath.diff of
that integral. The first lines print the closed form beside it once, to show
the two agree.

    python3 posts/refs/the-pion-loop-function-by-computer-algebra.py > posts/refs/the-pion-loop-function-by-computer-algebra.out
"""
import mpmath as mp

mp.mp.dps = 30
MPI = mp.mpf("138.039")


def L_feyn(q):
    q = mp.mpf(q)
    return 1 + mp.quad(lambda x: mp.log(1 + x * (1 - x) * q * q / (MPI * MPI)), [0, 1]) / 2


def L_closed(q):
    q = mp.mpf(q)
    w = mp.sqrt(4 * MPI * MPI + q * q)
    return w / q * mp.log((w + q) / (2 * MPI))


print(f"mpmath {mp.__version__}, {mp.mp.dps} digits")
for q in [50, 200, 500]:
    print(f"q = {q}: Feynman integral - closed form = {mp.nstr(L_feyn(q) - L_closed(q), 3)}")
print("q (MeV), L(q) from the Feynman-parameter integral")
for q in [1, 50, 100, 200, 300, 500, 700, 1000]:
    print(f"{q:5d}  {mp.nstr(L_feyn(q), 20)}")
print("q (MeV), dL/dq (1/MeV) by mpmath.diff of the Feynman-parameter integral")
for q in [50, 200, 500]:
    print(f"{q:5d}  {mp.nstr(mp.diff(L_feyn, q), 20)}")
print(f"small q: 1 + q^2/(12 m^2) at q = 1: {mp.nstr(1 + mp.mpf(1)/(12*MPI*MPI), 20)}")

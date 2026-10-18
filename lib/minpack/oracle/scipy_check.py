# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The second reference: scipy.optimize.leastsq on the same test problems.

scipy's leastsq wraps MINPACK's lmder (compiled from the same Fortran, with
whatever flags scipy's build used). The problems below are written again here
in numpy from Moré, Garbow and Hillstrom (1981) and the netlib driver
ref/file17.f, independently of the Fortran and of the Viv port. Each case runs
with lmder1's settings: ftol = xtol = sqrt(dpmpar(1)), gtol = 0,
maxfev = 100 (n + 1), factor = 100, mode 1.

    python3 lib/minpack/oracle/scipy_check.py <oracle plain.txt>

prints one line per case: scipy's info, nfev, njev and final norm next to the
Fortran oracle's, and the largest relative difference in x. It exits 1 if any
case differs in info, nfev or njev, or in x or the norm by more than 1e-8
relative (the problems are solved only to xtol = 1.5e-8, and scipy's build is
free to fuse multiply-adds).

Six cases are exempt from the count and x comparison, because their path
depends on rounding: Powell singular from all three starts (a singular
Jacobian at the solution; the residual goes to 1e-34 along a path that last
bits change), Kowalik-Osborne from 100 times the start (stops at maxfev) and
Meyer from 1 and 10 times the start (badly scaled; stops on the xtol test at
the edge of the tolerance, or at maxfev). A gfortran -O2 build of the same
Fortran, which fuses multiply-adds, also takes a different path on five of
them (all but Meyer from the standard start). For them the check asks only that both stop the same way:
at maxfev in both, or at the same minimum (norms within 1e-6 relative, or
both below 1e-30).
"""
SENSITIVE = {(6, 1.0), (6, 10.0), (6, 100.0), (9, 100.0), (10, 1.0), (10, 10.0)}
import sys
import numpy as np
from scipy.optimize import leastsq

TOL = np.sqrt(2.22044604926e-16)  # sqrt(dpmpar(1)), dpmpar's IEEE value

Y_BARD = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58,
                   0.73, 0.96, 1.34, 2.10, 4.39])
V_KO = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625])
Y_KO = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342,
                 0.0323, 0.0235, 0.0246])
Y_MEYER = np.array([34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0,
                    9744.0, 8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0])
Y_OS1 = np.array([0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.85, 0.818, 0.784,
                  0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.58, 0.558, 0.538, 0.522,
                  0.506, 0.49, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.42,
                  0.414, 0.411, 0.406])
Y_OS2 = np.array([1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725,
                  0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724,
                  0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495,
                  0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
                  0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632,
                  0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581,
                  0.428, 0.292, 0.162, 0.098, 0.054])


def bard_t():
    i = np.arange(1, 16, dtype=float)
    u, v = i, 16.0 - i
    w = np.where(i > 8, v, u)
    return u, v, w


def f_and_j(nprob, m):
    """(f, jac, x0) for problem nprob with m residuals."""
    if nprob == 4:  # Rosenbrock
        f = lambda x: np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
        j = lambda x: np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])
        return f, j, [-1.2, 1.0]
    if nprob == 5:  # helical valley
        def f(x):
            th = np.arctan2(x[1], x[0]) / (2 * np.pi)  # atan(x2/x1)/2pi, +1/2 when x1 < 0
            if x[0] > 0:
                th = np.arctan(x[1] / x[0]) / (2 * np.pi)
            elif x[0] < 0:
                th = np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5
            else:
                th = np.copysign(0.25, x[1])
            r = np.hypot(x[0], x[1])
            return np.array([10.0 * (x[2] - 10.0 * th), 10.0 * (r - 1.0), x[2]])

        def j(x):
            t = x[0] ** 2 + x[1] ** 2
            return np.array([[100.0 * x[1] / (2 * np.pi * t), -100.0 * x[0] / (2 * np.pi * t), 10.0],
                             [10.0 * x[0] / np.sqrt(t), 10.0 * x[1] / np.sqrt(t), 0.0],
                             [0.0, 0.0, 1.0]])
        return f, j, [-1.0, 0.0, 0.0]
    if nprob == 6:  # Powell singular
        f = lambda x: np.array([x[0] + 10 * x[1], np.sqrt(5.0) * (x[2] - x[3]),
                                (x[1] - 2 * x[2]) ** 2, np.sqrt(10.0) * (x[0] - x[3]) ** 2])

        def j(x):
            J = np.zeros((4, 4))
            J[0, 0], J[0, 1] = 1.0, 10.0
            J[1, 2], J[1, 3] = np.sqrt(5.0), -np.sqrt(5.0)
            J[2, 1] = 2 * (x[1] - 2 * x[2])
            J[2, 2] = -2 * J[2, 1]
            J[3, 0] = 2 * np.sqrt(10.0) * (x[0] - x[3])
            J[3, 3] = -J[3, 0]
            return J
        return f, j, [3.0, -1.0, 0.0, 1.0]
    if nprob == 7:  # Freudenstein and Roth
        f = lambda x: np.array([-13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1],
                                -29 + x[0] + ((1 + x[1]) * x[1] - 14) * x[1]])
        j = lambda x: np.array([[1.0, x[1] * (10 - 3 * x[1]) - 2], [1.0, x[1] * (2 + 3 * x[1]) - 14]])
        return f, j, [0.5, -2.0]
    if nprob == 8:  # Bard
        u, v, w = bard_t()
        f = lambda x: Y_BARD - (x[0] + u / (x[1] * v + x[2] * w))

        def j(x):
            d = (x[1] * v + x[2] * w) ** 2
            return np.column_stack([-np.ones(15), u * v / d, u * w / d])
        return f, j, [1.0, 1.0, 1.0]
    if nprob == 9:  # Kowalik and Osborne
        def f(x):
            return Y_KO - x[0] * V_KO * (V_KO + x[1]) / (V_KO * (V_KO + x[2]) + x[3])

        def j(x):
            t1 = V_KO * (V_KO + x[1])
            t2 = V_KO * (V_KO + x[2]) + x[3]
            j1 = -t1 / t2
            j2 = -V_KO * x[0] / t2
            j3 = j1 * j2
            return np.column_stack([j1, j2, j3, j3 / V_KO])
        return f, j, [0.25, 0.39, 0.415, 0.39]
    if nprob == 10:  # Meyer
        t = 5.0 * np.arange(1, 17) + 45.0

        def f(x):
            return x[0] * np.exp(x[1] / (t + x[2])) - Y_MEYER

        def j(x):
            d = t + x[2]
            e = np.exp(x[1] / d)
            return np.column_stack([e, x[0] * e / d, -(x[1] / d) * x[0] * e / d])
        return f, j, [0.02, 4000.0, 250.0]
    if nprob == 12:  # Box 3-D
        i = np.arange(1, m + 1, dtype=float)
        t = i / 10.0
        f = lambda x: np.exp(-t * x[0]) - np.exp(-t * x[1]) + (np.exp(-i) - np.exp(-t)) * x[2]
        j = lambda x: np.column_stack([-t * np.exp(-t * x[0]), t * np.exp(-t * x[1]), np.exp(-i) - np.exp(-t)])
        return f, j, [0.0, 10.0, 20.0]
    if nprob == 13:  # Jennrich and Sampson
        i = np.arange(1, m + 1, dtype=float)
        f = lambda x: 2 + 2 * i - (np.exp(i * x[0]) + np.exp(i * x[1]))
        j = lambda x: np.column_stack([-i * np.exp(i * x[0]), -i * np.exp(i * x[1])])
        return f, j, [0.3, 0.4]
    if nprob == 17:  # Osborne 1
        t = 10.0 * np.arange(33)
        f = lambda x: Y_OS1 - (x[0] + x[1] * np.exp(-x[3] * t) + x[2] * np.exp(-x[4] * t))
        j = lambda x: np.column_stack([-np.ones(33), -np.exp(-x[3] * t), -np.exp(-x[4] * t),
                                       t * x[1] * np.exp(-x[3] * t), t * x[2] * np.exp(-x[4] * t)])
        return f, j, [0.5, 1.5, -1.0, 0.01, 0.02]
    if nprob == 18:  # Osborne 2
        t = np.arange(65) / 10.0

        def f(x):
            return Y_OS2 - (x[0] * np.exp(-t * x[4]) + x[1] * np.exp(-(t - x[8]) ** 2 * x[5])
                            + x[2] * np.exp(-(t - x[9]) ** 2 * x[6]) + x[3] * np.exp(-(t - x[10]) ** 2 * x[7]))

        def j(x):
            e1 = np.exp(-x[4] * t)
            e2 = np.exp(-x[5] * (t - x[8]) ** 2)
            e3 = np.exp(-x[6] * (t - x[9]) ** 2)
            e4 = np.exp(-x[7] * (t - x[10]) ** 2)
            return np.column_stack([-e1, -e2, -e3, -e4, t * x[0] * e1,
                                    x[1] * (t - x[8]) ** 2 * e2, x[2] * (t - x[9]) ** 2 * e3,
                                    x[3] * (t - x[10]) ** 2 * e4,
                                    -2 * x[1] * x[5] * (t - x[8]) * e2,
                                    -2 * x[2] * x[6] * (t - x[9]) * e3,
                                    -2 * x[3] * x[7] * (t - x[10]) * e4])
        return f, j, [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5]
    return None


def oracle_cases(path):
    cases = []
    for line in open(path):
        w = line.split()
        if w[0] == 'C':
            cases.append({'nprob': int(w[1]), 'n': int(w[2]), 'm': int(w[3]), 'factor': float(w[4])})
        elif w[0] == 'F':
            c = cases[-1]
            c['info'], c['nfev'], c['njev'] = int(w[1]), int(w[2]), int(w[3])
            c['fnorm'] = float(w[4])
            c['x'] = np.array([float(v) for v in w[5:]])
    return cases


def main():
    bad = 0
    print(f"{'prob':>4} {'n':>3} {'m':>3} {'factor':>6}  {'info':>9} {'nfev':>9} {'njev':>9}  {'fnorm (scipy)':>22} {'fnorm (Fortran)':>22}  {'max rel dx':>10}")
    for c in oracle_cases(sys.argv[1]):
        fj = f_and_j(c['nprob'], c['m'])
        if fj is None:
            continue
        f, j, x0 = fj
        x0 = c['factor'] * np.array(x0, dtype=float)
        n = len(x0)
        x, cov, info, msg, ier = leastsq(f, x0, Dfun=j, full_output=True, ftol=TOL, xtol=TOL,
                                         gtol=0.0, maxfev=100 * (n + 1), factor=100.0)
        ier = 4 if ier == 8 else ier  # lmder1 reports info 8 as 4
        fn = float(np.linalg.norm(f(x)))
        scale = np.maximum(np.abs(c['x']), 1e-300)
        dx = float(np.max(np.abs(x - c['x']) / scale))
        dfn = abs(fn - c['fnorm']) / max(c['fnorm'], 1e-300)
        tiny = c['fnorm'] < 1e-10  # a zero-residual problem: compare absolutely
        sens = (c['nprob'], c['factor']) in SENSITIVE
        if sens:
            if c['info'] == 5:
                ok = ier == 5
            else:
                ok = dfn <= 1e-6 or (fn < 1e-30 and c['fnorm'] < 1e-30)
        else:
            ok = (ier == c['info'] and info['nfev'] == c['nfev'] and info['njev'] == c['njev']
                  and (dx <= 1e-8 or tiny) and (dfn <= 1e-8 or (tiny and fn < 1e-10)))
        bad += not ok
        print(f"{c['nprob']:>4} {n:>3} {c['m']:>3} {c['factor']:>6g}  {ier:>4}/{c['info']:<4} {info['nfev']:>4}/{c['nfev']:<4} "
              f"{info['njev']:>4}/{c['njev']:<4}  {fn:>22.15e} {c['fnorm']:>22.15e}  {dx:>10.2e}{'  rounding-sensitive' if sens and ok else ''}{'' if ok else '  DIFFERS'}")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

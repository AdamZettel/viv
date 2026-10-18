# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-the-free-neutron-star-limit.viv.

A twin written the other way round from the post. The post integrates the
TOV equations in geometric units with the log-enthalpy h = ln(mu / m_n c^2)
as the independent variable. This one integrates them in cgs units with the
radius as the independent variable and the Fermi momentum x = p_F / (m_n c)
as the unknown (dx/dr = (dP/dr) / (dP/dx), with dP/dx in closed form), with
scipy's solve_ivp (DOP853, rtol 1e-12) stopped by an event when x falls to
1e-4 of its central value (the pressure is then 1e-20 of the central one). The maximum over the central Fermi momentum
is found by minimize_scalar (bounded, xatol 1e-7).

Constants: scipy.constants (CODATA 2018) for G, c, hbar and the neutron
mass; the solar mass parameter GM_sun = 1.3271244e20 m^3/s^2 (IAU 2015
Resolution B3, nominal value).

Literature: Narain, Schaffner-Bielich and Mishustin, "Compact stars made of
fermionic dark matter", Phys. Rev. D 74, 063003 (2006), arXiv:astro-ph/0605724,
Eqs. (31)-(32) and the text after them: a star of free neutrons
(m_f = 939.6 MeV) has M_max = 0.710 M_sun at R = 9.192 km, "which match very
well the original results of Oppenheimer and Volkoff" (Phys. Rev. 55, 374, 1939).

    python3 posts/refs/the-free-neutron-star-limit.py > posts/refs/the-free-neutron-star-limit.out

Output recorded 2026-09-25 (scipy 1.12.0) is in
posts/refs/the-free-neutron-star-limit.out.
"""
import numpy as np
from scipy import constants as C
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

G = C.G * 1e3                  # cm^3 g^-1 s^-2
c = C.c * 1e2                  # cm/s
hbar = C.hbar * 1e7            # erg s
m_n = C.m_n * 1e3              # g
GM_SUN = 1.3271244e20 * 1e6    # cm^3/s^2
M_SUN = GM_SUN / G             # g

eps0 = m_n ** 4 * c ** 5 / (8 * np.pi ** 2 * hbar ** 3)   # erg/cm^3
print(f"eps0 = m_n^4 c^5 / (8 pi^2 hbar^3) = {eps0:.10e} erg/cm^3")
print(f"G M_sun / c^2 = {GM_SUN / c**2 / 1e5:.10f} km")


def eps_of_x(x):
    return eps0 * ((2 * x ** 3 + x) * np.sqrt(1 + x * x) - np.arcsinh(x))


def P_of_x(x):
    return eps0 / 3 * ((2 * x ** 3 - 3 * x) * np.sqrt(1 + x * x) + 3 * np.arcsinh(x))


def dP_dx(x):
    return eps0 * 8 / 3 * x ** 4 / np.sqrt(1 + x * x)


# dP/dx in closed form, checked against a centered difference of P(x)
for xt in [0.3, 1.0, 3.0]:
    hx = 1e-5 * xt
    fd = (P_of_x(xt + hx) - P_of_x(xt - hx)) / (2 * hx)
    print(f"dP/dx at x = {xt}: closed form / finite difference - 1 = {dP_dx(xt) / fd - 1:.1e}")


def rhs(r, u):
    m, x = u
    if x <= 0:
        return [0.0, 0.0]
    P = P_of_x(x)
    rho = eps_of_x(x) / c ** 2
    dm = 4 * np.pi * r * r * rho
    dP = -G * (rho + P / c ** 2) * (m + 4 * np.pi * r ** 3 * P / c ** 2) / (r * (r - 2 * G * m / c ** 2))
    return [dm, dP / dP_dx(x)]


RTOL = 1e-12
print(f"solve_ivp: DOP853, relative tolerance {RTOL:.6e}")


def star(xc):
    r0 = 1.0  # cm
    m0 = 4 / 3 * np.pi * r0 ** 3 * eps_of_x(xc) / c ** 2
    surface = lambda r, u: u[1] - 1e-4 * xc
    surface.terminal = True
    sol = solve_ivp(rhs, (r0, 1e8), [m0, xc], method="DOP853", rtol=RTOL, atol=[1e-30, 1e-16],
                    events=surface)
    return sol.t_events[0][0] / 1e5, sol.y_events[0][0][0] / M_SUN


for xc in [0.3, 0.5, 0.7, 1.0, 2.0, 4.0]:
    R, M = star(xc)
    print(f"x_c = {xc}: R = {R:.6f} km, M = {M:.8f} M_sun")

best = minimize_scalar(lambda lx: -star(np.exp(lx))[1], bounds=(np.log(0.3), np.log(3.0)), method="bounded",
                       options={"xatol": 1e-7})
xc_max = np.exp(best.x)
R_max, M_max = star(xc_max)
print(f"maximum mass: M = {M_max:.8f} M_sun at x_c = {xc_max:.6f}, R = {R_max:.6f} km")
print(f"central density there: {eps_of_x(xc_max) / c**2:.6e} g/cm^3")

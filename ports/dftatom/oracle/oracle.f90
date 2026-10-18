! SPDX-License-Identifier: MIT
! SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
! SPDX-FileCopyrightText: 2026 The Viv authors
! The upstream oracle: one self-consistent atom with upstream dftatom, printed
! in full precision, for ports/dftatom/oracle/make_reference.py.
!
! The body is upstream's drivers.f90 atom_lda / atom_rlda (Copyright (c)
! 2006-2012 Ondrej Certik, Jiri Vackar, John Pask; MIT), repeated here only so
! that the energy components, the iteration count and the density, which
! atom_lda keeps in its local dft_data_t, can be printed. It links upstream's
! own libdftatom.a, unchanged.
!
!   oracle Z rel N r_min r_max a c reigen_eps mixing_eps alpha max_iter [reps [perturb]]
program oracle
use types, only: dp
use constants, only: pi
use states, only: get_atomic_states_nonrel, get_atomic_states_rel
use dft, only: KS_step
use mesh, only: mesh_exp, mesh_exp_deriv
use energies, only: get_hydrogen_energies, thomas_fermi_potential, E_nl
use dft_data, only: dft_data_t
use mixings, only: mixing_anderson
implicit none

integer :: Z, rel, N, max_iter, i, j, relat, n_orb, reps, iperturb
real(dp) :: r_min, r_max, a, c, reigen_eps, mixing_eps, alpha
character(len=64) :: arg
integer, pointer :: no(:), lo(:), so(:)
real(dp), pointer :: fo(:)
real(dp), allocatable, target :: R(:), Rp(:), V_tot(:), density(:), V_h(:), &
    V_xc(:), e_xc(:), V_coulomb(:), tmp(:), orbitals(:, :), ks(:), Emin(:), Emax(:)
type(dft_data_t) :: d
integer(8) :: t0, t1, rate

call get_command_argument(1, arg); read(arg, *) Z
call get_command_argument(2, arg); read(arg, *) rel
call get_command_argument(3, arg); read(arg, *) N
call get_command_argument(4, arg); read(arg, *) r_min
call get_command_argument(5, arg); read(arg, *) r_max
call get_command_argument(6, arg); read(arg, *) a
call get_command_argument(7, arg); read(arg, *) c
call get_command_argument(8, arg); read(arg, *) reigen_eps
call get_command_argument(9, arg); read(arg, *) mixing_eps
call get_command_argument(10, arg); read(arg, *) alpha
call get_command_argument(11, arg); read(arg, *) max_iter
reps = 1
iperturb = 1
if (command_argument_count() >= 12) then
    call get_command_argument(12, arg); read(arg, *) reps
end if
if (command_argument_count() >= 13) then
    call get_command_argument(13, arg); read(arg, *) iperturb
end if

if (rel == 1) then
    call get_atomic_states_rel(Z, no, lo, so, fo)
else
    call get_atomic_states_nonrel(Z, no, lo, fo)
    allocate(so(size(no)))
    so = 1
end if
n_orb = size(no)
allocate(R(N+1), Rp(N+1), V_tot(N+1), density(N+1), V_h(N+1), V_xc(N+1), &
    e_xc(N+1), V_coulomb(N+1), tmp(N+1), orbitals(N+1, n_orb), ks(n_orb), &
    Emin(n_orb), Emax(n_orb))

call system_clock(t0, rate)
do j = 1, reps
    call solve()
end do
call system_clock(t1)

print '(a, es25.17)', "seconds ", real(t1 - t0, dp) / rate / reps
print '(a, i0)', "iterations ", d%scf_iter
print '(a, es25.17)', "etot ", d%Etot
print '(a, es25.17)', "ekin ", d%Ekin
print '(a, es25.17)', "ecoul ", d%Ecoul
print '(a, es25.17)', "eenuc ", d%Eenuc
print '(a, es25.17)', "exc ", d%Exc
do i = 1, n_orb
    print '(a, 3i4, es25.17, es25.17)', "orbital ", no(i), lo(i), so(i), fo(i), ks(i)
end do
! the density at every 500th point, and the last, for a pointwise check
do i = 1, N+1, 500
    print '(a, i0, es25.17, es25.17)', "density ", i-1, R(i), density(i)
end do
print '(a, es25.17)', "charge ", 4*pi*sum(density*R**2*Rp)

contains

subroutine solve()
R = mesh_exp(r_min, r_max, a, N)
Rp = mesh_exp_deriv(r_min, r_max, a, N)
ks = get_hydrogen_energies(Z, no)
V_tot = thomas_fermi_potential(R, Z)
V_coulomb = -Z/R
Emax = 10
do i = 1, n_orb
    if (rel == 1) then
        if (so(i) == 1) then
            relat = 2
        else
            relat = 3
        end if
    else
        relat = 0
    end if
    Emin(i) = E_nl(c, no(i), lo(i), Z, relat)
end do
Emin = 1.1_dp * Emin
d%Z = Z
d%R => R
d%Rp => Rp
d%rho => density
d%V_h => V_h
d%V_coulomb => V_coulomb
d%V_xc => V_xc
d%e_xc => e_xc
d%V_tot => V_tot
d%orbitals => orbitals
d%alpha = alpha
d%dirac = (rel == 1)
d%perturb = (iperturb == 1)
d%reigen_eps = reigen_eps
d%reigen_max_iter = 100
d%no => no
d%lo => lo
d%fo => fo
d%ks_energies => ks
d%Emax_init => Emax
d%Emin_init => Emin
d%c = c
d%so => so
tmp = mixing_anderson(KS_step, d%V_tot - d%V_coulomb, max_iter, &
    .true., d, alpha, mixing_eps)
end subroutine

end program

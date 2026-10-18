# Skyrme Hartree-Fock of closed-shell nuclei: the checks

`lib/skyrme/spherical.viv`, run through the exported program
(`lib/skyrme/skyrmehf.export.toml`), against an independent numpy solver,
against skyrme_rpa, against experiment, and against itself in three engines.
Every number below comes from a run on an Apple M-series machine on
2026-10-12:

```sh
viv export lib/skyrme/skyrmehf.export.toml -o target/export/skyrmehf
python3 lib/skyrme/refs/compare.py --json out.json        # the two solvers, 56 runs, 4.6 min
python3 tools/export/check.py lib/skyrme/skyrmehf.export.toml   # three engines, 7 cases
python3 tools/export/test_skyrmehf.py                    # oxygen-16 against the post and skyrme_rpa
```

## 1. Against the numpy solver

The reference is `lib/skyrme/refs/skyrme_hf.py`, a copy of the personal
site's solver (scripts/projects/skyrme-hf-vb/skyrme_hf.py). It writes the same
functional with its own code: dense `numpy.linalg.eigh` for each (l, j)
block instead of bisection and inverse iteration, shells filled in a
spectroscopic order it is given instead of by energy, and SIII's $t_3$ term
as the genuine three-body $\frac14 t_3\rho_n\rho_p\rho$ instead of the
density-dependent form with $x_3 = 1$, $\alpha = 1$. `compare.py` sets its
constants to Viv's (skyrme_rpa's $\hbar c$, nucleon mass and $e^2$; the
factor $1 - 1/A$ on $\hbar^2/2m$ in both the equation and the energy) and
runs both at the same mesh, box (Viv's default), linear mixing of 0.5 (the
numpy solver's; Viv's default is Pulay's mixing since 2026-10-14) and tolerance
$10^{-10}$ MeV.

All 56 runs (7 nuclei, 4 forces, spacings 0.1 and 0.05 fm):

| quantity | largest difference | where |
|---|---|---|
| E/A | $3.9 \times 10^{-14}$ MeV | SkM*, oxygen-16, 0.05 fm |
| $r_p$, $r_n$, $r_{ch}$ | $9.0 \times 10^{-13}$ fm | SkM*, calcium-48, 0.05 fm |
| occupied levels | $1.9 \times 10^{-11}$ MeV | SkM*, calcium-48, 0.05 fm |
| passes | equal in every run | 39 to 74 |
| occupied orbitals | the same set in every run | |

The tolerance in the cases file (`skyrmehf.cases.json`, through the
generated test) is $10^{-9}$, fifty times the largest difference seen. The
differences are the two eigensolvers' round-off: both iterations stop at
the same pass, so neither stopping point nor the filling differs. Seven of
the 28 pairs needed one refill (an orbital changed places after the first
pass: SIII for calcium-48, zirconium-90, tin-132 and lead-208; SkM* for
calcium-48, tin-132 and lead-208), and each ended with the reference's shells.

## 2. Per nucleus and force

E/A at 0.1 fm; E/A, $r_{ch}$ and the neutron skin $r_n - r_p$ extrapolated to
$h \to 0$ as $(4X(0.05) - X(0.1))/3$; the highest occupied neutron and
proton orbitals (extrapolated, MeV). Measured values in parentheses: binding
energies from AME2020 (Wang et al., Chinese Phys. C 45, 030003 (2021)) and
charge radii from Angeli and Marinova, At. Data Nucl. Data Tables 99, 69
(2013) (nickel-56 has none there).

| force | nucleus | E/A, 0.1 fm | E/A, extrapolated | $r_{ch}$ (fm) | $r_n - r_p$ (fm) | last n / last p (MeV) |
|---|---|---|---|---|---|---|
| SLy4 | 16O | −8.0526 | −8.0311 (−7.976) | 2.8017 (2.699) | −0.025 | 1p1/2 −14.536 / 1p1/2 −11.187 |
| SLy4 | 40Ca | −8.6293 | −8.6067 (−8.551) | 3.5112 (3.478) | −0.048 | 1d3/2 −15.309 / 1d3/2 −8.362 |
| SLy4 | 48Ca | −8.7299 | −8.7065 (−8.666) | 3.5435 (3.477) | 0.154 | 1f7/2 −9.793 / 1d3/2 −16.044 |
| SLy4 | 56Ni | −8.6550 | −8.6316 (−8.643) | 3.7865 | −0.055 | 1f7/2 −16.178 / 1f7/2 −7.061 |
| SLy4 | 90Zr | −8.7553 | −8.7308 (−8.710) | 4.2995 (4.269) | 0.062 | 1g9/2 −12.098 / 2p1/2 −8.316 |
| SLy4 | 132Sn | −8.3846 | −8.3603 (−8.355) | 4.7368 (4.709) | 0.221 | 1h11/2 −7.682 / 1g9/2 −15.604 |
| SLy4 | 208Pb | −7.8889 | −7.8640 (−7.867) | 5.5154 (5.501) | 0.160 | 3p1/2 −8.066 / 3s1/2 −8.815 |
| SLy5 | 16O | −8.0465 | −8.0250 | 2.7993 | −0.025 | 1p1/2 −14.428 / 1p1/2 −11.083 |
| SLy5 | 40Ca | −8.6250 | −8.6024 | 3.5073 | −0.048 | 1d3/2 −15.186 / 1d3/2 −8.246 |
| SLy5 | 48Ca | −8.6887 | −8.6655 | 3.5413 | 0.161 | 1f7/2 −9.262 / 1d3/2 −15.341 |
| SLy5 | 56Ni | −8.6434 | −8.6200 | 3.7826 | −0.055 | 1f7/2 −16.018 / 1f7/2 −6.937 |
| SLy5 | 90Zr | −8.7282 | −8.7038 | 4.2956 | 0.066 | 1g9/2 −11.580 / 2p1/2 −8.106 |
| SLy5 | 132Sn | −8.3877 | −8.3634 | 4.7298 | 0.225 | 1h11/2 −7.493 / 1g9/2 −15.483 |
| SLy5 | 208Pb | −7.8911 | −7.8662 | 5.5073 | 0.162 | 3p1/2 −8.150 / 3s1/2 −8.732 |
| SIII | 16O | −8.0377 | −8.0123 | 2.7558 | −0.023 | 1p1/2 −14.566 / 1p1/2 −11.172 |
| SIII | 40Ca | −8.5709 | −8.5458 | 3.4984 | −0.045 | 1d3/2 −15.517 / 1d3/2 −8.531 |
| SIII | 48Ca | −8.6850 | −8.6603 | 3.5536 | 0.143 | 1f7/2 −9.423 / 2s1/2 −15.347 |
| SIII | 56Ni | −8.5821 | −8.5582 | 3.8095 | −0.051 | 1f7/2 −15.623 / 1f7/2 −6.545 |
| SIII | 90Zr | −8.6886 | −8.6634 | 4.3319 | 0.058 | 1g9/2 −11.842 / 2p1/2 −6.910 |
| SIII | 132Sn | −8.3593 | −8.3351 | 4.7939 | 0.182 | 1h11/2 −7.668 / 1g9/2 −15.042 |
| SIII | 208Pb | −7.8644 | −7.8395 | 5.5813 | 0.125 | 3p1/2 −7.188 / 3s1/2 −7.424 |
| SkM* | 16O | −8.0020 | −7.9829 | 2.8099 | −0.025 | 1p1/2 −13.551 / 1p1/2 −10.270 |
| SkM* | 40Ca | −8.5476 | −8.5272 | 3.5174 | −0.049 | 1d3/2 −14.322 / 1d3/2 −7.477 |
| SkM* | 48Ca | −8.7718 | −8.7508 | 3.5360 | 0.155 | 1f7/2 −10.413 / 1d3/2 −14.693 |
| SkM* | 56Ni | −8.6895 | −8.6680 | 3.7724 | −0.056 | 1f7/2 −16.273 / 1f7/2 −7.148 |
| SkM* | 90Zr | −8.7214 | −8.6994 | 4.2950 | 0.064 | 1g9/2 −12.518 / 2p1/2 −7.366 |
| SkM* | 132Sn | −8.4349 | −8.4135 | 4.7269 | 0.233 | 1h11/2 −8.487 / 1g9/2 −14.635 |
| SkM* | 208Pb | −7.8887 | −7.8667 | 5.5093 | 0.169 | 3p1/2 −8.074 / 3s1/2 −7.596 |

The measured values are shown once (SLy4 rows); they are the same for every
force. The agreement with experiment is the force's: the solver's own error
at the extrapolated values is below $10^{-5}$ MeV per nucleon (section 3).
Oxygen-16's charge radius comes out 0.06 to 0.11 fm larger than measured
with every force.

In SIII, calcium-48's proton $2s_{1/2}$ lies above $1d_{3/2}$; both are
occupied, so the filling does not depend on the order.

## 3. The mesh and the box

- **Spacing.** Halving it from 0.1 to 0.05 fm raises E/A by 0.0143 to
  0.0191 MeV over the 28 pairs. Lead-208 with SLy4 at 0.1, 0.05 and 0.025 fm
  gives E/A = −7.888929, −7.870243 and −7.865577 MeV: successive changes in
  the ratio 4.005, as a second-order method gives, and the two Richardson
  values −7.864014 and −7.864021 MeV agree to $7 \times 10^{-6}$. For
  oxygen-16 with SLy5 the ratio is 3.997 and the two values agree to
  $4 \times 10^{-6}$.
- **Box.** The default is 15 fm, or $1.2A^{1/3} + 11.5$ fm when wider. With
  SLy4, widening it to 22, 26 and 30 fm changes E/A by at most
  $1.5 \times 10^{-7}$ MeV (lead-208), $1.2 \times 10^{-7}$ (tin-132) and
  $4 \times 10^{-8}$ (calcium-48), and $r_{ch}$ not at all to $10^{-7}$ fm.

## 4. Against skyrme_rpa

No copy of skyrme_rpa (Colò, Cao, Van Giai and Capelli, Comput. Phys.
Commun. 184, 142 (2013)) is on this machine, so the only comparison is its
published sample output for oxygen-16 with SLy5, already used by the post
(posts/refs/skyrme-hartree-fock-of-oxygen-16.out). The default run with
linear mixing gives the bits of lib/skyrme/skyrme.viv's oxygen-16 solver
(the test in spherical.viv), and the extrapolation of the 0.1 and 0.05 fm runs gives
E/A = −8.0250 MeV and $r_{ch}$ = 2.7993 fm against skyrme_rpa's −8.0261 MeV
and 2.7991 fm, and its six orbital energies within 3 keV
(tools/export/test_skyrmehf.py).

## 5. Three engines and timings

`tools/export/check.py` runs each case as the exported library, a native
build of the same Viv files and Viv's interpreter, and requires the same
result bits from all three:

| case | library (s) | native (s) | interpreter (s) | result SHA-256 | engines |
|---|---|---|---|---|---|
| o16_sly5 (the default) | 0.01 | 0.21 | 0.6 | a4cf9b111ffe00c1 | same |
| ca40_sly4 | 0.02 | 0.22 | 1.1 | aacabde1728526d6 | same |
| ca48_skms | 0.03 | 0.24 | 1.5 | 7367c75dc0fb74be | same |
| ni56_sly5 | 0.03 | 0.23 | 1.6 | 3b4fe95b73089bc9 | same |
| zr90_siii | 0.03 | 0.23 | 2.1 | 8b2452a468e26597 | same |
| sn132_sly4 | 0.06 | 0.26 | 3.7 | cda0e4296102da92 | same |
| pb208_sly5 | 0.09 | 0.29 | 5.8 | 1c581b2d91084460 | same |

The native time includes starting the process. The numpy reference takes
0.4 to 4.4 s per run at 0.1 fm and 2 to 22 s at 0.05 fm (one thread); the
exported library takes 0.016 to 0.089 s and 0.028 to 0.161 s.

On 2026-10-14 the default mixing became Pulay's (lib/skyrme/skyrmehf.viv,
`MIXING`): the default run takes 25 passes instead of 38 and its hash
changed from 8005a7dded050211. Against the linear run, E/A moves by
$1.1 \times 10^{-10}$ MeV, the radii by at most $1.2 \times 10^{-9}$ fm and
the levels by at most $2.0 \times 10^{-8}$ MeV; against a run to
`e_tol=1e-13` the Pulay run is within $7 \times 10^{-11}$ MeV in E/A and
$1 \times 10^{-12}$ fm in $r_{ch}$ (the linear run: $5 \times 10^{-11}$ MeV and
$1.1 \times 10^{-9}$ fm). The six other cases set `mixing = "linear"`, as
compare.py does, so that their expected values stay the numpy solver's at
the same iteration; their hashes are unchanged.

Earlier, the default run's hash changed from d95ce853... because the program now
returns more results (Z, N, the labels, the spectrum, the refills) and
LEVELS is one row, neutrons then protons. Its energy, charge radius, levels
and pass count are the same bits as before (the test at the end of
spherical.viv compares them with lib/skyrme/skyrme.viv's solver).

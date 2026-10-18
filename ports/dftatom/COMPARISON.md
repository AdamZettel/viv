# dftatom in Viv against dftatom

The port (`dftatom.viv`, `atom.viv`) against upstream's Fortran, upstream's own converged
energies and NIST's atomic reference data. `test_dftatom.py` holds every number below to its
tolerance; `python3 ports/dftatom/test_dftatom.py --table` prints these tables.

## The upstream build

dftatom at `e49b3042` (version 1.0.2), cloned outside the repository and built with CMake and
gfortran 15.2.0 (Homebrew GCC) on an Apple M2 Pro, twice:

- `build`: upstream's release flags, `-O3 -march=native -ffast-math -funroll-loops`. Its `ctest`
  passes 21 of 23 tests. The two that fail, `conv_lda` and `conv_rlda`, stop because `ctest` runs
  them in the build directory without their data files (`data3.dat`...`data8.dat`). Run from
  `tests/lda` and `tests/rlda` in their quick mode, they pass: E_tot within 5.7e-9 (LDA) and
  5.3e-9 (RLDA) of upstream's tables for Z = 92, 69, 46, 23, against their tolerance of 1.2e-8.
- `build-strict`: `-O2 -fno-fast-math -ffp-contract=off`, IEEE arithmetic with no fused
  multiply-add, the arithmetic a Viv program does. This is the oracle.

`oracle/oracle.f90` runs one SCF atom with upstream's library: it is upstream's `atom_lda` /
`atom_rlda` body, repeated so it can print the energy components, the iteration count and the
density that `atom_lda` keeps internally. `oracle/make_reference.py` runs it on every case below
and writes `oracle/upstream_reference.json`.

## The cases

- the exported program's four cases (`atom.cases.json`): iron with the default mesh, hydrogen,
  neon relativistic, iron with bisection only (`perturb = false`);
- upstream's `tests/lda` and `tests/rlda` runs for the 23 atoms NIST lists that the comparison
  uses (H to Ar, Fe, Kr, Xe, Au, U), each on the mesh upstream's `data8.dat` gives that atom: the
  meshes on which upstream converges to 1e-8 Hartree. Nonrelativistic: `r_min = 1e-7`,
  `a = 2.7e6`; relativistic: `r_min = 1e-8`, `a = 6.2e7`, `c = 137.0359895`. All with upstream's
  `reigen_eps = 1e-10`, `mixing_eps = 5e-9`, `mixing_alpha = 0.5`.

## Against the Fortran

Two Fortran runs of each case:

- **upstream**, the strict build linked with Viv's math library (`oracle/vmath_shim.c` puts Viv's
  `exp`, `log`, `pow` and `atan` in place of the platform's). Tolerance: **none**. Every compared
  result must be the same double: the five energies, every eigenvalue, the density at every
  500th mesh point, and the SCF iteration count.
- **upstream, platform libm**, the strict build as gfortran links it. Viv's `exp` is correctly
  rounded; the platform's is off by one ulp on some arguments (the first difference between the
  two runs is in the mesh itself, `alpha (exp(beta i) - 1) + r_min`, at 3 of the 3001 points
  of a 3000-interval mesh). Tolerance: the case's `mixing_eps`, on E_tot and on every eigenvalue. The SCF loop
  stops once no eigenvalue moves more than `mixing_eps` between iterations, so two runs that
  round differently can stop that far apart; nothing closer is promised. The components are not
  variational (an error in the density enters them to first order) and are reported, not held.

| case | Z | rel | N | SCF iterations | results not bit-identical to upstream (Viv libm) | platform libm: max |dE_tot| | max |de| | max |d part| |
|---|---|---|---|---|---|---|---|---|
| default | 26 | no | 6000 | 23 | 0 of 26 | 3.9e-12 | 4.8e-13 | 1.5e-11 |
| hydrogen | 1 | no | 3000 | 16 | 0 of 14 | 2.2e-16 | 1.1e-16 | 4.4e-16 |
| neon_rel | 10 | yes | 4000 | 21 | 0 of 19 | 2.6e-13 | 6.4e-14 | 2.6e-13 |
| iron_bisect | 26 | no | 3000 | 27 | 0 of 20 | 2.5e-10 | 1.3e-08 | 5.0e-07 |
| lda_1 | 1 | no | 2511 | 16 | 0 of 13 | 4.4e-16 | 8.3e-17 | 1.9e-15 |
| rlda_1 | 1 | yes | 2596 | 16 | 0 of 13 | 4.4e-16 | 4.4e-16 | 1.3e-15 |
| lda_2 | 2 | no | 2737 | 18 | 0 of 13 | 4.4e-16 | 1.0e-15 | 3.6e-15 |
| rlda_2 | 2 | yes | 2866 | 18 | 0 of 13 | 4.4e-15 | 2.2e-16 | 2.3e-14 |
| lda_3 | 3 | no | 3084 | 19 | 0 of 15 | 1.8e-15 | 1.6e-14 | 3.9e-14 |
| rlda_3 | 3 | yes | 2986 | 19 | 0 of 14 | 2.0e-14 | 6.0e-15 | 3.6e-14 |
| lda_4 | 4 | no | 3382 | 20 | 0 of 15 | 5.0e-14 | 3.9e-14 | 5.0e-14 |
| rlda_4 | 4 | yes | 3034 | 20 | 0 of 15 | 4.8e-14 | 3.9e-14 | 4.8e-14 |
| lda_5 | 5 | no | 3608 | 19 | 0 of 17 | 7.1e-15 | 1.0e-15 | 3.6e-14 |
| rlda_5 | 5 | yes | 3204 | 19 | 0 of 17 | 9.6e-14 | 5.6e-14 | 9.6e-14 |
| lda_6 | 6 | no | 3834 | 22 | 0 of 17 | 2.7e-13 | 1.1e-13 | 8.2e-13 |
| rlda_6 | 6 | yes | 3275 | 22 | 0 of 17 | 7.1e-14 | 3.9e-14 | 3.4e-13 |
| lda_7 | 7 | no | 4012 | 20 | 0 of 18 | 2.5e-13 | 2.5e-13 | 4.8e-13 |
| rlda_7 | 7 | yes | 3352 | 20 | 0 of 17 | 1.6e-13 | 6.8e-13 | 1.0e-12 |
| lda_8 | 8 | no | 4195 | 20 | 0 of 18 | 1.1e-13 | 1.1e-13 | 2.3e-13 |
| rlda_8 | 8 | yes | 3426 | 20 | 0 of 17 | 1.7e-13 | 2.2e-13 | 1.7e-13 |
| lda_9 | 9 | no | 4309 | 19 | 0 of 18 | 1.4e-13 | 1.4e-14 | 2.3e-13 |
| rlda_9 | 9 | yes | 3480 | 19 | 0 of 17 | 2.1e-13 | 4.3e-14 | 4.4e-13 |
| lda_10 | 10 | no | 4521 | 21 | 0 of 19 | 4.0e-13 | 1.9e-13 | 4.0e-13 |
| rlda_10 | 10 | yes | 3539 | 21 | 0 of 18 | 0.0e+00 | 3.9e-13 | 9.1e-13 |
| lda_11 | 11 | no | 4706 | 20 | 0 of 20 | 6.3e-13 | 3.0e-12 | 7.7e-12 |
| rlda_11 | 11 | yes | 3715 | 20 | 0 of 19 | 1.7e-13 | 1.1e-12 | 2.2e-12 |
| lda_12 | 12 | no | 4908 | 23 | 0 of 20 | 1.4e-13 | 6.4e-13 | 1.0e-12 |
| rlda_12 | 12 | yes | 3985 | 23 | 0 of 19 | 6.0e-13 | 2.1e-13 | 2.2e-12 |
| lda_13 | 13 | no | 5178 | 22 | 0 of 22 | 4.5e-13 | 6.5e-13 | 2.8e-12 |
| rlda_13 | 13 | yes | 4291 | 22 | 0 of 22 | 6.3e-13 | 1.2e-12 | 4.0e-12 |
| lda_14 | 14 | no | 5307 | 23 | 0 of 22 | 6.8e-13 | 7.0e-13 | 3.8e-12 |
| rlda_14 | 14 | yes | 4525 | 23 | 0 of 23 | 1.2e-12 | 2.4e-12 | 2.4e-11 |
| lda_15 | 15 | no | 5423 | 23 | 0 of 22 | 1.7e-13 | 2.3e-13 | 5.7e-13 |
| rlda_15 | 15 | yes | 4722 | 23 | 0 of 23 | 5.7e-14 | 7.1e-14 | 5.7e-13 |
| lda_16 | 16 | no | 5614 | 20 | 0 of 23 | 5.7e-14 | 2.8e-12 | 4.7e-12 |
| rlda_16 | 16 | yes | 4940 | 20 | 0 of 23 | 1.1e-13 | 2.1e-12 | 2.1e-12 |
| lda_17 | 17 | no | 5841 | 20 | 0 of 23 | 1.8e-12 | 2.9e-12 | 4.7e-12 |
| rlda_17 | 17 | yes | 5163 | 20 | 0 of 24 | 2.3e-13 | 9.4e-13 | 2.0e-12 |
| lda_18 | 18 | no | 5939 | 20 | 0 of 23 | 2.7e-12 | 7.4e-13 | 1.3e-11 |
| rlda_18 | 18 | yes | 5345 | 20 | 0 of 24 | 1.1e-13 | 1.7e-11 | 4.9e-11 |
| lda_26 | 26 | no | 7534 | 23 | 0 of 29 | 3.2e-12 | 5.6e-13 | 2.9e-11 |
| rlda_26 | 26 | yes | 7203 | 23 | 0 of 31 | 4.8e-12 | 8.0e-13 | 3.5e-11 |
| lda_36 | 36 | no | 8677 | 23 | 0 of 32 | 1.5e-11 | 3.6e-11 | 2.5e-10 |
| rlda_36 | 36 | yes | 8694 | 23 | 0 of 36 | 3.6e-12 | 1.9e-11 | 9.8e-11 |
| lda_54 | 54 | no | 12119 | 23 | 0 of 42 | 4.4e-10 | 7.9e-11 | 1.2e-09 |
| rlda_54 | 54 | yes | 12002 | 22 | 0 of 48 | 1.6e-10 | 9.9e-11 | 5.2e-10 |
| lda_79 | 79 | no | 14841 | 27 | 0 of 50 | 3.3e-11 | 3.0e-12 | 9.0e-10 |
| rlda_79 | 79 | yes | 15782 | 25 | 0 of 60 | 1.6e-10 | 4.7e-11 | 7.6e-10 |
| lda_92 | 92 | no | 18099 | 26 | 0 of 61 | 1.9e-09 | 6.7e-10 | 1.9e-08 |
| rlda_92 | 92 | yes | 18082 | 23 | 0 of 72 | 1.5e-11 | 9.1e-13 | 2.9e-11 |

All 50 cases equal upstream bit for bit with the same libm. With the platform's libm the largest
differences are 1.9e-9 Hartree in E_tot (uranium, LDA), 1.3e-8 in an eigenvalue (iron with
bisection only, whose `mixing_eps` is 1e-7), and 5.0e-7 in a component (the same case), each
within the stated tolerance.

**Upstream's converged tables.** `tests/lda/nonrel_energies.dat` and `tests/rlda/rel_energies.dat`
hold upstream's converged E_tot and eigenvalues to 1e-10, and its `conv_lda` and `conv_rlda`
hold a run to them within 1.2e-8. The port, on the same 46 runs: E_tot within 7.2e-9 (LDA,
uranium) and 4.9e-9 (RLDA, gold), eigenvalues within 2.3e-9.

**Upstream's radial tests** (`tests.viv`, from `tests/Z92n7all`): every state with n <= 7 of
hydrogen-like uranium, 28 Schrödinger states and 49 Dirac states, within 1e-6 Hartree of
`-Z^2/2n^2` and of Sommerfeld's formula, on upstream's meshes and tolerances. The port prints
upstream's digits: for example 1s1/2 at -4861.197905 with error 1.39e-7 and 2p3/2 at
-1257.395852 with error 5.09e-8, as upstream's `Z92_rel` does.

## Against NIST

NIST Standard Reference Database 141 (Kotochigova, Levine, Shirley, Stiles and Clark, Phys. Rev.
A 55, 191 (1997)), the LDA and RLDA columns, fetched from
<https://math.nist.gov/DFTdata/atomdata/tables/ptable.html> by `oracle/nist.py` into
`oracle/nist_reference.json`. The tolerance is upstream's own statement of NIST's accuracy
(README.rst, "Accuracy"): **1e-6 Hartree in total energies and 2e-6 in eigenvalues**. NIST prints
six decimals, so its rounding alone is up to 5e-7. NIST states no accuracy for the energy
components; they are reported, not held.

| atom | method | E_tot NIST | E_tot Viv | |dE_tot| | max |d eigenvalue| | max |d part| |
|---|---|---|---|---|---|---|
| H (1) | LDA | -0.445671 | -0.4456705183 | 4.8e-07 | 1.8e-09 (1s) | 4.2e-07 |
| H (1) | RLDA | -0.445668 | -0.4456681624 | 1.6e-07 | 2.1e-07 (1s k=-1) | 4.3e-07 |
| He (2) | LDA | -2.834836 | -2.8348356241 | 3.8e-07 | 2.8e-07 (1s) | 4.2e-07 |
| He (2) | RLDA | -2.834785 | -2.8347852847 | 2.8e-07 | 3.6e-07 (1s k=-1) | 5.0e-07 |
| Li (3) | LDA | -7.335195 | -7.3351951888 | 1.9e-07 | 3.4e-07 (2s) | 4.2e-07 |
| Li (3) | RLDA | -7.335231 | -7.3352306819 | 3.2e-07 | 2.8e-07 (2s k=-1) | 4.6e-07 |
| Be (4) | LDA | -14.447209 | -14.4472094743 | 4.7e-07 | 3.9e-07 (1s) | 4.8e-07 |
| Be (4) | RLDA | -14.447997 | -14.4479969883 | 1.2e-08 | 1.5e-07 (1s k=-1) | 4.4e-07 |
| B (5) | LDA | -24.344198 | -24.3441981032 | 1.0e-07 | 1.7e-07 (2s) | 4.6e-07 |
| B (5) | RLDA | -24.347331 | -24.3473306522 | 3.5e-07 | 1.9e-07 (2s k=-1) | 3.8e-07 |
| C (6) | LDA | -37.425749 | -37.4257485364 | 4.6e-07 | 2.8e-07 (2p) | 2.9e-07 |
| C (6) | RLDA | -37.434171 | -37.4341706246 | 3.8e-07 | 4.1e-07 (1s k=-1) | 4.9e-07 |
| N (7) | LDA | -54.025016 | -54.0250161401 | 1.4e-07 | 3.0e-07 (2p) | 4.9e-07 |
| N (7) | RLDA | -54.043484 | -54.0434835098 | 4.9e-07 | 2.4e-07 (2s k=-1) | 2.1e-07 |
| O (8) | LDA | -74.473077 | -74.4730768053 | 1.9e-07 | 2.2e-07 (2p) | 3.5e-07 |
| O (8) | RLDA | -74.508653 | -74.5086533387 | 3.4e-07 | 4.8e-07 (2p k=-2) | 3.4e-07 |
| F (9) | LDA | -99.099648 | -99.0996481094 | 1.1e-07 | 3.2e-07 (2s) | 3.9e-07 |
| F (9) | RLDA | -99.162239 | -99.1622390493 | 4.9e-08 | 4.7e-07 (1s k=-1) | 4.7e-07 |
| Ne (10) | LDA | -128.233481 | -128.2334812700 | 2.7e-07 | 4.3e-07 (2s) | 4.9e-07 |
| Ne (10) | RLDA | -128.336403 | -128.3364032511 | 2.5e-07 | 4.7e-07 (2p k=-2) | 3.8e-07 |
| Na (11) | LDA | -161.440060 | -161.4400603103 | 3.1e-07 | 5.2e-07 (1s) | 3.3e-07 |
| Na (11) | RLDA | -161.601691 | -161.6016906960 | 3.0e-07 | 4.5e-07 (1s k=-1) | 3.2e-07 |
| Mg (12) | LDA | -199.139406 | -199.1394063167 | 3.2e-07 | 4.3e-07 (2s) | 4.5e-07 |
| Mg (12) | RLDA | -199.382965 | -199.3829647097 | 2.9e-07 | 4.5e-07 (3s k=-1) | 4.6e-07 |
| Al (13) | LDA | -241.315573 | -241.3155734087 | 4.1e-07 | 4.2e-07 (2p) | 3.4e-07 |
| Al (13) | RLDA | -241.669323 | -241.6693226960 | 3.0e-07 | 4.9e-07 (1s k=-1) | 2.9e-07 |
| Si (14) | LDA | -288.198397 | -288.1983966050 | 4.0e-07 | 4.4e-07 (3p) | 5.2e-07 |
| Si (14) | RLDA | -288.696925 | -288.6969246453 | 3.5e-07 | 5.1e-07 (2p k=1) | 4.8e-07 |
| P (15) | LDA | -339.946219 | -339.9462189628 | 3.7e-08 | 2.5e-07 (1s) | 4.7e-07 |
| P (15) | RLDA | -340.631017 | -340.6310171174 | 1.2e-07 | 4.4e-07 (2p k=-2) | 5.0e-07 |
| S (16) | LDA | -396.716081 | -396.7160805033 | 5.0e-07 | 2.8e-07 (3p) | 4.2e-07 |
| S (16) | RLDA | -397.636285 | -397.6362849621 | 3.8e-08 | 4.4e-07 (1s k=-1) | 4.9e-07 |
| Cl (17) | LDA | -458.664179 | -458.6641794953 | 5.0e-07 | 3.6e-07 (3s) | 2.8e-07 |
| Cl (17) | RLDA | -459.877336 | -459.8773357322 | 2.7e-07 | 3.6e-07 (1s k=-1) | 5.4e-07 |
| Ar (18) | LDA | -525.946195 | -525.9461949212 | 7.9e-08 | 4.7e-07 (1s) | 5.3e-07 |
| Ar (18) | RLDA | -527.519049 | -527.5190491499 | 1.5e-07 | 4.0e-07 (1s k=-1) | 5.9e-07 |
| Fe (26) | LDA | -1261.093056 | -1261.0930558462 | 1.5e-07 | 4.6e-07 (1s) | 4.9e-07 |
| Fe (26) | RLDA | -1269.229080 | -1269.2290800709 | 7.1e-08 | 5.3e-07 (4s k=-1) | 3.8e-07 |
| Kr (36) | LDA | -2750.147940 | -2750.1479404237 | 4.2e-07 | 4.4e-07 (2s) | 9.6e-07 |
| Kr (36) | RLDA | -2784.199238 | -2784.1992381263 | 1.3e-07 | 4.8e-07 (2s k=-1) | 7.0e-07 |
| Xe (54) | LDA | -7228.856107 | -7228.8561064889 | 5.1e-07 | 4.5e-07 (3d) | 2.0e-06 |
| Xe (54) | RLDA | -7433.498065 | -7433.4980651025 | 1.0e-07 | 4.9e-07 (2p k=-2) | 2.1e-06 |
| Au (79) | LDA | -17860.790943 | -17860.7909426366 | 3.6e-07 | 5.2e-07 (5d) | 3.2e-06 |
| Au (79) | RLDA | -18998.624707 | -18998.6247073550 | 3.5e-07 | 5.5e-07 (1s k=-1) | 6.8e-06 |
| U (92) | LDA | -25658.417889 | -25658.4178888606 | 1.4e-07 | 1.2e-06 (1s) | 8.0e-06 |
| U (92) | RLDA | -28001.132325 | -28001.1323254916 | 4.9e-07 | 5.4e-07 (1s k=-1) | 6.3e-05 |

Every total energy is within 5.1e-7 of NIST's and every eigenvalue within 1.2e-6 (uranium LDA
1s, where upstream's own table has -3689.3551398369 against NIST's -3689.355141, the same
difference). The components of the heavy atoms differ from NIST's by up to 6.3e-5 (uranium,
RLDA) in upstream as in the port, while the totals agree: NIST's components and upstream's are
not computed to the totals' precision.

## The same bits in every engine

`tools/export/check.py ports/dftatom/atom.export.toml` runs each case of `atom.cases.json` as
the exported library (C through the Python wrapper), as a native program and in the
interpreter, and requires the same SHA-256 of all results from all three. The hashes are in
`dftatom.exact_hashes.json` and `test_dftatom.py` checks the library against them.

| case | library (s) | native (s) | interpreter (s) | sha256 | native | interpreter |
|---|---|---|---|---|---|---|
| default | 0.10 | 0.40 | 47.9 | 1e19607d950443fa | same | same |
| hydrogen | 0.02 | 0.35 | 3.6 | a0c3e0bb7c980812 | same | same |
| neon_rel | 0.05 | 0.31 | 23.2 | 5aaaa188673a9749 | same | same |
| iron_bisect | 0.46 | 0.91 | 254.8 | 3b122b062cb5b178 | same | same |

(The native column includes starting the process; the machine was loaded during this run.)

## Timings

One SCF solve, on the same machine, each engine best of 5 with the runs interleaved so that
other load on the machine falls on all of them alike. Upstream with its own release flags and
with the strict flags; the Viv library in its exact build (the one compared above) and its fast
build (`vipyr.load(..., fast=True)`, which may differ in the last bits).

| case | upstream, -O3 -march=native -ffast-math (s) | upstream, -O2 strict (s) | Viv library, exact (s) | Viv library, fast (s) | Viv exact / upstream -O3 |
|---|---|---|---|---|---|
| default | 0.070 | 0.091 | 0.105 | 0.084 | 1.49 |
| neon_rel | 0.028 | 0.036 | 0.049 | 0.039 | 1.74 |
| lda_54 | 0.173 | 0.223 | 0.253 | 0.217 | 1.46 |
| rlda_54 | 0.303 | 0.367 | 0.447 | 0.406 | 1.47 |
| lda_92 | 0.474 | 0.609 | 0.672 | 0.608 | 1.42 |
| rlda_92 | 0.851 | 1.026 | 1.234 | 1.147 | 1.45 |

The port is about 1.45 times slower than upstream built with `-ffast-math`, and 1.1 to 1.4
times slower than upstream built with the same strict IEEE flags. The fast build runs within 12
percent of the strict Fortran, and faster than it on two cases. These are losses; nothing in the
port has been tuned yet.

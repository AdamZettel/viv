# The port against upstream

Upstream: https://github.com/Allen-Tildesley/examples at `4818bc82`.
Checks, in order of trust: the same doubles as upstream's Fortran; the
equation of state and GUIDE.md's test results, within the runs' error bars;
the dt^2 law of velocity Verlet. `python3 ports/liquids/test_liquids.py`
runs all of them (`--table` prints the first table).

## 1. The same doubles as upstream

oracle/build.py builds upstream's programs with GNU Fortran 15.2.0
(Homebrew GCC), upstream's flags (`-fdefault-real-8 -fall-intrinsics`) and
`-O2 -ffp-contract=off -fno-fast-math`, with three changes that leave the
arithmetic alone: the random numbers are MINSTD (oracle/minstd.f90, the
generator behind Viv's `rand()`, seeded by MINSTD_SEED as `srand` seeds it),
the libm is Viv's (oracle/vmath_shim.c; gfortran turns Box-Muller's SIN and
COS into one `cexp` call, so that is replaced too), and the output has 17
significant digits. oracle/make_reference.py runs initialize and then the
program, both seeded with 1, and keeps every number printed
(oracle/upstream_reference.json); the port runs the same case from the same
seed.

Tolerance: zero. A double that differs in any bit fails.

| case | upstream program | doubles compared | differing |
|---|---|---|---|
| md_nve_small (nblock=2, nstep=20; N=108) | md_nve_lj | 673 | 0 |
| md_nve_ll (nblock=2, nstep=10; N=500, link cells) | md_nve_lj_ll | 3025 | 0 |
| md_nve_vl (nblock=2, nstep=50; N=256, Verlet list) | md_nve_lj_vl | 1561 | 0 |
| md_nve_long (nblock=5, nstep=400; N=256) | md_nve_lj | 1585 | 0 |
| md_nvt_small (nblock=2, nstep=20; N=108) | md_nvt_lj | 679 | 0 |
| mc_nvt_small (nblock=2, nstep=5; N=108) | mc_nvt_lj | 346 | 0 |
| mc_nvt_long (nblock=4, nstep=100; N=256) | mc_nvt_lj | 804 | 0 |
| mc_npt_small (nblock=2, nstep=5; N=256) | mc_npt_lj | 802 | 0 |
| eos_lj at (T, rho) = (1, 0.75), (2, 0.1), (0.9, 0.85), (1.5, 0.5) | eos_lj | 4 x 28 | 0 |
| pair_distribution of 30 configurations of md_nve_lj, N=108 | pair_distribution | 131 x 2 | 0 |
| diffusion of the same 30 frames, nt=20, origins every 4 | diffusion | 21 x 3 | 0 |
| error_calc, nstep=4096, nequil=1000, n_repeat=4 | error_calc | 6 + 27 x 2 | 0 |

"Doubles compared" for a simulation: the run averages, every block average
and the final configuration (positions, and velocities for MD) and box.
The run errors (standard errors of the block averages) are compared too.
For eos_lj, 16 properties and the 12 entries of a_res used, per state point.

## 2. The same bits in every engine

`tools/export/check.py --record` ran each case of each manifest
(`<program>.cases.json`) three ways: the vipyr library (exact build),
the program compiled natively (cc -O2 -ffp-contract=off), and Viv's
interpreter. All three gave the same result hash for every case; the hashes
are `<program>.exact_hashes.json`, and test_liquids.py holds the library to
them. `tools/three-engines.sh ports/liquids/tests` runs tests/engines.viv,
which exercises every algorithm (all pairs, link cells, Verlet lists, Monte
Carlo sweeps, the equation of state, g(r), the correlation functions,
blocking and the averages) and prints each result to 17 digits, in the
interpreter, native C and wasm: identical output.

## 3. Longer runs against the equation of state and GUIDE.md

N = 256, rho = 0.75, T = 1.0. MD runs start from an equilibrated liquid
(md_nvt, 4 blocks of 1000 steps from the lattice); MC runs from 500 sweeps
of their own. Errors are standard errors of 10 block averages. Tolerance:
4 standard errors (of the run, combined with GUIDE.md's own error where
the reference is GUIDE.md).

| program | quantity | port | reference | source of the reference | deviation |
|---|---|---|---|---|---|
| md_nvt (10 x 2000 steps) | E/N, cut and shifted | -2.9343 +- 0.0072 | -2.9287 | Thol et al. 2015 fit (eos_lj) | -0.8 sigma |
| md_nvt | P, cut and shifted | 0.9841 +- 0.0122 | 0.9897 | Thol et al. 2015 fit | -0.5 sigma |
| md_nvt | T kinetic | 0.9986 +- 0.0033 | 1 | the thermostat | -0.4 sigma |
| md_nve (10 x 2000 steps) | P, cut and shifted | 1.1724 +- 0.0071 | 1.1821 | Thol et al. 2015 fit at the run's T = 1.0479 | -1.4 sigma |
| mc_nvt (10 x 1000 sweeps) | E/N, cut | -3.3323 +- 0.0026 | -3.332(1) | GUIDE.md, mc_nvt_lj | -0.1 sigma |
| mc_nvt | P, cut | 0.6509 +- 0.0092 | 0.651(3) | GUIDE.md | 0.0 sigma |
| mc_nvt | E/N, full | -3.7339 +- 0.0026 | -3.734(1) | GUIDE.md | 0.0 sigma |
| mc_nvt | P, full | 0.3498 +- 0.0092 | 0.350(3) | GUIDE.md | 0.0 sigma |
| mc_nvt | C_v/N, full | 2.233 +- 0.031 | 2.28(1) | GUIDE.md | -1.5 sigma |
| mc_npt (P = 0.69, 10 x 500 steps) | density | 0.7527 +- 0.0015 | 0.7501(2) | GUIDE.md, mc_npt_lj | +1.7 sigma |
| mc_npt | E/N, cut | -3.352 +- 0.012 | -3.331(1) | GUIDE.md | -1.7 sigma |
| mc_npt | P, cut | 0.6622 +- 0.0045 | 0.666(2) | GUIDE.md | -0.8 sigma |
| mc_npt | P, full | 0.3587 +- 0.0045 | 0.364(2) | GUIDE.md | -1.1 sigma |

The Monte Carlo programs are compared with GUIDE.md rather than with the
equation of state: their cut (not shifted) potential is reached from the
full-potential fit through long-range and delta corrections that assume
g(r) = 1 beyond r_c, and GUIDE.md shows upstream's own runs sitting 0.012
below the fit's E/N and 0.05 below its P for that reason. The port
reproduces upstream's values, bias included.

Not checked against a reference: the configurational temperature of md_nve
(1.057 at T = 1.048). The cut-and-shifted force jumps at r_c, which adds a
term to the Laplacian that upstream leaves out (GUIDE.md: "there should be
a delta correction for the configurational temperature").

The gallery page "The Lennard-Jones fluid against its equation of state"
adds seven more state points from the gas (rho = 0.1) to the dense liquid
(rho = 0.85), each within 2.3 standard errors of the 2015 fit.

## 4. Energy conservation and the time step

md_nve from the equilibrated liquid for 4 time units at each step: the
mean-square deviation of E/N within the run.

| dt | port | GUIDE.md, md_nve_lj (10 blocks of 16 time units) |
|---|---|---|
| 0.008 | 2.080e-07 | 1.8430e-7 |
| 0.004 | 1.480e-08 | 1.3121e-8 |
| 0.002 | 8.385e-10 | 1.2621e-9 |
| 0.001 | 5.854e-11 | 1.8530e-10 |

Least-squares slope of log MSD on log dt: 3.95; velocity Verlet's energy
error is O(dt^2), so its mean square is O(dt^4). test_liquids.py requires
4 +- 0.4. (GUIDE.md's runs are longer and its smallest steps level off
above dt^4, as GUIDE.md notes.)

## 5. Timings

Wall seconds, best of 2, measured 2026-10-13 on Darwin arm64, arm, GNU Fortran (Homebrew GCC 15.2.0) 15.2.0, Python 3.11.5 with NumPy 1.24.4, by oracle/bench.py; timings.json has them. The machine was shared: the load average before each set is listed. Fortran: upstream's stock build with -O2 (fused multiply-adds allowed, RANDOM_NUMBER, the platform's libm). Python: upstream's python_examples. Viv: the vipyr library, exact build (-O2 -ffp-contract=off) and fast build (-O3 -ffp-contract=fast).

| run | load | Fortran | Python | Viv exact | Viv fast | Viv exact / Fortran |
|---|---|---|---|---|---|---|
| md_nve N=256, 10 x 200 steps | 7.7 | 0.45 | 28.1 | 0.94 | 0.81 | 2.1 |
| md_nve link cells N=864, 10 x 200 steps | 8.1 | 3.41 | - | 4.66 | 4.25 | 1.4 |
| md_nvt N=256, 10 x 200 steps | 8.9 | 0.27 | - | 0.41 | 0.41 | 1.5 |
| mc_nvt N=256, 10 x 100 sweeps | 8.8 | 0.56 | 18.1 | 0.90 | 0.88 | 1.6 |

The port is 1.4 to 2.1 times slower than gfortran here and 20 to 30 times faster than upstream's NumPy. Viv's C checks the bounds and rank of every array access; upstream's Fortran build does not. Not profiled further in this slice.

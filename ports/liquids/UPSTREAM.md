# Upstream

- Code: Allen and Tildesley's examples, https://github.com/Allen-Tildesley/examples,
  by Michael P. Allen and Dominic J. Tildesley, accompanying *Computer
  Simulation of Liquids*, 2nd ed. (Oxford University Press, 2017)
- Ported from: commit `4818bc821a5acbfb3ebfc54ef9117968bdb22ad1` (2026-03-22),
  fetched on 2026-10-13
- License: CC0-1.0, a public domain dedication (LICENSE is upstream's
  COPYING.txt, unchanged). Nothing obliges the attribution; the port gives it
  anyway, as every port does.
- The equation of state is Thol et al.'s fit (Journal of Physical and
  Chemical Reference Data 45, 023101 (2016); International Journal of
  Thermophysics 36, 25 (2015)), which upstream coded
  independently of those authors' C++, as this port codes it from upstream.

## What was ported

| Upstream file | Port | What |
|---|---|---|
| md_nve_lj.f90 with md_lj_module, md_lj_ll_module, md_lj_vl_module | md_nve.viv, lj.viv | velocity Verlet NVE; all pairs, link cells or Verlet lists (`neighbors`) |
| md_nvt_lj.f90 (and its md_nvt_lj_ll build) | md_nvt.viv | Nose-Hoover chain NVT, the MTTK splitting |
| mc_nvt_lj.f90 with mc_lj_module | mc_nvt.viv, lj.viv | Metropolis NVT |
| mc_npt_lj.f90 | mc_npt.viv | Metropolis NPT with box-scaling moves |
| lrc_lj_module.f90 | lj.viv | long-range and delta corrections |
| link_list_module.f90 | lj.viv | cell lists and the 13-cell forward search |
| verlet_list_module.f90 | lj.viv | Verlet lists with a skin |
| initialize.f90, initialize_module.f90 (atoms) | setup.viv | fcc lattice, Maxwell-Boltzmann velocities |
| maths_module.f90 (the parts used) | setup.viv | random_normal, random_translate_vector, metropolis, exprel, expm1, polyval |
| averages_module.f90 | averages.viv | block averages, mean-square deviations, the C_v formula for NVE |
| pair_distribution.f90 | analysis.viv, pair_distribution.viv | g(r) |
| diffusion.f90 | analysis.viv, diffusion.viv | VACF, MSD, cross-correlation over time origins |
| error_calc.f90 | analysis.viv, error_calc.viv | block analysis of correlated series |
| eos_lj.f90, eos_lj_module.f90 | eos.viv, eos_lj.viv | Thol et al.'s fits, full and cut-and-shifted |

Not ported in this slice: hard spheres, Gibbs and grand-canonical Monte
Carlo, Wang-Landau, Ewald and mesh electrostatics, chains, polyatomics,
path-integral programs, the Monte Carlo link-list builds (mc_*_lj_ll),
md_npt_lj, corfun.f90 (its FFT correlation; the direct correlation it is
compared with is what diffusion.f90 computes) and the Python examples.

## Differences from upstream

- **Random numbers.** Upstream calls Fortran's RANDOM_NUMBER, seeded by
  RANDOM_INIT (a different stream every run). The port uses Viv's `rand()`,
  Park-Miller MINSTD seeded by `srand(seed)`, so a run is the same on every
  machine and in every engine, and `seed` is a parameter. MINSTD's period is
  2^31 - 2, enough for runs of up to about 10^8 Monte Carlo moves (four draws
  per move); its successive draws have the lattice structure of any linear
  congruential generator.
- **One program instead of two.** Upstream writes a starting configuration
  with `initialize` and reads it in the simulation; each seeds its own
  generator. The port does both in one run and seeds twice the same way:
  `srand(seed)` before the lattice velocities, and again before the run's
  own draws (md_nvt's thermostat momenta, the Monte Carlo moves).
  `r_start`, `v_start` and `box_start` replace reading `cnf.inp`; the final
  configuration is returned instead of written to `cnf.out`.
- **No equilibration inside a run**, as upstream: a run from the lattice
  includes the melting. Chain two runs (the second from the first's
  `r_final`, `v_final`, `box`) as upstream chains `cnf.out` to `cnf.inp`.
- **Analysis in the run.** With `sample > 0`, the MD programs feed g(r) and
  the correlation functions as they go, instead of writing `cnf.001`, ...
  for pair_distribution and diffusion. pair_distribution and diffusion are
  also programs of their own, on arrays of frames.
- **Additions:** `d_vacf` and `d_msd` (diffusion coefficients from the
  curves; upstream leaves that step to the reader), the per-step energy
  series, error_calc's `series` (analyze a given series).
- **Defaults.** nstep defaults to 1000 (upstream 20000 for MD, 10000 for
  MC) so that a default call takes about a second.
- **Not kept:** upstream's printed table layout, the timestamp and compiler
  lines, the numbered configuration files.
- **mc_npt** returns no g(r): the box changes during the run.

## How the port is checked against it

oracle/build.py builds upstream's programs with gfortran twice: a stock
build (upstream's sources and flags, -O2) for timings, and an exact build
whose only changes are MINSTD in place of RANDOM_NUMBER (oracle/minstd.f90),
17-digit output, Viv's libm (oracle/vmath_shim.c) and no fused
multiply-adds. oracle/make_reference.py runs the exact build on the cases
of upstream_reference.json; test_liquids.py requires every double the port
returns for those cases to equal upstream's. COMPARISON.md has the tables.

## Correspondence with the authors

Not yet written to. To do before publishing vipyr-liquids: send Michael
Allen and Dominic Tildesley the comparison and the attribution, and offer to
list them as maintainers.

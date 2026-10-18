# lib/minpack: MINPACK's Levenberg-Marquardt, in Viv

A port of `lmder` (nonlinear least squares with a user-supplied Jacobian) and
the routines it calls, from the Fortran MINPACK on netlib, written so that a
reader can run it and follow what it does:

- the Fortran's operations are kept in the Fortran's order, so the port can
  be compared with a gfortran build of the original number by number;
- the go-to blocks of the Fortran are named `step` blocks in place, and each
  step has a plain-language entry in the module's `.intent` file;
- `.debug` files watch the iteration (`x`, `fnorm`, `par`, `delta`,
  `ratio`, `pnorm`) and hold invariants, and `viv story` prints a run
  iteration by iteration in the intent's words.

```sh
lib/minpack/check.sh                                   # everything below, exits nonzero on a failure
viv run lib/minpack/mgh_check.viv                      # 24 test cases against the Fortran oracle
viv story lib/minpack/story_rosenbrock.viv             # one run as a story (add --expand for every step)
viv story lib/minpack/story_osborne1.viv
viv view lib/minpack/lmder.viv --lens all              # code and intent interleaved
```

## Provenance

Fetched from <https://www.netlib.org/minpack/> on 2026-09-25 into `ref/`,
unmodified (the dates are netlib's Last-Modified headers; the Fortran
headers say "Argonne National Laboratory. MINPACK Project. March 1980"):

| file | what | netlib date |
|---|---|---|
| `lmder.f` | the Levenberg-Marquardt driver loop | 1984-01-05 |
| `lmpar.f` | the Levenberg-Marquardt parameter for a trust radius | 1984-01-05 |
| `qrfac.f` | Householder QR with column pivoting | 1984-01-05 |
| `qrsolv.f` | the damped least-squares system from a QR factorization | 1984-01-05 |
| `enorm.f` | Euclidean norm without overflow or underflow | 1984-01-05 |
| `dpmpar.f` | machine constants (the IEEE branch is used) | 1996-11-26 |
| `lmder1.f` | the easy-to-use driver | 1984-01-05 |
| `lmdif.f`, `fdjac2.f` | the finite-difference variant (fetched, not ported) | 1984-01-05 |
| `file17.f` | `ex/file17`, the double-precision LMDER test driver (SSQFCN, SSQJAC, INITPT) | 1988-02-23 |
| `file22.dat` | `ex/file22`, its input: the 28 test cases | 1988-02-23 |
| `disclaimer` | the MINPACK copyright notice and license (University of Chicago, 1999) | 2005-06-10 |

MINPACK's license permits redistribution with the notice, which is
`ref/disclaimer`. As its clause 3 asks: this product includes software
developed by the University of Chicago, as Operator of Argonne National
Laboratory.

The Viv port is a derived work and carries the same notice: `enorm.viv`,
`fit.viv`, `lmder.viv`, `lmpar.viv`, `qrfac.viv`, `qrsolv.viv`, `mgh.viv`,
`mgh_ref.viv`, their `.intent` and `.debug` files and
`kernels/k1/norms.viv` are under `LicenseRef-Minpack`
(`LICENSES/LicenseRef-Minpack.txt`), in their SPDX lines and in REUSE.toml.

`data/*.dat` are NIST's 27 Statistical Reference Datasets for nonlinear
regression, fetched unmodified from
<https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/> (Thurber on
2026-09-25 for the post, the other 26 on 2026-10-13), with their certified
parameters and standard deviations.

## Files

| file | what |
|---|---|
| `enorm.viv` | `enorm`, and `EPSMCH`, `DWARF` as dpmpar gives them |
| `qrfac.viv` | `qrfac` |
| `qrsolv.viv` | `qrsolv` |
| `lmpar.viv` | `lmpar` |
| `lmder.viv` | `lmder`, `lmder1`, the `H_` columns of the per-step record |
| `*.intent`, `*.debug` | the intent and debug lenses of those five (with LaTeX: glossary symbols, equations that cite the hold or test checking them, and `run` lines filled in from a recorded run) |
| `mgh.viv` | 11 of the 18 problems of the netlib test driver |
| `mgh_ref.viv` | generated: the Fortran oracle's results and every trial step |
| `mgh_check.viv` | runs the 24 cases and compares them with `mgh_ref.viv` |
| `story_rosenbrock.viv`, `story_osborne1.viv` | runs to read with `viv story`; the first, with `story_rosenbrock.intent`, is the explorer's headline program (viv-lang.org/explore/minpack-rosenbrock) |
| `fit.viv` | `lm_fit` (lmder from a start, returning SciPy's least_squares fields) and `lm_fd_jac` (fdjac2, the difference Jacobian of lmdif), for vp.optimize; its tests fit NIST's Misra1a |
| `optimize.docs.toml` | the reference prose of `vp.optimize.least_squares`, `curve_fit` and `root` (python/vipyr/_optimize.py writes the drivers) |
| `nist_strd.py` | reads `data/*.dat`; the 27 NIST models as Python functions in SciPy's form; NIST's LRE |
| `mgh_ls.viv` | the MGH problems of `mgh.viv` in least_squares' form, for the comparison with SciPy |
| `bench_optimize.py`, `bench_optimize.json` | vp.optimize against NIST's certified values and SciPy's least_squares, curve_fit and root: digits, evaluation counts, times |
| `oracle/` | the Fortran oracle (`build.sh`, `driver.f`, `thurber.f`), `make_ref.py`, and the scipy comparisons |
| `check.sh` | runs all of it |

The post is `posts/2026-09-25-fitting-electron-mobility-with-minpack.viv`.

### Differences from the Fortran

None changes an arithmetic operation or its order.

- Indices count from 0; `ipvt` holds 0-based column numbers.
- Arrays are `m x n` Viv arrays instead of leading-dimension arrays. Where
  the Fortran hands enorm a piece of a column (`enorm(m-j+1, a(j,j))`),
  qrfac copies that piece into a vector first: Viv has no view of part of a
  column.
- lmder allocates its work arrays instead of taking them as arguments,
  returns `info`, and writes `nfev`, `njev` into `counts`. lmpar returns
  `par` instead of updating an in-out argument.
- The user function is a def `fcn(data, x, fvec, fjac, iflag)` that returns
  iflag (negative to stop); `data` carries whatever the function needs.
- lmder writes one row per trial step into `hist` (the `history` step). The
  Fortran has no such record; the oracle adds the same record to a copy of
  the Fortran (below).
- `x**2` in the Fortran is written `x * x`, and `a*b**2` as `a * (b * b)`,
  which is how gfortran evaluates them.

## From the Fortran to the steps

`lmder.f` (labels are the Fortran's statement labels):

| Fortran | Viv step (in `lmder`) |
|---|---|
| declarations, `epsmch = dpmpar(1)`, `info = 0` ... | `setup` |
| input checks, loop 10 (mode 2: diag > 0), `go to 300` | `check_input` |
| `iflag = 1`, first `fcn`, `fnorm = enorm(m,fvec)`, `par = 0`, `iter = 1` | `start` |
| label 30: `iflag = 2`, `fcn` | outer loop, `jacobian` |
| block before label 40 (`nprint`) | the `nprint > 0` branch (no step) |
| label 40: `call qrfac` | `qr` |
| `if (iter .ne. 1) go to 80`, loop 50 | `scale` |
| label 60, loop 70, `delta = factor*xnorm` | `first_radius` |
| label 80, loops 90-130 (Q^T f into qtf, R's diagonal back into fjac) | `qtf` |
| loops 140-170, `if (gnorm .le. gtol) info = 4` | `gradient` |
| loop 180 (mode 1) | `rescale` |
| label 200: `call lmpar` | inner loop, `lm_parameter` |
| loop 210, `pnorm`, `if (iter .eq. 1) delta = dmin1(delta,pnorm)` | `trial` |
| `iflag = 1`, `fcn` at x + p | `evaluate` |
| `fnorm1`, `actred` | `reductions/actual` |
| loops 220-230, `temp1`, `temp2`, `prered`, `dirder` | `reductions/predicted` |
| `ratio = actred/prered` | `reductions/quotient` |
| `if (ratio .gt. p25) go to 240` and the block before it | `radius/shrink` |
| label 240 up to 250 | `radius/expand` |
| label 260, `if (ratio .lt. p0001) go to 290`, loops 270-280 | `accept` |
| (not in the Fortran) | `history` |
| label 290, info 1, 2, 3 | `converged` |
| info 5, 6, 7, 8 | `stringent` |
| `if (ratio .lt. p0001) go to 200`, `go to 30` | the ends of the two loops |
| label 300 | `finish` |

`lmpar.f`:

| Fortran | Viv step (in `lmpar`) |
|---|---|
| loops 10-60 (Gauss-Newton direction, rank-deficient case) | `gauss_newton` |
| loop 70, `dxnorm`, `fp`, `if (fp .le. p1*delta) go to 220` | `gn_length`, then the `if` |
| loops 80-110, `parl` (label 120 skips it when `nsing < n`) | `lower` |
| loops 130-140, `paru` | `upper` |
| `par = dmax1(par,parl)` ... `if (par .eq. zero) par = gnorm/dxnorm` | `start` |
| label 150, `iter = iter + 1`, loops 160-170, `call qrsolv`, `fp` | loop, `evaluate` |
| the acceptance test before loop 180 | the `break` |
| loops 180-210, `parc` | `newton` |
| the two sign tests on fp | `bracket` |
| `par = dmax1(parl,par+parc)`, `go to 150` | `improve` |
| label 220 | `finish` |

`qrfac.f`: loop 10 is `norms`; loop 110 (pivoting, loops 20-30; the
Householder vector, loop 50; applying it and downdating the norms, loops
60-90) is `reduce`, with no steps inside it, so the story shows qrfac as one
call. `qrsolv.f`: loops 10-20 `copy`, loops 30-100 `eliminate`, loops
110-150 `solve`, loop 160 `permute`. `enorm.f` has no steps (see
`check.sh`); the three sums of its loop 90 and the three cases after it are
explained in `enorm.intent`.

## The references

**The Fortran oracle.** `oracle/build.sh` compiles the netlib files with
`gfortran -O0 -ffp-contract=off` (no fused multiply-add; Viv's engines do not
fuse either) and runs:

- the netlib test driver on its own data, unmodified (`netlib.txt`);
- `oracle/driver.f`, which runs lmder1 on the same 28 cases and prints the
  results at 17 digits (`plain.txt`);
- the same driver linked with copies of lmder.f and lmpar.f into which
  `build.sh` inserts write statements (with awk, visible in the script): one
  record per trial step, after label 290 (`trace.txt`). The script requires
  the traced build's results to equal the plain build's byte for byte.

`oracle/make_ref.py` turns `trace.txt` into `mgh_ref.viv`, and `check.sh`
regenerates it and requires the committed file to be identical.

**scipy.** `oracle/scipy_check.py` writes the eleven problems again in numpy
from the paper and the driver, runs `scipy.optimize.leastsq` (scipy 1.12,
which calls MINPACK's lmder) with lmder1's settings, and compares with the
oracle.

## Validation

Produced by `lib/minpack/check.sh` (2026-09-25, Apple M2 Pro, gfortran
15.2.0, scipy 1.12.0; the Viv columns of the exp and atan problems again on
2026-10-07, after Viv's `exp` became correctly rounded). The Viv columns are
from `mgh_check.viv`; the interpreter and the native build print the same
bytes.

| prob | problem | start | info | nfev | njev | Viv x vs Fortran (rel.) | Viv history vs Fortran | scipy vs Fortran |
|---|---|---|---|---|---|---|---|---|
| 4 | Rosenbrock | 1 | 4 | 21 | 16 | 0 | bit for bit | same counts, x equal |
| 4 | Rosenbrock | 10 | 2 | 8 | 5 | 0 | bit for bit | same, x equal |
| 4 | Rosenbrock | 100 | 2 | 6 | 4 | 0 | bit for bit | same, x equal |
| 5 | helical valley | 1 | 2 | 11 | 8 | 0 | 8.9e-16 | same, x equal |
| 5 | helical valley | 10 | 2 | 20 | 15 | 0 | bit for bit | same, x equal |
| 5 | helical valley | 100 | 2 | 19 | 16 | 0 | 2.6e-15 | same, x equal |
| 6 | Powell singular | 1 | 4 | 59 | 58 | 0 | bit for bit | rounding-sensitive: 58/57 evaluations |
| 6 | Powell singular | 10 | 4 | 72 | 71 | 0 | bit for bit | rounding-sensitive: 62/61 |
| 6 | Powell singular | 100 | 4 | 68 | 67 | 0 | bit for bit | rounding-sensitive: 65/64 |
| 7 | Freudenstein-Roth | 1 | 1 | 14 | 8 | 0 | bit for bit | same, x to 8.6e-15 |
| 7 | Freudenstein-Roth | 10 | 1 | 19 | 12 | 0 | bit for bit | same, x to 1.3e-15 |
| 7 | Freudenstein-Roth | 100 | 1 | 24 | 17 | 0 | bit for bit | same, x to 6.4e-15 |
| 8 | Bard | 1 | 1 | 6 | 5 | 0 | bit for bit | same, x to 2.0e-15 |
| 8 | Bard | 10 | 1 | 37 | 36 | 0 | bit for bit | same, x to 3.8e-15 |
| 8 | Bard | 100 | 1 | 14 | 13 | 0 | bit for bit | same, x to 1.6e-15 |
| 9 | Kowalik-Osborne | 1 | 1 | 18 | 16 | 0 | bit for bit | same, x to 2.5e-15 |
| 9 | Kowalik-Osborne | 10 | 1 | 78 | 70 | 0 | bit for bit | same, x to 8.0e-13 |
| 9 | Kowalik-Osborne | 100 | 5 | 500 | 380 | 0 | bit for bit | rounding-sensitive: 500/381 |
| 10 | Meyer | 1 | 2 | 126 | 116 | 1.9e-13 | 1.0e-10 | rounding-sensitive: info 3 |
| 10 | Meyer | 10 | 5 | 400 | 346 (Viv: 345) | 0.25 | 1 | rounding-sensitive: 400/354 |
| 12 | Box 3-D | 1 | 2 | 7 | 6 | 0 | bit for bit | same, x equal |
| 13 | Jennrich-Sampson | 1 | 1 | 21 | 12 | 0 | bit for bit | same, x to 5.7e-14 |
| 17 | Osborne 1 | 1 | 1 | 18 | 15 | 2.6e-14 | 5.4e-13 | same, x to 9.4e-14 |
| 18 | Osborne 2 | 1 | 1 | 16 | 12 | 1.6e-16 | 7.9e-14 | same, x to 3.7e-16 |
| - | NIST Thurber (the post) | NIST start 1 | 1 | 33 | 27 | 0 | bit for bit (the norm at every step) | same, x to 1e-13 |

How to read it:

- info, nfev and njev are the Fortran's, and the Viv port's are identical in
  every case but one, with the same number of trial steps and the same
  accept or reject decision at each. The exception is Meyer from 10 times
  the start (below): both stop at maxfev, Viv after 345 Jacobians.
- "Viv history vs Fortran" is the largest difference, over every recorded
  number of every trial step (the radius lmpar was given, par, the step
  length, the trial norm, the actual and predicted reductions, the ratio,
  the new radius and par, the decision, the norm, gnorm and x), divided by
  the largest magnitude in that column for that case. The ratio is compared
  through the actual reduction it implies, since near convergence it is the
  quotient of two numbers at the rounding level.
- The 15 cases whose functions use only arithmetic and square roots
  (problems 4, 6, 7, 8, 9), and the post's, are bit for bit: the same
  operations on the same numbers. The rest call exp or atan, which Viv
  computes with its own libm and gfortran with the system's, so their
  inputs can differ in the last bit. Since Viv's `exp` is correctly rounded
  (release 0.0.1), Box 3-D and Jennrich-Sampson match the Fortran bit for
  bit, and the others stay at rounding level, except Meyer from 10 times the
  start: a 400-step run along a badly scaled valley that stops at maxfev in
  both. Its path depends on the last bits of exp (with musl's exp the two
  ended 0.3% apart; now 25%), so for it the check asks only what it asks of
  scipy's six below: that both stop the same way, at maxfev.
- scipy: in 18 of the 24 cases scipy's lmder stops with the same info,
  nfev and njev, and x agrees to `1e-12` or better. The other six take
  different paths because their path depends on the last bits of the
  arithmetic. A gfortran `-O2` build of the same Fortran, which fuses
  multiply-adds, departs from the `-O0` build on five of them (Powell
  singular from all three starts, Kowalik-Osborne and Meyer from 100 and 10
  times the start; with `-O2`, Powell singular from 10 times the start runs
  to maxfev and Kowalik-Osborne from 100 times converges). On the sixth,
  Meyer from the standard start, scipy and the `-O0` build differ only in
  which tolerance test fires first (info 3 against 2), with x `1e-9` apart.
  For these six the check asks only that both stop the same way: the same
  minimum, or maxfev.

The post's fit, NIST Thurber from start 1: residual sum of squares
5642.70826 against NIST's certified 5642.7082397 (the fit stops at
`xtol = 1.5e-8`), and every parameter within 0.001 standard deviations of
its certified value. Every number is asserted by a test in the
post.

## Invariants the debug lens holds

`lmder.debug` and `lmpar.debug`, checked in both stories (189 and 161
checks, none failing) and in all 24 cases of `mgh_check.viv` (11,673 checks
under `viv story`, none failing):

- the trust radius stays positive, and par is never negative;
- the predicted reduction is not negative;
- a step the test will accept (`ratio >= LM_RATIO_ACCEPT`, 1e-4) lowers the norm;
- a poor step (`ratio <= LM_RATIO_SHRINK`) at least halves the radius and
  leaves it at most 5 times the step's length; a good step
  (`ratio >= LM_RATIO_EXPAND`) or a Gauss-Newton step sets it to twice the
  step's length and halves par;
- lmpar's lower bound is not negative, and its step is at most 10% longer
  than the radius, unless it stopped at its 10-iteration limit;
- a step with par > 0 has `|D p|` within a tenth of delta (the secular
  equation, solved loosely), unless lmpar ran out of iterations or had no
  lower bound (the two exits below).

The intent's equations cite these holds, and the explorer's reading view
shows each hold's result in the recorded run beside the equation it checks.
A hold may call only pure builtins, not a def, so the damped normal
equations `(J^T J + par D^T D) x = J^T f` are checked by a test in
`lmpar.viv` instead, with pivoting and scaling, for five radii.

The request to hold `|pnorm - delta| <= 0.1 delta` whenever par > 0 is not
an invariant: lmpar also stops, with par > 0, when it has no lower bound
(`parl = 0`, a rank-deficient J) and the step is already inside the region,
and after 10 iterations. The hold that is kept states what is true in all
three exits.

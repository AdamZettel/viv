# SISSO in Viv

`sisso.viv` is a port of SISSO 3.5 (Runhai Ouyang, github.com/rouyang2017/SISSO
@ bc18cae, Apache-2.0) for regression. `python/vipyr/sisso.py` calls it:

```python
from vipyr import sisso
r = sisso.fit(df, target="y", units={"a": "m", "b": "m", "c": "s"}, rung=2, dim=2,
              ops=["+", "-", "*", "/", "^2", "sqrt"])
r.models[-1].descriptors, r.models[-1].coefficients, r.models[-1].rmse
r.predict(df)
```

## What the port computes, and how it matches the Fortran

The steps are the Fortran's, in its order:

- **Feature construction** (FC.f90 `combine`): each rung applies the unary
  operators to the newest features and the binary ones to each pair of a
  newest feature with every stored feature. A sum or difference needs equal
  units. The Fortran's pruning rules are kept: no exp(log), no sqrt(x^2),
  no A/(B/C) before the last rung, and its name test for A*B/B, which runs
  on the rendered names.
- **Duplicates** (`dup_scheck`): new features of a rung are inserted by
  bisection into a list ordered by `sqrt(sum((tag + x)^2))`; two features
  whose keys agree to 1e-8 and whose correlation exceeds 0.99 keep the
  simpler one. The last rung is not stored and not checked.
- **SIS**: every kept feature is scored by its correlation with the target
  (later dimensions: with the residual); a buffer of
  `2 nf_sis + min(nf_sis, 50000)` is pruned to the best `2 nf_sis` when full,
  which raises the score threshold, as `update_select` does.
- **L0** (DI.f90 `model`): every descriptor of dimension d from the subspace,
  in lexicographic order, fit by classical Gram-Schmidt on the centered
  columns (`orth_de`); the smallest RMSE wins. The subspace is first
  rounded to ten significant digits, which is what the Fortran's
  `e20.10` write and read of `SIS_subspaces/Uspace.dat` does.

Formulas are trees of integer codes. Names are built (as character codes)
only where the Fortran compares names: the A*B/B test and ties in
`simpler`. The tie break by name length uses lengths computed from the
tree.

The Fortran adds its sums in loops, left to right. The port writes them as
`sum(x, ordered=true)` and `dot(u, v)`, which add in that order (a plain
`sum(x)` adds in NumPy's pairwise order and would differ in the last
digits), and the fit of each descriptor as `b @ XP`, whose entries add the
terms in the order of the rows. Each sum over a feature takes a slice of the
flat feature array, which copies it: the ordered forms made the `timing` case
about 1.4 times slower than the loops they replaced (measured on a loaded
machine). The compiled engines now read such a slice in place
(lang/viv/src/fuse.rs): on 2026-10-13, at a load average near 13, the
`timing` case took 4.7 s against 5.1 s with the slices copied, and the
results hash the same. Most of the rest of `timing` is the fit `b @ XP`.

gfortran's `maxval` and `minval` skip NaN. A feature with some NaN values
(the cube root of a negative difference) is therefore kept and counted, and
its SIS score is 0. The port does the same; without it, the last rung of
the `fcheavy` case had 26 fewer features.

Left out: classification (ptype=2), multi-task (ntask > 1), the sign
constraint (scmt), MaxAE as the selection metric, L1L0, restarts, and a
per-rung operator set. Names longer than the Fortran's 150 characters are
not truncated. On an exact tie in RMSE the port keeps the first descriptor,
while the Fortran's top-`nmodel` slots could, in principle, pick another.

## The Fortran as the oracle

`compare_fortran.py` runs the Fortran and the port on five cases and
requires the same size of every Phi_k at every dimension, the same
subspace (names, in order), the same descriptors, and the coefficients,
intercept, RMSE and MaxAE to the Fortran's ten printed digits. It runs the
Fortran once per dimension, because `SISSO.out` prints only the last
dimension's model. `--record` writes `fortran_reference.json`, which
`python/tests/test_sisso.py` reads.

Building the Fortran with gfortran 15 and Open MPI 5 took these changes to
a copy of `src/` (the originals are written for Intel Fortran):
`a.b` component access to `a%b`; `include 'mpif.h'` after `implicit none`
(`use mpi` checks argument ranks the code does not meet) and, in
libsisso.f90, inside `mtlasso_mpi` with its own `status`; the literals
passed to `combine` as `integer*8`; `ifport`'s `makedirqq` and
`delfilesqq` replaced by `execute_command_line`; `line_verylong*(line_len)`;
no assignment to the unallocated `nsample` in `initialization`; and the
`iFCDI>1 .and. nf_sis_avai(iFCDI-1)` test split in two. Linked without
`-ld_classic`. It runs as one process (`ulimit -s 65520`; `TMPDIR` set to a
writable directory for Open MPI's shared memory).

## Results on this machine (shared with other jobs while it ran)

| case | primaries, samples | operators | complexity, dim | Phi sizes | Fortran s | Viv s |
|---|---|---|---|---|---|---|
| example (SISSO's template) | 3, 5 | + - * / | 3, 2 | 3, 15, 335 | 0.13 | 0.006 |
| synthetic | 5, 50 | + - * / ^2 sqrt exp | 3, 2 | 5, 56, 5286 | 0.15 | 0.03 |
| deep | 3, 30 | + - * / ^2 | 4, 2 | 3, 18, 232, 6423 | 0.14 | 0.03 |
| fcheavy | 10, 100 | 13 operators | 3, 2 | 10, 257, 112864 | 0.46 | 0.30 |
| timing | 7, 200 | 11 operators | 3, 3 | 7, 124, 24398 | 7.2 | 3.4 |

Times are wall seconds, best of three, the Viv library warm; the Fortran's
include about 0.1 s of MPI start-up. In `timing` the 3D search (955,860
least-squares fits) is most of the time: the port reuses the Gram-Schmidt
columns a descriptor shares with the previous one, the Fortran recomputes
them. Every case agrees on every check; `deep`, `fcheavy` and `timing`
recover the formula their data was generated from.

`tools/export/check.py study/ports/sisso/sisso.export.toml` holds the
exported library, the native build and the interpreter to the same bits on
the cases of `sisso.cases.json` (`sisso.exact_hashes.json`). Integer results
leave the program as float64 arrays: the export's report prints `Arr` results
but not `[Int]` ones.

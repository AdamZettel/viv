# lib/randla: randomized numerical linear algebra in Viv

Multiply a large matrix by a random one with far fewer columns, and the
product keeps most of what you want to know about the large one: its
dominant range, its trace, the geometry of its column space. This library is
built on that idea, following the split in the RandBLAS/RandLAPACK design
(Murray, Demmel, Mahoney, Dongarra and others, arXiv:2302.11474): sketching
operators at the bottom, drivers on top.

```sh
viv run lib/randla/lowrank.viv                    # each module's own tests
tools/three-engines.sh lib/randla                 # interpreter, C and wasm print the same bytes
VIV=target/release/viv python3 lib/randla/check_ref.py      # against NumPy, SciPy, scikit-learn
VIV=target/release/viv python3 lib/randla/example_rsvd.py   # randomized_svd(df_or_array, k) from Python
```

## The modules

| module | what it has |
|---|---|
| `philox` | Philox4x32-10 (Random123's known-answer vectors pass), uniforms, Box-Muller normals, signs |
| `sketch` | `gaussian`, `rademacher`, `saso_apply` (sparse sign), `srht_apply` (subsampled randomized Hadamard) |
| `dense` | Householder QR, CholeskyQR2 with a Householder fallback (`orth`), triangular solves |
| `lowrank` | `range_finder`, `qb`, `rsvd`, `nystrom`/`nystrom_fn`, `rpcholesky`/`rpcholesky_fn` |
| `lstsq` | `sap_lstsq` (sketch-and-precondition with LSQR), `qr_lstsq` (the direct reference) |
| `trace` | `hutchinson`/`hutchinson_fn`, `hutchpp`/`hutchpp_fn` |
| `api` | typed wrappers that `vipyr.load` exports to Python |

The `_fn` forms take the matrix as a def: `apply(X)` returns A X, and
`column(s)` returns column s. A lambda that captures a matrix works, so
`hutchpp_fn(fn(X) -> A @ X, n, m, seed, stream)` is `hutchpp(A, m, seed, stream)`.

## Reproducibility

Every random number comes from Philox keyed by the seed, at a counter that
names the entry: entry (i, j) of a `rows`-by-`cols` Gaussian operator is
normal number (i * cols + j) mod 2 of Philox block (i * cols + j) / 2 of its
stream. Nothing depends on the order in which entries are drawn, and a new
`stream` gives an independent operator from the same seed. The same call
gives the same sketch in the interpreter, in C and in wasm (`tools/three-
engines.sh lib/randla` compares their output byte for byte), and
`check_ref.py` rebuilds the Rademacher operator in NumPy from its own Philox,
bit for bit. The Gaussian operator agrees with NumPy's rebuild to 2 ulps,
from the platform's `log`, `cos` and `sin`, which Viv does not use.

## Choosing a driver

**`rsvd`** for the leading k singular triplets of a matrix too large for a
full SVD, when the singular values decay. Two power iterations (the default)
bring the error within a fraction of a percent of the best rank-k error even
for 1/j decay. It does not help when the spectrum is flat: with no gap after
the k-th singular value, no rank-k approximation is good, randomized or not.
For a matrix small enough that `svd` takes milliseconds, call `svd`.

**`nystrom`** for a symmetric positive semidefinite matrix (a kernel matrix,
a covariance, a Hessian of a convex loss): one product A Omega instead of the
range finder's two, with eigenvalues that never overshoot. Not for an
indefinite matrix; its error bound assumes A is PSD.

**`rpcholesky`** when reading an entry of A is the expensive part (a kernel
evaluated on data points): k pivots read k columns and the diagonal of A. The trace error tracks the best rank-k trace error within a modest
factor. When A is cheap to multiply, Nystrom from a Gaussian sketch is just
as good and uses dense products.

**`sap_lstsq`** for a tall least-squares problem, m much larger than n, to
full accuracy: the sketch's R makes A R^-1 well-conditioned (condition near 3
for the default d = 4n), so LSQR needs about 30 iterations for a relative
tolerance of 1e-14 whatever the condition of A. It pays once m n^2 dominates
the cost; for an m-by-n problem LAPACK factors in milliseconds, `qr_lstsq`
or `numpy.linalg.lstsq` is simpler. SRHT and SASO cost m log m and nnz * m n
to apply; the Gaussian sketch costs a d-by-m product and is the slowest.

**`hutchpp`** for the trace of an operator you can only apply, when its
spectrum decays: on 1/j^2 eigenvalues its error falls like m^-2 against
Hutchinson's m^-1/2. **`hutchinson`** when the spectrum is flat (A near a
multiple of I), where Hutch++'s deflation finds nothing worth a third of the
budget. Neither makes sense for a matrix you hold: `trace(A)` is exact.

## Against the references

`check_ref.py`, one core for NumPy (Apple M-series laptop, shared with other
jobs; times are the best of 3 and move by tens of percent between runs).
"viv" is the exact C build vipyr compiles, "fast" the fast build (FMA,
Accelerate for `@`); sklearn is `randomized_svd` with the same oversampling,
2 power iterations and QR normalization.

| rSVD | k | spectrum | viv / optimum | sklearn / optimum | viv ms | fast ms | sklearn ms |
|---|---|---|---|---|---|---|---|
| 500 x 300 | 10 | 1/j | 1.00006 | 1.00011 | 8.6 | 3.2 | 1.9 |
| 1000 x 600 | 20 | 1/j^2 | 1.00002 | 1.00016 | 36.3 | 10.4 | 8.7 |
| 2000 x 1000 | 20 | exp(-j/5) | 1.00000 | 1.00000 | 113 | 26.8 | 27.9 |
| 4000 x 2000 | 50 | 1/j | 1.00373 | 1.00397 | 645 | 159 | 177 |

"optimum" is the Frobenius error of the truncated SVD (Eckart-Young). The
test requires viv's error within 5% of it, and the leading singular values
within 1e-3 of the largest.

Least squares, 3000 x 60 with condition number 1e6, against
`numpy.linalg.lstsq`: residual norms agree to 4e-15 to 1.5e-14 relative
(required: 1e-13), solutions to 7e-10 to 3.7e-9 relative (required: the
least-squares perturbation bound at unit roundoff, 6e-8), 33 or 34 LSQR
iterations for every sketch. In time it loses to LAPACK at this size: 44 ms
(SRHT), 50 ms (SASO) and 144 ms (Gaussian) against 3 ms.

`example_rsvd.py`, a 3000 x 400 pandas frame driven by 15 factors, k = 15:
error equal to the optimum to 6 digits for both; 52 ms exact, 10 ms fast,
sklearn 14 ms.

Trace of a 400 x 400 PSD matrix with eigenvalues 1/j^2, mean relative error
over 8 seeds:

| products | Hutchinson | Hutch++ |
|---|---|---|
| 12 | 1.3e-1 | 2.5e-2 |
| 48 | 5.8e-2 | 2.9e-3 |
| 192 | 3.7e-2 | 7.4e-5 |

RPCholesky on a Gaussian kernel (500 points, bandwidth 0.05): trace error
2.0, 3.8 and 22 times the best rank-k trace error at k = 10, 20, 40; at
k = 40 the error (9.7e-5) is below the best rank-20 error (1.4).

## References

- N. Halko, P.-G. Martinsson, J. A. Tropp, "Finding structure with
  randomness", SIAM Review 53 (2011).
- P.-G. Martinsson, J. A. Tropp, "Randomized numerical linear algebra:
  foundations and algorithms", Acta Numerica 29 (2020).
- R. Murray, J. Demmel, M. W. Mahoney, J. Dongarra and others,
  "Randomized numerical linear algebra: a perspective on
  the field with an eye to software", arXiv:2302.11474.
- R. A. Meyer, C. Musco, C. Musco, D. P. Woodruff, "Hutch++: optimal
  stochastic trace estimation", SOSA 2021.
- Y. Chen, E. N. Epperly, J. A. Tropp, R. J. Webber, "Randomly pivoted
  Cholesky", arXiv:2207.06503 (2022).
- H. Avron, P. Maymounkov, S. Toledo, "Blendenpik: supercharging LAPACK's
  least-squares solver", SIAM J. Sci. Comput. 32 (2010).
- C. C. Paige, M. A. Saunders, "LSQR", ACM TOMS 8 (1982).
- T. Fukaya, Y. Nakatsukasa, Y. Yanagisawa, Y. Yamamoto, "CholeskyQR2",
  ScalA 2014.
- J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel random
  numbers: as easy as 1, 2, 3", SC11.

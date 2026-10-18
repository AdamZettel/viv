# Deploying a scikit-learn pipeline as one C file

`vipyr compile` turns a fitted scikit-learn pipeline, plus the Viv
functions that compute its features, into one C99 source file:

```sh
vipyr compile model.pkl --data holdout.csv --features features.viv -o churn.c
cc -O2 -ffp-contract=off churn.c -o churn       # libc only, nothing else to install
./churn --selftest                              # re-checks embedded rows against scikit-learn's outputs
./churn < transactions.csv > predictions.csv
```

The file is the artifact. It holds the whole pipeline (feature code,
scalers, encoders, model) and a few hundred rows with scikit-learn's
outputs for them. Anyone with a C compiler can rebuild it and check that
their build still gives those outputs. m2cgen and ONNX export only the
estimator. Here the custom features are in the same verified file.

`deploy.py` runs the whole example: it trains the model, compiles it, checks
it and times it.

## The example

Card transactions (synthetic): `card`, `ts` (int64 nanoseconds), `amount`,
`merchant` (a string), `declined`, `chargeback`. The label is whether the
holder closes the card within 30 days (synthetic too).

1. Features: `card_features` from
   [card_velocity/features.viv](../datasci/card_velocity/features.viv) gives
   `n_prev_1h` (the card's transactions in the hour before) and `status`
   (ok, watch or locked). Training calls the same Viv file through
   `vipyr.load`, so the model is fitted on the features the C file
   computes.
2. A `ColumnTransformer`: `StandardScaler` on `amount` and `n_prev_1h`,
   `OneHotEncoder` on `merchant` and `status`.
3. A `GradientBoostingClassifier`, 150 trees of depth 3.

`vipyr compile` writes `churn.viv`, the pipeline as Viv you can read, then
`viv export` makes it C and adds the command line. The input schema is
part of the file: `./churn --schema` prints it. A missing or unexpected
column, a cell that does not parse, or a row the pipeline refuses (a NaN
amount, which scikit-learn's scaler rejects) ends with a message and a
nonzero exit status.

## Results

Apple M2 Pro, scikit-learn 1.4.1, clang 17 and GCC 15, machine load 7.6
from other work (`results.json` has every number):

| | |
|---|---|
| churn.c | 441,847 bytes, 11,827 lines (the model's trees as code, and Viv's runtime); churn.viv 5,882 lines |
| build | clang 0.6 s, gcc-15 2.1 s; no warnings with `-std=c99 -pedantic -Wvla` |
| `--selftest` | passes under both: 300 rows, 900 values |
| 30,000 held-out rows | labels identical; 21 of 90,000 values (17 rows) differ from scikit-learn's by 1 ulp |
| batch scoring, in process | C 1,960,000 rows/s (features included); scikit-learn `predict_proba` 1,010,000 rows/s (972,000 with the Viv features) |
| one row, in process | C 19 µs (2.8 µs of it in C, the rest reading the DataFrame); scikit-learn 850 µs |
| command line, CSV in and out | 577,000 rows/s |

m2cgen, ONNX and onnxruntime are not installed on this machine, so they
were not timed. Every output is the same bits as the earlier version of this
file, which walked node tables (checked on all 30,000 rows).

### How the trees run

Each tree is a def of nested `if`s whose thresholds are constants, and it
takes only the features it reads:

```
def _tree0(z0: Num, z1: Num, z10: Num) -> Num {
    if z10 <= 0.5000000298023224 {
        if z1 <= 0.4003589302301407 {
            ...
```

The score loop calls the trees in order and adds each leaf's value with the
same `fma` as before, so a row's sum is formed in the same order. Three
things set the speed; each was measured on this model (30,000 rows, the C
call alone):

| change | C call |
|---|---|
| node tables, bounds-checked loads, Viv's integer `fma` (before) | 98 ms |
| `fma` as the hardware instruction where the target has one | 50 ms |
| the same, with the tables' loads unchecked (a hand edit, for comparison) | 23 ms |
| trees as code, the transformers and trees in one row loop | 13 ms |

`fma` is exactly rounded by IEEE 754, so the instruction gives the integer
code's bits; 50 million random and edge-case triples (zeros, infinities,
NaNs, subnormals, near-cancellations) agree to the bit.

Models of more than 50,000 nodes run in blocks of 128 rows: each tree runs
over the block before the next tree starts, so one tree's code stays in the
instruction cache. Each row still adds the trees in order. Above 250,000
nodes the compiler writes node tables instead (`--trees tables` forces
them, `--trees branches` the code), since the C grows with the node count.

Scoring time by model, 10 float columns, in process through `load_model`
(µs for one call; load 3 to 9, the HistGradientBoosting row at 9.3):

| model | nodes | C file | build | 1 row | 100 rows | 30,000 rows | 1,000,000 rows |
|---|---|---|---|---|---|---|---|
| GradientBoosting, 50 trees, depth 3 | 750 | 0.3 MB | 0.6 s | 21 / 239 | 41 / 297 | 7,400 / 18,500 | 243,000 / 596,000 |
| GradientBoosting, 500 trees, depth 3 | 7,350 | 0.8 MB | 3.5 s | 26 / 315 | 205 / 734 | 58,000 / 113,000 | 1,970,000 / 3,730,000 |
| GradientBoosting, 50 trees, depth 8 | 18,978 | 2.1 MB | 1.6 s | 21 / 234 | 52 / 416 | 20,200 / 43,100 | 661,000 / 1,412,000 |
| GradientBoosting, 500 trees, depth 8 (blocks) | 115,484 | 14 MB | 15 s | 31 / 342 | 544 / 1,737 | 191,000 / 290,000 | 6,330,000 / 9,710,000 |
| RandomForest, 100 trees, depth 8 | 31,120 | 4.1 MB | 4.5 s | 22 / 1,131 | 106 / 1,657 | 61,000 / 114,000 | 2,040,000 / 3,860,000 |
| HistGradientBoosting, 200 iterations | 12,200 | 1.4 MB | 2.1 s | 22 / 12,800 | 98 / 12,500 | 46,000 / 45,000 | 1,500,000 / 1,076,000 |

Each cell is C / scikit-learn. "build" is `compile_model`: writing the Viv,
`viv export`, building the shared library and checking every held-out row.
HistGradientBoosting's `predict_proba` runs on all 12 cores; the C runs on
one. The same models as node tables take 2.8 to 4.8 times as long as the
code in batches of 30,000 rows; outputs are identical either way
(`test_branches_and_tables_agree`).

### Where the 1-ulp differences come from

The class probability is `1 / (1 + exp(-raw))`. Viv's `exp` is correctly
rounded and gives the same bits on every platform. scikit-learn calls the
platform's `exp`, which macOS rounds the other way in rare cases (for
example `exp(1.6880048295554648)`: macOS gives 5.4086786969106715, the
correctly rounded value is 5.408678696910672). The raw scores, the labels
and every output that does not go through `exp` are bit-identical.

So the self-test checks two things. Each value must be within the
embedded tolerance of scikit-learn's (4 times the largest difference seen at
compile time, here 4.4e-16), and it must equal the file's own compile-time
output to the bit. The second check holds on any machine, because Viv
computes with its own math library and without contraction.

## What is exact, by family

`python python/tests/test_compile.py` prints this table (800 held-out rows,
mixed numeric, integer and string columns):

| model | bit-identical to scikit-learn | otherwise |
|---|---|---|
| DecisionTree, RandomForest, ExtraTrees (classifier and regressor) | yes | |
| GradientBoostingRegressor, HistGradientBoostingRegressor | yes | |
| GradientBoostingClassifier, HistGradientBoostingClassifier | raw scores and labels yes | probabilities within 1 ulp, through `exp` |
| LogisticRegression (binary, multinomial) | labels yes | within 2.2e-16: scikit-learn's dot product is BLAS's (OpenBLAS here), summed in its own order; the file sums left to right |
| LinearRegression, Ridge | no | the same BLAS order; the difference grows with the size of the coefficients |

How the trees match:

- scikit-learn casts X to float32 before comparing it with a node's
  threshold. `float32(x) <= t` is the same test as `x <= T` in float64 for
  one bound T per node (the midpoint between two float32 values, adjusted
  for ties to even), so the file stores T and compares doubles.
  `test_le_bound_is_the_float32_comparison` checks the bound at the
  midpoints and one ulp either side.
- On arm64, scikit-learn's compiled gradient-boosting loop fuses
  `raw += learning_rate * value` into one rounding. The compiler detects
  this per model by computing both ways on held-out rows. The Viv source
  then uses `fma`, a builtin added for this, which rounds once in every
  engine (C: musl's integer fma, added only to programs that call it, so no libm is needed).
- Forests add the trees in order and divide by the count, as `predict_proba`
  does with `n_jobs=None`. With `n_jobs > 1` scikit-learn adds them in the
  order its threads finish, so its own last bits vary from run to run.

## Supported

Pipeline and ColumnTransformer (with `remainder` drop or passthrough);
StandardScaler, MinMaxScaler (with `clip`), SimpleImputer (mean, median,
most frequent, constant; NaN or a numeric missing marker); OneHotEncoder on
strings or numbers, with `drop`, and `handle_unknown` `error` (the row is
refused) or `ignore` (all zeros); the estimators in the table above, and
HistGradientBoosting with missing values. Feature functions: each public
def of the `--features` module whose parameters are `Arr` or `[Int]` input
columns and whose result is an array or a record of arrays, run on the whole
batch before the pipeline.

## Limits

- The compiler refuses with a message: sparse ColumnTransformer output (set
  `sparse_threshold=0`), OneHotEncoder with infrequent categories,
  SimpleImputer with `add_indicator`, HistGradientBoosting with categorical
  features, one-vs-rest LogisticRegression with more than two classes,
  multi-output models, other estimators and transformers.
- Feature functions see the whole input at once. card_features needs rows
  sorted by card and time, and stops with a message otherwise.
- A row is refused when scikit-learn refuses a NaN or infinity in that
  column (found by asking scikit-learn at compile time). A NaN that appears
  in the middle of the pipeline is not checked.
- The file uses Viv's runtime, which includes `<complex.h>` and
  `_Complex` for complex numbers, `setjmp`/`longjmp`, and compound literals.
  The model code itself has no VLAs and no complex numbers. MSVC's C
  compiler has no `_Complex` (it has `_Dcomplex`), so this file does not
  build with MSVC yet (clang-cl and MinGW GCC are untested).
- Linear models agree only to the stated tolerance (see above).

## Python

```python
from vipyr.compile import compile_model, load_model
report = compile_model(pipe, holdout_df, "churn.c", features="features.viv")
m = load_model("churn.c")         # builds churn.c as a shared library once (cached), ctypes
m.predict_proba(df)               # columns by name; strings mapped to the encoder's codes
```

`compile_model(..., tree_code="tables")` (or `vipyr compile --trees tables`)
writes node tables instead of code.

The C entries are `churn_predict`, `churn_predict_proba` and
`churn_predict_all`. Each takes the float columns as a row-major
`rows x NF` array of doubles and the integer columns and category codes as a
row-major `rows x NI` array of int64 (`churn_category_code(col, "travel")`
gives a code). `churn_schema_json()` describes both.

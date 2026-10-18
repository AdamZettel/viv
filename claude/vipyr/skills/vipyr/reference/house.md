# House style for Viv code

The house style of the Viv repository, condensed.

- **Intent.** A def's purpose in `#|` lines above it: what it computes, in
  one plain sentence. How it does it, when needed, in `#||` lines.
- **Settings.** Every constant a person might change is a top-level name
  with a `#:` line saying what it is, with units. `viv run --set NAME=value`
  changes it for one run.
- **Steps.** Inside a def, short `#:` headings say what each step is for,
  not what the line does. A def over about 40 code lines is split into
  steps or into defs.
- **Names.** Lower case with underscores for values and defs; CamelCase for
  records and enums; module prefixes after an import (`mix.pulay`).
- **Options.** Named at the call: `sum(X, axis=-1)`, `fit(X, centered=true)`.
  `viv fix --keywords` names positional ones.
- **Sizes and thresholds.** No magic integers: sizes come from `len` or the
  shape, tolerances and counts are named constants (`N_GRID: Int = 64`).
- **Loops.** One `for` and one `}` per line. A loop that only builds one
  array from another is a comprehension.
- **Braces.** Stroustrup: `}` ends its line, `else {` starts the next.
- **Gathers out of loops.** Copy a row or a slice once before the inner
  loop, not on every pass.
- **Production and verification.** The fast method is the program; the
  slow, plain method lives beside it as a contract method or a test, so
  `using every method` checks one against the other.
- **Tests.** Every program ends with `test "..." { ... }` blocks that pin
  its results against an independent value: a closed form, a published
  number with its citation, or another method. The test name states the
  check in words. Assertions: `assert_near(a, b, tol)`,
  `assert_allclose(A, B, tol)`, `assert_true(c)`, `assert_lt(a, b)`,
  `assert_gt(a, b)`.
- **Layout.** At most 100 columns.

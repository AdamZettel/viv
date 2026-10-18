---
name: vipyr
description: Do computational science with vipyr and Viv. vipyr has validated engines (power flow, Skyrme Hartree-Fock, Hartree-Fock, DFT atoms, ODEs, MINPACK fits, option pricing, NAS kernels and more), each checked against a named reference and measured against plain Python where a comparison exists. Use when the user mentions vipyr, viv, a .viv file, a claim, or asks to compute something one of those engines solves, compare solvers, or check a numerical claim.
when_to_use: When the user says to work with vipyr or viv, asks whether a solver already exists, asks whether method A converges in fewer passes than B, asks for timings beside NumPy, or edits or runs .viv files.
allowed-tools: Bash(vipyr help *) Bash(vipyr catalog *) Bash(viv help *) Bash(viv features *) Bash(viv explain *) Bash(viv map *) Bash(viv whatis *) Bash(viv src *) Bash(viv lsfunc *) Bash(viv check *) Bash(viv claims --json*)
---

# Working with vipyr and Viv

vipyr is the front door (`pip install vipyr`, `import vipyr as vp`); Viv is
its language and `viv` the language's tool. Every command below takes
`--json` and then prints one object on stdout:
`{"viv":1,"cmd","ok","exit","diagnostics","result","truncated","more"}`.

## First: what already exists

1. `vipyr catalog --json --summary` (or `--topic power|nuclear|chemistry|ode|optimize|finance|...`).
   Each engine lists what it solves, the reference it was checked against
   and the agreement, its claims, its code map command, and, where one was
   measured, the timings of the plain Python/NumPy/SciPy way beside Viv's
   (and line counts where counted). `viv_slower: true` marks entries where
   the Viv time is the larger one. Quote both numbers with the machine and
   load, and no comparative words: the reader compares.
2. If an engine fits, use it (`vipyr` call or `#! use` the Viv module); do
   not rewrite it. `viv map <file> --text` shows how it is built.
3. Otherwise `viv features --search <words> --json --summary`, then
   `viv features <name>`. The library often has the solver already.
4. `viv help --agent` prints the full working guide once per session.

## The loop

1. Write or edit the .viv file. The plugin's hook runs `viv check` after
   every edit and shows its errors.
2. `viv check <file> --json --summary`. Fix every diagnostic. When a
   diagnostic has `fix.edits`, apply exactly those edits (`viv fix
   --apply-json <saved envelope>` does it).
3. `viv run <file> --json --summary`. Read `result.tests` and the failures.
   `--set NAME=value` changes a `#:` setting; `--using CONTRACT=method`
   picks a method.

Exit codes: 0 ok, 1 a test failed, 2 usage, 3 does not compile, 4 run-time
error, 5 a limit stopped it, 6 refused by policy, 70 a viv bug.
On 5 (a time, memory or step limit), do not raise the limit on your own.
Say which limit stopped the run and ask the user. On 6, do not work around it: ask.
`viv run` is not in the allowed tools: the user approves each run.

## Writing Viv

- `viv features <name>` before using a builtin you have not used here.
- A def's purpose goes in `#|` lines above it: what it computes, in one
  plain sentence.
- Constants and settings get a `#:` line saying what they are, with units.
- Options are named at the call: `sum(X, axis=-1)`.
- No magic integers: sizes from `len`/shape, thresholds as named constants.
- One `for` and one `}` per line; Stroustrup braces (`}` ends its line,
  `else {` starts the next).
- Python habits that fail: `**` (write `^`), `range(n)` (write
  `for i to n`), `2x` (write `2 * x`), an array comparison as a condition
  (write `all(a == b)`), `assert` (write `assert_near(a, b, tol)`,
  `assert_true(c)`, `assert_lt(a, b)`).
- Every program ends with `test` blocks that pin its results against an
  independent value: a closed form, a published number with its citation,
  or another method (`using every method` runs every method of a contract).

## Checking a result by another route

- When a contract exists for the problem (`viv features <problem>` shows
  it), run every method; a disagreement is exit 1 with rule `C-disagree`.
- `viv digits <file>` says which printed digits survive reordered sums and
  random rounding. Quote only those digits.
- Compare against a reference you can cite, and cite only what you have
  read.

## Claims

A claim is a bounded statement with the program that checks it and its
vipyr twin. `viv claims --json` lists them with their status; see
`reference/claims.md` for the header and the rules. Never set `lane`
above `ai`; people review and promote claims.

## Writing about results

- Plain sentences: what was measured, on what, and the numbers.
- No adjectives that grade the result. Say what was not tested.
- American spelling. Never say people use something unless the user said so.

## More

- `reference/house.md`: the house style for Viv code.
- `reference/claims.md`: the claims header and the twin rule.
- `viv help rules`, `viv help exit-codes`, `viv help json`.

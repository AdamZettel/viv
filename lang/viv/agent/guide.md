vipyr and viv: one app with two sides. `vipyr` is the front door: the
catalog of validated engines, and every viv command. `viv` is the language's
tool: check, run, build .viv programs. Both share claims and runs. vipyr is
not on PyPI yet; from a checkout, use PYTHONPATH=python or
pip install --no-build-isolation ./python.

The loop:
  1. `vipyr catalog --json` (or --topic T): is the problem already solved here?
  2. `viv features --search <words> --json --summary`, then `viv features <name>`
  3. write or edit the .viv file
  4. `viv check f.viv --json --summary`: fix every diagnostic; apply fix.edits
  5. `viv run f.viv --json --summary`: read result.tests and the failures
  6. exit codes: 0 ok, 1 test failed, 2 usage, 3 compile, 4 run time, 5 limit,
     6 policy, 70 viv bug. On 5 or 6 ask the user; do not raise limits yourself.

{{catalog}}
Each engine above is checked against the named reference (`agreement`), has
claims that state what it was not tested on, and has a code map
(`viv map <file> --text`). Where timings were measured beside plain Python,
`comparison.speed` gives both numbers, `comparison.machine` the machine and
load, and `viv_slower` is true where the Viv time is the larger one.

Commands (`viv help <cmd>` for flags, `viv help <cmd> --json` for the schema):
  learn   viv features, viv explain <f> [--from def], viv map <f> [def] --text,
          viv whatis <f> <name>, viv lsfunc <f>, viv src <f> <def>
  write   viv check <f>, viv fix --apply-json <envelope.json>, viv lint <f>
  run     viv run <f> [--set NAME=value] [--using CONTRACT=method]
  record  viv claims [--json] [--only slug]
  agent   viv skill install, viv hook post-edit, viv help rules|exit-codes|json

With --json, every command above prints one object on stdout, the envelope:
  {"viv":1,"cmd","ok","exit","diagnostics":[{"rule","message","file","line",
   "col","notes","fix":{"message","edits":[...]}}],"result","truncated","more"}
--summary keeps counts and the first items (a run: the last 20 lines);
--max-lines N bounds each list (default 200; 0 for all); `more` names the flag
for the next part.

Python habits that fail in Viv (the rule id in brackets):
  x ** 2              write x ^ 2                                 [H-pow]
  for i in range(n)   write for i to n  (i = 0 .. n-1)            [H-range]
  2x                  write 2 * x                                 [H-juxtapose]
  if a == b (arrays)  write if all(a == b)  or any(a > 0)         [H-array-cond]
  assert x == y       write assert_near(x, y, tol), assert_true(c),
                      assert_lt(a, b), assert_allclose(A, B, tol)  [E-call-unknown]

A program, its test and its run:

```viv
#| the sum of the squares of the first n whole numbers
def sum_squares(n: Int) -> Num {
    total = 0.0
    for i to n {
        total = total + i ^ 2
    }
    return total
}

show sum_squares(4)

test "0 + 1 + 4 + 9 = 14, and n(n-1)(2n-1)/6 at n = 10 is 285" {
    assert_near(sum_squares(4), 14.0, 0.0)
    assert_near(sum_squares(10), 285.0, 0.0)
}
```

`viv run sq.viv` prints `14` and `PASS 0 + 1 + 4 + 9 = 14, ...`;
with --json, `result.tests` is {"pass": 1, "fail": 0, "failed": []}.

House rules for Viv code:
  - a def's purpose in `#|` lines above it, one plain sentence
  - constants and settings get a `#:` line saying what they are, with units
  - options named at the call: sum(X, axis=-1); no magic integers (sizes from
    len or shape, thresholds as named constants)
  - one `for` and one `}` per line; Stroustrup braces (`}` ends its line,
    `else {` starts the next)
  - every program ends with test blocks that pin its results against an
    independent value: a closed form, a published number with its citation,
    or another method (`using every method` runs every method of a contract)

Writing about results: plain sentences, the numbers and what they were
compared against, and what was not tested. No grading adjectives. Quote only
digits `viv digits <f>` says survive.

# Claims

A claim is a statement about a topic, specific and bounded, that a
program demonstrates. It is two files in the Viv repository:

- `claims/<topic>/<slug>.viv`: a fenced `###` header, then a Viv program
  that computes the quantities the statement names and asserts its bounds in
  `test` blocks.
- `claims/<topic>/<slug>.py`: its vipyr twin (optional), the same
  quantities through `import vipyr as vp`, the same bounds as PASS / FAIL
  lines (tools/claims/claimlib.py).

The header:

```
###
title: <the claim in a few words>
line: <its key numbers in one sentence>
claim: <the full statement: the measured result with its numbers and what
  it was compared against>
topic: <finance | math | stats | control | power | hardware | science | ...>
section: <a part of the topic>
reference: <the independent check: a package and version, a dataset, a
  published value with its citation, or an analytic result>
scope: <what was not tested>
lane: ai
by: claude
cost: fast
agree: 1e-12
###
```

Rules:

- A claim holds while both programs pass and agree: every value both
  `show` under the same name is equal within `agree` (relative; default
  1e-12). Values whose names start with `time` are not compared.
- Every number in `title:`, `line:` and `claim:` is a literal of a program
  or rounds to a value one shows; `tools/claims/run.py --text` checks them
  against the recorded runs. The recorded run's hash covers the programs and
  `agree:`, not the header's words, so editing a title needs no rerun.
- `lane: ai` for anything an agent writes. People review and promote it.
- `cost: fast` when the pair runs in about ten seconds; otherwise `heavy`.
- `viv claims --json` lists every claim and its recorded status
  (`pass`, `fail`, `mismatch`, `stale`, `unrun`).
- `viv claims run --only <slug>` runs one; `--write` records the result in
  claims/results.json. Run it only when the user asks: it builds the
  program natively.

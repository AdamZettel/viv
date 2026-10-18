---
title: Printing and formatting
order: 1
output: identical
---
- `print` puts a space between its arguments in both, and `sep=", "` (or any
  other text) replaces it.
- A bare float prints differently. Python's `print(x)` writes the shortest digits
  that read back as the same double; Viv's `print(x)` writes six significant
  digits, as C's `%g` does. Each language has the other's form: `fmt(x)` in Viv,
  `f"{x:g}"` in Python.
- f-strings are the same on both sides for these specs: `{x}`, `{x=}`,
  `{x:.3f}`, `{x:10.4f}`, `{word!r}`. The counts in this program are numbers, not `Int`s, so an integer is
  written with `:d` (`{count:d}`); without it, `f"{count}"` would be `3.0`.
- `for i in range(count)` is `for i to count`. A loop body is in braces instead
  of indented.
- Python's `assert` statements become assertions inside a Viv `test` block, which
  prints one `PASS formats` line.
